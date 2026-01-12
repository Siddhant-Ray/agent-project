"""
End-to-end: Dynamic LangGraph "trace replayer" driven by Annotator Metadata.Steps
✅ Dynamic number of steps (parsed from Annotator Metadata.Steps)
✅ For EVERY step: step input/output + timing
✅ For EVERY TOOL CALL inside a step: tool_call_logs with tool input/output + timing
✅ Total execution time
✅ Record + Replay via LangGraph SQLite checkpointer

Run vLLM:
  vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000

Install:
  pip install -U langgraph langchain langchain-openai langchain-community playwright lxml pymupdf pillow pytesseract duckduckgo-search
  playwright install chromium
  # system tesseract:
  #   macOS: brew install tesseract
  #   Ubuntu: sudo apt-get install -y tesseract-ocr

Usage:
  python trace_replay_langgraph_io_tools.py --task_json task1.json --mode record
  python trace_replay_langgraph_io_tools.py --task_json task1.json --mode replay
"""

import os
import re
import json
import argparse
import asyncio
import time
import hashlib
from typing import Any, Dict, List, Optional, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Browser toolkit (LangChain community)
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

# Search tool (no API key)
from langchain_community.tools import DuckDuckGoSearchResults

# OCR deps
import requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


# ----------------------------
# Config (vLLM online)
# ----------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")

SQLITE_PATH = os.getenv("SQLITE_PATH", "trace_checkpoints.sqlite")

# Tool-call output capture safety:
# store full outputs in scratch; tool_call_logs store compact summaries by default
MAX_LOG_CHARS = int(os.getenv("MAX_LOG_CHARS", "20000"))


# ----------------------------
# Utils
# ----------------------------
def sha256_short(x: Any) -> str:
    b = json.dumps(x, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def compact_value(x: Any, max_chars: int = MAX_LOG_CHARS) -> Dict[str, Any]:
    """
    Return a compact representation of x suitable for logs, while keeping
    full values stored elsewhere (e.g., scratch).
    """
    try:
        s = json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    if len(s) > max_chars:
        return {"hash": sha256_short(s), "len": len(s), "preview": s[:max_chars] + "...[truncated]"}
    return {"hash": sha256_short(s), "len": len(s), "preview": s}


# ----------------------------
# Task parsing
# ----------------------------
def parse_annotator_steps(task_json: Dict[str, Any]) -> List[str]:
    steps_blob = task_json["Annotator Metadata"]["Steps"]
    matches = re.findall(r"(?m)^\s*\d+\.\s+(.*)$", steps_blob)
    if not matches:
        raise ValueError("Could not parse steps from Annotator Metadata.Steps")
    return [m.strip() for m in matches]


# ----------------------------
# Minimal arXiv helpers
# ----------------------------
def parse_arxiv_abs_links(text: str) -> List[str]:
    return sorted(set(re.findall(r"https?://arxiv\.org/abs/\d+\.\d+(?:v\d+)?", text)))


def make_arxiv_pdf_url(abs_url: str) -> str:
    m = re.search(r"/abs/(\d+\.\d+)", abs_url)
    if not m:
        return abs_url
    return f"https://arxiv.org/pdf/{m.group(1)}.pdf"


# ----------------------------
# OCR tool (LangChain Tool)
# ----------------------------
@tool("ocr_pdf_find_axis_labels")
def ocr_pdf_find_axis_labels(pdf_url: str) -> Dict[str, Any]:
    """
    Download a PDF and OCR early pages to find axis label words.
    Returns OCR text and the page index where labels were detected.
    """
    os.makedirs("downloads", exist_ok=True)
    pdf_path = os.path.join("downloads", "paper.pdf")

    r = requests.get(pdf_url, timeout=60)
    r.raise_for_status()
    with open(pdf_path, "wb") as f:
        f.write(r.content)

    doc = fitz.open(pdf_path)
    keywords = ["deontological", "egalitarian", "utilitarian", "consequential", "localized", "standardized"]

    found_text = ""
    found_page = None

    for i in range(min(12, doc.page_count)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=220, alpha=False)
        img_path = os.path.join("downloads", f"page_{i}.png")
        pix.save(img_path)

        text = pytesseract.image_to_string(Image.open(img_path), lang="eng")
        low = text.lower()
        hits = sum(1 for w in keywords if w in low)
        if hits >= 3:
            found_text = text
            found_page = i
            break

    doc.close()

    if not found_text:
        raise RuntimeError("OCR could not find axis labels in first pages. Increase scan range if needed.")

    return {"pdf_path": pdf_path, "page_index": found_page, "ocr_text": found_text}


async def llm_extract_label_words(llm: ChatOpenAI, ocr_text: str) -> List[str]:
    sys = SystemMessage(content="Extract the six axis-end label words. Return ONLY a JSON array of lowercase words.")
    user = HumanMessage(content=f"OCR TEXT:\n{ocr_text}")
    resp = await llm.ainvoke([sys, user])
    txt = (resp.content or "").strip()

    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return [str(x).strip().lower() for x in data]
    except Exception:
        pass

    words = []
    for w in ["deontological", "egalitarian", "localized", "standardized", "utilitarian", "consequential"]:
        if w in ocr_text.lower():
            words.append(w)
    return sorted(set(words))


async def llm_finalize(llm: ChatOpenAI, question: str, scratch: Dict[str, Any]) -> str:
    sys = SystemMessage(content="Answer strictly in the format requested by the question. No extra text.")
    evidence = json.dumps(scratch, ensure_ascii=False, default=str)
    if len(evidence) > 200_000:
        evidence = evidence[:200_000] + "\n...[truncated]..."
    user = HumanMessage(content=f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}")
    resp = await llm.ainvoke([sys, user])
    return (resp.content or "").strip()


# ----------------------------
# State + Logs
# ----------------------------
class ToolCallLog(TypedDict, total=False):
    step_no: int
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    tool_start_ts: float
    tool_end_ts: float
    tool_duration_s: float
    error: str


class StepLog(TypedDict, total=False):
    step_no: int
    step_text: str
    step_input: Dict[str, Any]
    step_output: Dict[str, Any]
    step_start_ts: float
    step_end_ts: float
    step_duration_s: float
    error: str


class TraceState(TypedDict, total=False):
    task: Dict[str, Any]
    steps: List[str]
    step_idx: int
    scratch: Dict[str, Any]
    final_answer: Optional[str]

    total_start_ts: float
    total_end_ts: float
    total_duration_s: float

    step_logs: List[StepLog]
    tool_call_logs: List[ToolCallLog]


def ensure_timers_and_logs(state: TraceState) -> None:
    if "total_start_ts" not in state:
        state["total_start_ts"] = time.time()
    state.setdefault("scratch", {})
    state.setdefault("step_logs", [])
    state.setdefault("tool_call_logs", [])
    state.setdefault("step_idx", 0)


def finalize_total_timer(state: TraceState) -> None:
    state["total_end_ts"] = time.time()
    state["total_duration_s"] = float(state["total_end_ts"] - state["total_start_ts"])


# ----------------------------
# Tool-call logging wrappers
# ----------------------------
async def call_tool_logged_async(
    state: TraceState,
    step_no: int,
    tool_name: str,
    tool_obj: Any,
    tool_input: Dict[str, Any],
    scratch_key: Optional[str] = None,
) -> Any:
    """
    For LangChain async tools: await tool_obj.arun(tool_input)
    Stores full output in scratch[scratch_key] if provided, and logs compact tool IO.
    """
    log: ToolCallLog = {
        "step_no": step_no,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_start_ts": time.time(),
    }
    try:
        out = await tool_obj.arun(tool_input)
        if scratch_key:
            state["scratch"][scratch_key] = out

        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["tool_output"] = compact_value(out)
        state["tool_call_logs"].append(log)
        return out
    except Exception as e:
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["error"] = f"{type(e).__name__}: {e}"
        state["tool_call_logs"].append(log)
        raise


def call_tool_logged_sync(
    state: TraceState,
    step_no: int,
    tool_name: str,
    tool_fn_invoke: Any,
    tool_input: Dict[str, Any],
    scratch_key: Optional[str] = None,
) -> Any:
    """
    For sync tools (like our OCR tool invoke): tool_fn_invoke(tool_input)
    """
    log: ToolCallLog = {
        "step_no": step_no,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_start_ts": time.time(),
    }
    try:
        out = tool_fn_invoke(tool_input)
        if scratch_key:
            state["scratch"][scratch_key] = out

        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["tool_output"] = compact_value(out)
        state["tool_call_logs"].append(log)
        return out
    except Exception as e:
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["error"] = f"{type(e).__name__}: {e}"
        state["tool_call_logs"].append(log)
        raise


async def call_llm_logged(
    state: TraceState,
    step_no: int,
    llm: ChatOpenAI,
    messages: List[Any],
    scratch_key: Optional[str] = None,
) -> Any:
    """
    Log LLM calls as "tools" too (optional but usually wanted for replayable traces).
    """
    tool_name = "llm.chat"
    log: ToolCallLog = {
        "step_no": step_no,
        "tool_name": tool_name,
        "tool_input": {"messages": [getattr(m, "content", str(m)) for m in messages]},
        "tool_start_ts": time.time(),
    }
    try:
        out = await llm.ainvoke(messages)
        if scratch_key:
            state["scratch"][scratch_key] = getattr(out, "content", str(out))

        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["tool_output"] = compact_value(getattr(out, "content", str(out)))
        state["tool_call_logs"].append(log)
        return out
    except Exception as e:
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["error"] = f"{type(e).__name__}: {e}"
        state["tool_call_logs"].append(log)
        raise


# ----------------------------
# Tools setup (browser + search)
# ----------------------------
async def make_tools():
    async_browser = create_async_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    browser_tools = {t.name: t for t in browser_toolkit.get_tools()}

    search_tool = DuckDuckGoSearchResults(num_results=5)
    return browser_tools, search_tool


# ----------------------------
# Dynamic graph builder (N steps)
# ----------------------------
def build_graph_dynamic(num_steps: int, execute_step_fn):
    g = StateGraph(TraceState)
    for i in range(1, num_steps + 1):
        g.add_node(f"step_{i}", execute_step_fn)
    g.set_entry_point("step_1")
    for i in range(1, num_steps):
        g.add_edge(f"step_{i}", f"step_{i+1}")
    g.add_edge(f"step_{num_steps}", END)
    return g


# ----------------------------
# Step executor (routes step text -> tools) with step IO + tool-call logs
# ----------------------------
async def make_execute_step(browser_tools, search_tool, llm: ChatOpenAI):
    navigate = browser_tools.get("navigate_browser")
    extract_text = browser_tools.get("extract_text")

    if navigate is None or extract_text is None:
        raise RuntimeError("PlayWrightBrowserToolkit missing navigate_browser/extract_text tools.")

    async def execute_step(state: TraceState) -> TraceState:
        ensure_timers_and_logs(state)

        idx = state["step_idx"]
        step_text = state["steps"][idx]
        step_no = idx + 1
        state["step_idx"] = idx + 1

        step_log: StepLog = {
            "step_no": step_no,
            "step_text": step_text,
            "step_input": {"step_text": step_text},
            "step_output": {},
            "step_start_ts": time.time(),
        }

        try:
            t = step_text.lower()

            # ---- explicit URL in step text ----
            url_m = re.search(r"(https?://\S+)", step_text)
            if url_m:
                url = url_m.group(1).rstrip(").,")
                step_log["step_input"]["url"] = url
                await call_tool_logged_async(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                out = {"action": "navigate", "url": url}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            # ---- arXiv trace patterns ----
            elif "go to arxiv.org" in t and "advanced search" in t:
                url = "https://arxiv.org/search/advanced"
                step_log["step_input"]["url"] = url
                await call_tool_logged_async(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                out = {"action": "navigate", "url": url}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif 'enter "ai regulation"' in t and "all fields" in t:
                out = {"noted": {"query": "AI regulation", "field": "all"}}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "enter 2022-06-01 and 2022-07-01" in t and "submission date (original)" in t:
                url = (
                    "https://arxiv.org/search/advanced?"
                    "advanced=&terms-0-operator=AND&terms-0-term=AI+regulation&terms-0-field=all"
                    "&classification-physics_archives=all&classification-include_cross_list=include"
                    "&date-filter_by=submitted_date_first&date-from_date=2022-06-01&date-to_date=2022-07-01"
                    "&abstracts=show&size=50&order=-announced_date_first"
                )
                step_log["step_input"]["url"] = url
                await call_tool_logged_async(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                page = await call_tool_logged_async(state, step_no, "browser.extract_text", extract_text, {}, scratch_key="ai_reg_search_page_text")
                abs_links = parse_arxiv_abs_links(page)
                state["scratch"]["ai_reg_abs_links"] = abs_links
                out = {"action": "arxiv_search", "abs_links_count": len(abs_links)}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "go through the search results to find the article" in t and "fairness in agreement with european values" in t:
                page = state["scratch"].get("ai_reg_search_page_text")
                if not page:
                    page = await call_tool_logged_async(state, step_no, "browser.extract_text", extract_text, {}, scratch_key="ai_reg_search_page_text")
                abs_links = parse_arxiv_abs_links(page)
                state["scratch"]["ai_reg_abs_links"] = abs_links
                out = {"action": "capture_candidates", "abs_links_count": len(abs_links)}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "note the six words used as labels" in t:
                abs_links = state["scratch"].get("ai_reg_abs_links", [])
                if not abs_links:
                    page = await call_tool_logged_async(state, step_no, "browser.extract_text", extract_text, {}, scratch_key="ai_reg_search_page_text")
                    abs_links = parse_arxiv_abs_links(page)
                    state["scratch"]["ai_reg_abs_links"] = abs_links

                step_log["step_input"]["candidate_abs_links_count"] = len(abs_links)

                sys = SystemMessage(content="Pick the single arXiv abs URL matching the title. Return ONLY the URL.")
                user = HumanMessage(content=f"Title: Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\nCandidates:\n{abs_links}")
                resp = await call_llm_logged(state, step_no, llm, [sys, user], scratch_key=f"llm_step_{step_no}_chosen_abs")
                chosen_abs = (resp.content or "").strip()
                if chosen_abs not in abs_links and abs_links:
                    chosen_abs = abs_links[0]

                pdf_url = make_arxiv_pdf_url(chosen_abs)

                # OCR tool (sync)
                ocr_out = call_tool_logged_sync(
                    state,
                    step_no,
                    "ocr_pdf_find_axis_labels",
                    ocr_pdf_find_axis_labels.invoke,
                    {"pdf_url": pdf_url},
                    scratch_key="ai_reg_ocr_result",
                )

                labels = await llm_extract_label_words(llm, ocr_out["ocr_text"])
                state["scratch"]["ai_reg_label_words"] = labels
                out = {
                    "action": "ocr_and_extract_labels",
                    "chosen_abs": chosen_abs,
                    "pdf_url": pdf_url,
                    "ocr_page_index": ocr_out["page_index"],
                    "label_words": labels,
                }
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "go back to arxiv.org" in t:
                url = "https://arxiv.org"
                await call_tool_logged_async(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                out = {"action": "navigate", "url": url}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif 'find "physics and society"' in t:
                url = "https://arxiv.org/list/physics.soc-ph/recent"
                await call_tool_logged_async(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                out = {"action": "navigate", "url": url, "tag": "physics.soc-ph"}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif 'note that the tag for this category is "physics.soc-ph"' in t:
                state["scratch"]["physics_society_tag"] = "physics.soc-ph"
                out = {"noted": {"tag": "physics.soc-ph"}}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif 'enter "physics.soc-ph"' in t and "all fields" in t:
                out = {"noted": {"query": "physics.soc-ph", "field": "all"}}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "enter 2016-08-11 and 2016-08-12" in t and "submission date (original)" in t:
                url = (
                    "https://arxiv.org/search/advanced?"
                    "advanced=&terms-0-operator=AND&terms-0-term=physics.soc-ph&terms-0-field=all"
                    "&classification-physics_archives=all&classification-include_cross_list=include"
                    "&date-filter_by=submitted_date_first&date-from_date=2016-08-11&date-to_date=2016-08-12"
                    "&abstracts=show&size=50&order=-announced_date_first"
                )
                await call_tool_logged_async(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                page = await call_tool_logged_async(state, step_no, "browser.extract_text", extract_text, {}, scratch_key="phys_search_page_text")
                abs_links = parse_arxiv_abs_links(page)
                state["scratch"]["phys_abs_links"] = abs_links
                out = {"action": "arxiv_search", "abs_links_count": len(abs_links)}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "search for instances of the six words" in t:
                labels = state["scratch"].get("ai_reg_label_words", [])
                page = state["scratch"].get("phys_search_page_text")
                if not page:
                    page = await call_tool_logged_async(state, step_no, "browser.extract_text", extract_text, {}, scratch_key="phys_search_page_text")
                low = page.lower()
                matched = next((w for w in labels if w in low), None)
                state["scratch"]["phys_matched_word"] = matched
                out = {"action": "match_label_word_in_results", "matched_word": matched}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            # ---- Finding Nemo / USGS patterns ----
            elif t.startswith("search the web for"):
                q = re.search(r"search the web for\s+[“\"](.+?)[”\"]\s*$", step_text, flags=re.IGNORECASE)
                query = q.group(1) if q else step_text
                step_log["step_input"]["query"] = query
                # DuckDuckGoSearchResults is sync-ish but exposes ainvoke in LC; handle as async tool-style call
                res = await call_tool_logged_async(state, step_no, "search.duckduckgo", search_tool, query, scratch_key=f"search_{step_no}_raw")
                out = {"action": "search", "query": query, "stored_in_scratch": f"search_{step_no}_raw"}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "click result" in t:
                # very lightweight approximation: open first URL seen in last stored search blob
                last_key = None
                for k in sorted(state["scratch"].keys(), reverse=True):
                    if k.startswith("search_") and k.endswith("_raw"):
                        last_key = k
                        break
                blob = state["scratch"].get(last_key, "")
                m = re.search(r"(https?://\S+)", str(blob))
                target_url = m.group(1).rstrip(").,") if m else None
                step_log["step_input"]["target_url"] = target_url
                if target_url:
                    await call_tool_logged_async(state, step_no, "browser.navigate_browser", navigate, {"url": target_url})
                    out = {"action": "navigate", "url": target_url}
                else:
                    out = {"action": "click_result", "warning": "no URL found in search results"}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            elif "note the results" in t or "note the place" in t or "note the zip code" in t or "scroll" in t or "find" in t:
                page = await call_tool_logged_async(state, step_no, "browser.extract_text", extract_text, {}, scratch_key=f"page_{step_no}_text")
                out = {"action": "extract_text", "stored_in_scratch": f"page_{step_no}_text", "text_len": len(page)}
                state["scratch"][f"step_{step_no}_output"] = out
                step_log["step_output"] = out

            # ---- Finalize (if step implies conclusion OR last step) ----
            else:
                is_last = (step_no == len(state["steps"]))
                if "this is the answer" in t or "indicating that" in t or is_last:
                    # log LLM finalize as tool call
                    # (we call llm_finalize, but also want tool-style logs)
                    sys = SystemMessage(content="Answer strictly in the format requested by the question. No extra text.")
                    evidence = json.dumps(state["scratch"], ensure_ascii=False, default=str)
                    if len(evidence) > 200_000:
                        evidence = evidence[:200_000] + "\n...[truncated]..."
                    user = HumanMessage(content=f"QUESTION:\n{state['task']['Question']}\n\nEVIDENCE:\n{evidence}")
                    resp = await call_llm_logged(state, step_no, llm, [sys, user], scratch_key="final_answer_raw")
                    ans = (resp.content or "").strip()
                    state["final_answer"] = ans
                    out = {"action": "finalize", "final_answer": ans}
                    state["scratch"][f"step_{step_no}_output"] = out
                    step_log["step_output"] = out
                else:
                    out = {"action": "unhandled_step", "step_text": step_text}
                    state["scratch"][f"step_{step_no}_output"] = out
                    step_log["step_output"] = out

            step_log["step_end_ts"] = time.time()
            step_log["step_duration_s"] = float(step_log["step_end_ts"] - step_log["step_start_ts"])
            state["step_logs"].append(step_log)

            # If this was the final step node, finalize total timer here
            if step_no == len(state["steps"]):
                finalize_total_timer(state)

            return state

        except Exception as e:
            step_log["step_end_ts"] = time.time()
            step_log["step_duration_s"] = float(step_log["step_end_ts"] - step_log["step_start_ts"])
            step_log["error"] = f"{type(e).__name__}: {e}"
            state["step_logs"].append(step_log)
            # also finalize total timer on failure
            finalize_total_timer(state)
            raise

    return execute_step


# ----------------------------
# Runner
# ----------------------------
async def run(task_json: Dict[str, Any], mode: Literal["record", "replay"]) -> Dict[str, Any]:
    steps = parse_annotator_steps(task_json)
    task_id = task_json.get("task_id", "thread")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0.0,
    )

    browser_tools, search_tool = await make_tools()
    execute_step = await make_execute_step(browser_tools, search_tool, llm)

    graph = build_graph_dynamic(len(steps), execute_step)
    checkpointer = SqliteSaver.from_conn_string(SQLITE_PATH)
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": task_id}}

    if mode == "record":
        init_state: TraceState = {
            "task": task_json,
            "steps": steps,
            "step_idx": 0,
            "scratch": {},
            "step_logs": [],
            "tool_call_logs": [],
            "total_start_ts": time.time(),
        }
        out = await app.ainvoke(init_state, config=config)
        if "total_duration_s" not in out:
            finalize_total_timer(out)
        return {
            "mode": "record",
            "task_id": task_id,
            "num_steps": len(steps),
            "final_answer": out.get("final_answer"),
            "total_duration_s": out.get("total_duration_s"),
            "step_logs": out.get("step_logs", []),
            "tool_call_logs": out.get("tool_call_logs", []),
        }

    # replay: load last checkpointed state
    st = await app.aget_state(config)
    restored = dict(st.values) if st and st.values else {}
    return {
        "mode": "replay",
        "task_id": task_id,
        "num_steps": len(restored.get("steps", [])) if restored else None,
        "final_answer": restored.get("final_answer"),
        "total_duration_s": restored.get("total_duration_s"),
        "step_logs": restored.get("step_logs", []),
        "tool_call_logs": restored.get("tool_call_logs", []),
        "restored_keys": sorted(list(restored.keys())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", required=True, help="Path to task JSON file")
    ap.add_argument("--mode", choices=["record", "replay"], default="record")
    args = ap.parse_args()

    task = json.loads(open(args.task_json, "r", encoding="utf-8").read())
    result = asyncio.run(run(task, args.mode))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
