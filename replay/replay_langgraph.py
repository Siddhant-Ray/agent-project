"""
End-to-end LangGraph trace replayer (dynamic steps) with:
✅ vLLM OpenAI-compatible LLM
✅ Playwright browser tools + DuckDuckGo search tool
✅ OCR tool with robust candidate-page strategy (keeps tesseract)
✅ Per-step FULL input + FULL output saved (not compact)
✅ Per-tool-call FULL input + FULL output saved (not compact)
✅ Total execution time
✅ JSON checkpoint record/replay across runs

Run vLLM:
  vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000

Install:
  pip install -U langgraph langchain langchain-openai langchain-community playwright pymupdf pillow pytesseract duckduckgo-search
  playwright install chromium

Usage:
  python replay_langgraph_fullio.py --task_json task.json --mode record
  python replay_langgraph_fullio.py --task_json task.json --mode replay
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

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools import DuckDuckGoSearchResults

from playwright.async_api import async_playwright, Browser

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

JSON_CKPT_DIR = os.getenv("JSON_CKPT_DIR", "checkpoints_json")


# ----------------------------
# JSON checkpoints (cross-run replay)
# ----------------------------
def json_ckpt_path(task_id: str) -> str:
    os.makedirs(JSON_CKPT_DIR, exist_ok=True)
    return os.path.join(JSON_CKPT_DIR, f"{task_id}.json")


def json_ckpt_save(task_id: str, state: Dict[str, Any]) -> None:
    with open(json_ckpt_path(task_id), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def json_ckpt_load(task_id: str) -> Optional[Dict[str, Any]]:
    p = json_ckpt_path(task_id)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Helpers
# ----------------------------
def parse_annotator_steps(task_json: Dict[str, Any]) -> List[str]:
    steps_blob = task_json["Annotator Metadata"]["Steps"]
    matches = re.findall(r"(?m)^\s*\d+\.\s+(.*)$", steps_blob)
    if not matches:
        raise ValueError("Could not parse steps from Annotator Metadata.Steps")
    return [m.strip() for m in matches]


def parse_arxiv_abs_links(text: str) -> List[str]:
    return sorted(set(re.findall(r"https?://arxiv\.org/abs/\d+\.\d+(?:v\d+)?", text)))


def make_arxiv_pdf_url(abs_url: str) -> str:
    m = re.search(r"/abs/(\d+\.\d+)", abs_url)
    if not m:
        return abs_url
    return f"https://arxiv.org/pdf/{m.group(1)}.pdf"


def sha256_short(x: Any) -> str:
    b = json.dumps(x, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


# ----------------------------
# OCR tool (keeps tesseract)
# ----------------------------
@tool("ocr_pdf_find_axis_labels")
def ocr_pdf_find_axis_labels(
    pdf_url: str,
    max_pages: int = 80,
    dpi: int = 300,
    min_hits_to_accept: int = 3,
    candidate_pool: int = 15,
    fallback_stride: int = 1,
) -> dict:
    """Download a PDF and OCR pages likely to contain axis label words (tesseract).

    Returns:
      pdf_path
      first_page_index_ge_threshold (or None)
      best_page_index
      best_hits
      best_found_words
      best_ocr_text
      tried_pages
      params
    """
    os.makedirs("downloads", exist_ok=True)
    pdf_path = os.path.join("downloads", "paper.pdf")

    r = requests.get(pdf_url, timeout=60)
    r.raise_for_status()
    with open(pdf_path, "wb") as f:
        f.write(r.content)

    keywords = ["deontological", "egalitarian", "localized", "standardized", "utilitarian", "consequential"]

    doc = fitz.open(pdf_path)
    scan_pages = min(max_pages, doc.page_count)

    # (A) Text-layer pre-scan: choose candidate pages
    scored_pages = []
    for i in range(scan_pages):
        try:
            text = (doc.load_page(i).get_text("text") or "").lower()
        except Exception:
            text = ""
        hits = sum(1 for w in keywords if w in text)
        boost = 0
        if "figure" in text or "fig." in text:
            boost += 1
        if "axis" in text:
            boost += 1
        scored_pages.append((hits + boost, hits, i))
    scored_pages.sort(reverse=True)
    candidate_pages = [i for _, __, i in scored_pages[:candidate_pool]]
    for i in range(min(6, scan_pages)):
        if i not in candidate_pages:
            candidate_pages.append(i)

    tried_pages: List[int] = []
    best = {"best_hits": -1, "best_page_index": None, "best_found_words": [], "best_ocr_text": ""}

    def ocr_page(page_index: int) -> dict:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_path = os.path.join("downloads", f"page_{page_index}.png")
        pix.save(img_path)
        text = pytesseract.image_to_string(Image.open(img_path), lang="eng")
        low = text.lower()
        found = [w for w in keywords if w in low]
        return {"text": text, "found": found, "hits": len(found)}

    first_ge = None

    # (B) OCR candidate pages
    for pi in candidate_pages:
        if pi < 0 or pi >= scan_pages:
            continue
        tried_pages.append(pi)
        out = ocr_page(pi)
        if out["hits"] > best["best_hits"]:
            best = {
                "best_hits": out["hits"],
                "best_page_index": pi,
                "best_found_words": out["found"],
                "best_ocr_text": out["text"],
            }
        if first_ge is None and out["hits"] >= min_hits_to_accept:
            first_ge = pi
            break

    # (C) Fallback scan
    if first_ge is None:
        for pi in range(0, scan_pages, max(1, fallback_stride)):
            if pi in tried_pages:
                continue
            tried_pages.append(pi)
            out = ocr_page(pi)
            if out["hits"] > best["best_hits"]:
                best = {
                    "best_hits": out["hits"],
                    "best_page_index": pi,
                    "best_found_words": out["found"],
                    "best_ocr_text": out["text"],
                }
            if out["hits"] >= min_hits_to_accept:
                first_ge = pi
                break

    doc.close()

    return {
        "pdf_path": pdf_path,
        "first_page_index_ge_threshold": first_ge,
        "best_page_index": best["best_page_index"],
        "best_hits": best["best_hits"],
        "best_found_words": best["best_found_words"],
        "best_ocr_text": best["best_ocr_text"],
        "tried_pages": tried_pages,  # FULL list
        "params": {
            "max_pages": max_pages,
            "dpi": dpi,
            "min_hits_to_accept": min_hits_to_accept,
            "candidate_pool": candidate_pool,
            "fallback_stride": fallback_stride,
        },
    }


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
    # fallback
    words = []
    for w in ["deontological", "egalitarian", "localized", "standardized", "utilitarian", "consequential"]:
        if w in ocr_text.lower():
            words.append(w)
    return sorted(set(words))


# ----------------------------
# Logs + State (FULL IO)
# ----------------------------
class ToolCallLog(TypedDict, total=False):
    step_no: int
    tool_name: str
    tool_input_full: Any
    tool_output_full: Any
    tool_start_ts: float
    tool_end_ts: float
    tool_duration_s: float
    error: str


class StepLog(TypedDict, total=False):
    step_no: int
    step_text: str
    step_input_full: Dict[str, Any]
    step_output_full: Dict[str, Any]
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


def ensure_state_initialized(state: TraceState) -> None:
    state.setdefault("scratch", {})
    state.setdefault("step_logs", [])
    state.setdefault("tool_call_logs", [])
    state.setdefault("step_idx", 0)
    if "total_start_ts" not in state:
        state["total_start_ts"] = time.time()


def finalize_total(state: TraceState) -> None:
    state["total_end_ts"] = time.time()
    state["total_duration_s"] = float(state["total_end_ts"] - state["total_start_ts"])


async def call_async_tool(
    state: TraceState,
    step_no: int,
    name: str,
    tool_obj: Any,
    tool_input: Any,
    scratch_key: Optional[str] = None,
):
    log: ToolCallLog = {
        "step_no": step_no,
        "tool_name": name,
        "tool_input_full": tool_input,  # FULL
        "tool_start_ts": time.time(),
    }
    try:
        if isinstance(tool_input, dict):
            out = await tool_obj.arun(tool_input)
        else:
            out = await tool_obj.ainvoke(tool_input)

        if scratch_key:
            state["scratch"][scratch_key] = out

        log["tool_output_full"] = out  # FULL
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        state["tool_call_logs"].append(log)
        return out
    except Exception as e:
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["error"] = f"{type(e).__name__}: {e}"
        state["tool_call_logs"].append(log)
        raise


def call_sync_tool(
    state: TraceState,
    step_no: int,
    name: str,
    fn,
    tool_input: Dict[str, Any],
    scratch_key: Optional[str] = None,
):
    log: ToolCallLog = {
        "step_no": step_no,
        "tool_name": name,
        "tool_input_full": tool_input,  # FULL
        "tool_start_ts": time.time(),
    }
    try:
        out = fn(tool_input)

        if scratch_key:
            state["scratch"][scratch_key] = out

        log["tool_output_full"] = out  # FULL
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        state["tool_call_logs"].append(log)
        return out
    except Exception as e:
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["error"] = f"{type(e).__name__}: {e}"
        state["tool_call_logs"].append(log)
        raise


async def call_llm(
    state: TraceState,
    step_no: int,
    llm: ChatOpenAI,
    messages: List[Any],
    scratch_key: Optional[str] = None,
):
    log: ToolCallLog = {
        "step_no": step_no,
        "tool_name": "llm.chat",
        "tool_input_full": {"messages": [getattr(m, "content", str(m)) for m in messages]},  # FULL
        "tool_start_ts": time.time(),
    }
    try:
        out = await llm.ainvoke(messages)
        content = getattr(out, "content", str(out))

        if scratch_key:
            state["scratch"][scratch_key] = content

        log["tool_output_full"] = content  # FULL
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        state["tool_call_logs"].append(log)
        return out
    except Exception as e:
        log["tool_end_ts"] = time.time()
        log["tool_duration_s"] = float(log["tool_end_ts"] - log["tool_start_ts"])
        log["error"] = f"{type(e).__name__}: {e}"
        state["tool_call_logs"].append(log)
        raise


# ----------------------------
# Playwright tools setup (async-safe)
# ----------------------------
async def make_tools_and_browser():
    pw = await async_playwright().start()
    browser: Browser = await pw.chromium.launch(headless=True)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    browser_tools = {t.name: t for t in toolkit.get_tools()}
    search_tool = DuckDuckGoSearchResults(num_results=5)
    return browser_tools, search_tool, pw, browser


# ----------------------------
# Dynamic graph builder
# ----------------------------
def build_graph_dynamic(n: int, execute_step_fn):
    g = StateGraph(TraceState)
    for i in range(1, n + 1):
        g.add_node(f"step_{i}", execute_step_fn)
    g.set_entry_point("step_1")
    for i in range(1, n):
        g.add_edge(f"step_{i}", f"step_{i+1}")
    g.add_edge(f"step_{n}", END)
    return g


# ----------------------------
# Step executor (FULL step IO saved)
# ----------------------------
async def make_execute_step(browser_tools, search_tool, llm: ChatOpenAI):
    navigate = browser_tools.get("navigate_browser")
    extract_text = browser_tools.get("extract_text")
    if navigate is None or extract_text is None:
        raise RuntimeError("PlayWrightBrowserToolkit missing navigate_browser/extract_text tools.")

    async def execute_step(state: TraceState) -> TraceState:
        ensure_state_initialized(state)

        idx = state["step_idx"]
        step_text = state["steps"][idx]
        step_no = idx + 1
        state["step_idx"] = idx + 1

        # FULL step input includes: step text, step index, and selected state context
        step_input_full = {
            "step_no": step_no,
            "step_text": step_text,
            "state_before": {
                "step_idx": idx,
                "known_scratch_keys": sorted(list(state["scratch"].keys())),
                "final_answer": state.get("final_answer"),
            },
        }

        step_log: StepLog = {
            "step_no": step_no,
            "step_text": step_text,
            "step_input_full": step_input_full,   # FULL
            "step_output_full": {},               # FULL
            "step_start_ts": time.time(),
        }

        try:
            t = step_text.lower()
            full_out: Dict[str, Any] = {}

            # URL in text
            url_m = re.search(r"(https?://\S+)", step_text)
            if url_m:
                url = url_m.group(1).rstrip(").,")
                await call_async_tool(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                full_out = {"action": "navigate", "url": url}

            elif "go to arxiv.org" in t and "advanced search" in t:
                url = "https://arxiv.org/search/advanced"
                await call_async_tool(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                full_out = {"action": "navigate", "url": url}

            elif "enter 2022-06-01 and 2022-07-01" in t and "submission date (original)" in t:
                url = (
                    "https://arxiv.org/search/advanced?"
                    "advanced=&terms-0-operator=AND&terms-0-term=AI+regulation&terms-0-field=all"
                    "&date-filter_by=submitted_date_first&date-from_date=2022-06-01&date-to_date=2022-07-01"
                    "&abstracts=show&size=50&order=-announced_date_first"
                )
                await call_async_tool(state, step_no, "browser.navigate_browser", navigate, {"url": url})
                page = await call_async_tool(state, step_no, "browser.extract_text", extract_text, {}, scratch_key="ai_reg_search_page_text")
                abs_links = parse_arxiv_abs_links(page)
                state["scratch"]["ai_reg_abs_links"] = abs_links
                full_out = {
                    "action": "arxiv_search",
                    "url": url,
                    "abs_links_count": len(abs_links),
                    "abs_links_full": abs_links,  # FULL
                }

            elif "note the six words used as labels" in t:
                abs_links = state["scratch"].get("ai_reg_abs_links", [])
                if not abs_links:
                    page = await call_async_tool(state, step_no, "browser.extract_text", extract_text, {}, scratch_key="ai_reg_search_page_text")
                    abs_links = parse_arxiv_abs_links(page)
                    state["scratch"]["ai_reg_abs_links"] = abs_links

                sys = SystemMessage(content="Pick the single arXiv abs URL matching the title. Return ONLY the URL.")
                user = HumanMessage(
                    content="Title: Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\n"
                            f"Candidates:\n{abs_links}"
                )
                resp = await call_llm(state, step_no, llm, [sys, user], scratch_key=f"llm_step_{step_no}_chosen_abs")
                chosen_abs = (resp.content or "").strip()
                if chosen_abs not in abs_links and abs_links:
                    chosen_abs = abs_links[0]

                pdf_url = make_arxiv_pdf_url(chosen_abs)

                ocr_out = call_sync_tool(
                    state,
                    step_no,
                    "ocr_pdf_find_axis_labels",
                    ocr_pdf_find_axis_labels.invoke,
                    {
                        "pdf_url": pdf_url,
                        "max_pages": 120,
                        "dpi": 350,
                        "min_hits_to_accept": 3,
                        "candidate_pool": 20,
                        "fallback_stride": 1,
                    },
                    scratch_key="ai_reg_ocr_result",
                )

                ocr_text = ocr_out.get("best_ocr_text", "")
                labels = await llm_extract_label_words(llm, ocr_text)
                state["scratch"]["ai_reg_label_words"] = labels

                full_out = {
                    "action": "ocr_and_extract_labels",
                    "chosen_abs_full": chosen_abs,
                    "pdf_url": pdf_url,
                    "ocr_out_full": ocr_out,      # FULL
                    "label_words_full": labels,   # FULL
                }

            elif t.startswith("search the web for"):
                q = re.search(r"search the web for\s+[“\"](.+?)[”\"]\s*$", step_text, flags=re.IGNORECASE)
                query = q.group(1) if q else step_text
                res = await call_async_tool(state, step_no, "search.duckduckgo", search_tool, query, scratch_key=f"search_{step_no}_raw")
                full_out = {
                    "action": "search",
                    "query": query,
                    "results_full": res,  # FULL
                    "scratch_key": f"search_{step_no}_raw",
                }

            elif "note the results" in t or "note the place" in t or "note the zip code" in t or "scroll" in t or "find" in t:
                page = await call_async_tool(state, step_no, "browser.extract_text", extract_text, {}, scratch_key=f"page_{step_no}_text")
                full_out = {
                    "action": "extract_text",
                    "text_full": page,  # FULL
                    "scratch_key": f"page_{step_no}_text",
                }

            else:
                is_last = (step_no == len(state["steps"]))
                if "this is the answer" in t or "indicating that" in t or is_last:
                    sys = SystemMessage(content="Answer strictly in the format requested by the question. No extra text.")
                    evidence = json.dumps(state["scratch"], ensure_ascii=False, default=str)
                    user = HumanMessage(content=f"QUESTION:\n{state['task']['Question']}\n\nEVIDENCE:\n{evidence[:300000]}")
                    resp = await call_llm(state, step_no, llm, [sys, user], scratch_key="final_answer_raw")
                    ans = (resp.content or "").strip()
                    state["final_answer"] = ans
                    full_out = {"action": "finalize", "final_answer": ans}
                else:
                    full_out = {"action": "unhandled_step", "step_text": step_text}

            # Persist FULL step output
            state["scratch"][f"step_{step_no}_input_full"] = step_input_full
            state["scratch"][f"step_{step_no}_output_full"] = full_out

            step_log["step_output_full"] = full_out
            step_log["step_end_ts"] = time.time()
            step_log["step_duration_s"] = float(step_log["step_end_ts"] - step_log["step_start_ts"])
            state["step_logs"].append(step_log)

            if step_no == len(state["steps"]):
                finalize_total(state)

            return state

        except Exception as e:
            step_log["step_end_ts"] = time.time()
            step_log["step_duration_s"] = float(step_log["step_end_ts"] - step_log["step_start_ts"])
            step_log["error"] = f"{type(e).__name__}: {e}"
            state["step_logs"].append(step_log)
            finalize_total(state)
            raise

    return execute_step


# ----------------------------
# Run (JSON record/replay)
# ----------------------------
async def run(task_json: Dict[str, Any], mode: Literal["record", "replay"]):
    steps = parse_annotator_steps(task_json)
    task_id = task_json.get("task_id", "thread")

    if mode == "replay":
        restored = json_ckpt_load(task_id)
        if restored is None:
            return {"mode": "replay", "task_id": task_id, "error": f"No JSON checkpoint found at {json_ckpt_path(task_id)}"}
        return {"mode": "replay", "task_id": task_id, **restored}

    llm = ChatOpenAI(model=MODEL_NAME, base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, temperature=0.0)

    browser_tools, search_tool, pw, browser = await make_tools_and_browser()
    try:
        execute_step = await make_execute_step(browser_tools, search_tool, llm)
        graph = build_graph_dynamic(len(steps), execute_step)
        app = graph.compile()

        init_state: TraceState = {
            "task": task_json,
            "steps": steps,
            "step_idx": 0,
            "scratch": {},
            "step_logs": [],
            "tool_call_logs": [],
            "total_start_ts": time.time(),
        }

        out = await app.ainvoke(init_state)

        if "total_duration_s" not in out:
            finalize_total(out)

        # Save FULL step IO separately too (easy access)
        step_io_full = []
        for sl in out.get("step_logs", []):
            step_io_full.append(
                {
                    "step_no": sl["step_no"],
                    "step_text": sl["step_text"],
                    "step_input_full": sl.get("step_input_full", {}),
                    "step_output_full": sl.get("step_output_full", {}),
                    "step_duration_s": sl.get("step_duration_s"),
                }
            )

        result = {
            "task_id": task_id,
            "num_steps": len(steps),
            "final_answer": out.get("final_answer"),
            "total_duration_s": out.get("total_duration_s"),
            "step_logs_full": out.get("step_logs", []),       # FULL
            "tool_call_logs_full": out.get("tool_call_logs", []),  # FULL
            "step_io_full": step_io_full,                    # FULL convenience list
            # Also keep scratch, because you asked "beyond instructions already present"
            # and scratch contains full page texts, OCR outputs, etc.
            "scratch_full": out.get("scratch", {}),           # FULL (can be large!)
        }

        json_ckpt_save(task_id, result)
        return {"mode": "record", **result}
    finally:
        await browser.close()
        await pw.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", required=True)
    ap.add_argument("--mode", choices=["record", "replay"], default="record")
    args = ap.parse_args()

    task = json.loads(open(args.task_json, "r", encoding="utf-8").read())
    result = asyncio.run(run(task, args.mode))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
