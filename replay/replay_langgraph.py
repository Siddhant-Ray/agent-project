"""
End-to-end: Dynamic LangGraph "trace replayer" driven by Annotator Metadata.Steps.

- Parses step count from task JSON (NO hardcoded 12/10).
- Builds a linear LangGraph with N nodes (N = number of parsed steps).
- Uses LangChain "inbuilt" toolkits where available:
  - Web browser: PlayWrightBrowserToolkit (LangChain community)
  - Search engine: DuckDuckGoSearchResults (LangChain community; no API key)
- Uses an OCR tool implemented as a LangChain @tool (LangChain doesn't ship a first-party OCR tool).
- Uses open-source LLM served via vLLM online (OpenAI-compatible) through ChatOpenAI.

Run vLLM (new CLI):
  vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000

Install:
  pip install -U langgraph langchain langchain-openai langchain-community playwright lxml pymupdf pillow pytesseract duckduckgo-search
  playwright install chromium
  # plus system tesseract:
  #   macOS: brew install tesseract
  #   Ubuntu: sudo apt-get install -y tesseract-ocr

Usage:
  python trace_replay_langgraph.py --task_json task1.json --mode record
  python trace_replay_langgraph.py --task_json task1.json --mode replay

Notes:
- "record" runs tools + writes checkpoints into SQLite.
- "replay" loads the stored state for that task_id and prints the last checkpointed state (no new tool calls).
"""

import os
import re
import json
import argparse
import asyncio
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
# Minimal arXiv result parsing helper
# (works on extracted text; robust enough for this use)
# ----------------------------
def parse_arxiv_abs_links(text: str) -> List[str]:
    # Extract arXiv /abs/ links
    return sorted(set(re.findall(r"https?://arxiv\.org/abs/\d+\.\d+(?:v\d+)?", text)))


def make_arxiv_pdf_url(abs_url: str) -> str:
    # abs -> pdf
    # https://arxiv.org/abs/xxxx.yyyy -> https://arxiv.org/pdf/xxxx.yyyy.pdf
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

    # fallback scan
    words = []
    for w in ["deontological", "egalitarian", "localized", "standardized", "utilitarian", "consequential"]:
        if w in ocr_text.lower():
            words.append(w)
    return sorted(set(words))


async def llm_finalize(llm: ChatOpenAI, question: str, scratch: Dict[str, Any]) -> str:
    sys = SystemMessage(content="Answer strictly in the format requested by the question. No extra text.")
    user = HumanMessage(content=f"QUESTION:\n{question}\n\nEVIDENCE:\n{json.dumps(scratch, ensure_ascii=False)[:200000]}")
    resp = await llm.ainvoke([sys, user])
    return (resp.content or "").strip()


# ----------------------------
# LangGraph state
# ----------------------------
class TraceState(TypedDict, total=False):
    task: Dict[str, Any]
    steps: List[str]
    step_idx: int
    scratch: Dict[str, Any]
    final_answer: Optional[str]


# ----------------------------
# Tool setup (browser + search)
# ----------------------------
async def make_tools():
    async_browser = create_async_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    browser_tools = {t.name: t for t in browser_toolkit.get_tools()}

    # DuckDuckGo search results tool (no API key)
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
# Step executor: routes step text -> tools
# This is intentionally rule-based so step order is exactly the annotator order.
# Extend these rules as you add more task types.
# ----------------------------
async def make_execute_step(browser_tools, search_tool, llm: ChatOpenAI):
    # Common browser tools
    navigate = browser_tools.get("navigate_browser")
    extract_text = browser_tools.get("extract_text")

    if navigate is None or extract_text is None:
        raise RuntimeError("PlayWrightBrowserToolkit tools missing (navigate_browser/extract_text). Check installation.")

    async def execute_step(state: TraceState) -> TraceState:
        idx = state.get("step_idx", 0)
        step_text = state["steps"][idx]
        state["step_idx"] = idx + 1
        state.setdefault("scratch", {})

        t = step_text.lower()
        step_no = idx + 1

        # ------------- Generic: explicit URL navigation -------------
        url_m = re.search(r"(https?://\S+)", step_text)
        if url_m:
            url = url_m.group(1).rstrip(").,")
            await navigate.arun({"url": url})
            state["scratch"][f"step_{step_no}_navigate"] = url
            return state

        # ------------- Task-specific patterns: arXiv AI regulation trace -------------
        if "go to arxiv.org" in t and "advanced search" in t:
            await navigate.arun({"url": "https://arxiv.org/search/advanced"})
            state["scratch"][f"step_{step_no}"] = "opened arXiv advanced search"
            return state

        if 'enter "ai regulation"' in t and "all fields" in t:
            # We use deterministic URL submission next step; here we just log.
            state["scratch"][f"step_{step_no}"] = 'noted query="AI regulation", field="All fields"'
            return state

        if "enter 2022-06-01 and 2022-07-01" in t and "submission date (original)" in t:
            # Deterministic advanced search URL (acts like submitting the form)
            url = (
                "https://arxiv.org/search/advanced?"
                "advanced=&terms-0-operator=AND&terms-0-term=AI+regulation&terms-0-field=all"
                "&classification-physics_archives=all&classification-include_cross_list=include"
                "&date-filter_by=submitted_date_first&date-from_date=2022-06-01&date-to_date=2022-07-01"
                "&abstracts=show&size=50&order=-announced_date_first"
            )
            await navigate.arun({"url": url})
            page = await extract_text.arun({})
            state["scratch"]["ai_reg_search_page_text"] = page
            state["scratch"]["ai_reg_abs_links"] = parse_arxiv_abs_links(page)
            return state

        if "go through the search results to find the article" in t and "fairness in agreement with european values" in t:
            # Very simple selection: find an abs link whose surrounding text contains the title phrase
            page = state["scratch"].get("ai_reg_search_page_text", "")
            if not page:
                page = await extract_text.arun({})
                state["scratch"]["ai_reg_search_page_text"] = page

            # Heuristic: grab all abs links and just store them; LLM can pick later if needed.
            abs_links = parse_arxiv_abs_links(page)
            state["scratch"]["ai_reg_abs_links"] = abs_links
            state["scratch"][f"step_{step_no}"] = "captured abs links for target selection"
            return state

        if "note the six words used as labels" in t:
            # We need the target paper PDF. If we don't have it explicitly, ask LLM to select best abs link.
            abs_links = state["scratch"].get("ai_reg_abs_links", [])
            if not abs_links:
                # fallback: read current page
                page = await extract_text.arun({})
                abs_links = parse_arxiv_abs_links(page)

            # Ask LLM: which abs link corresponds to that title?
            sys = SystemMessage(content="Pick the single arXiv abs URL most likely matching the title. Return only the URL.")
            user = HumanMessage(content=f"Title: Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\nCandidates:\n{abs_links}")
            resp = await llm.ainvoke([sys, user])
            chosen_abs = (resp.content or "").strip()
            # sanitize to an actual candidate if possible
            if chosen_abs not in abs_links and abs_links:
                chosen_abs = abs_links[0]

            pdf_url = make_arxiv_pdf_url(chosen_abs)
            ocr_out = ocr_pdf_find_axis_labels.invoke({"pdf_url": pdf_url})
            labels = await llm_extract_label_words(llm, ocr_out["ocr_text"])

            state["scratch"]["ai_reg_target_abs"] = chosen_abs
            state["scratch"]["ai_reg_target_pdf"] = pdf_url
            state["scratch"]["ai_reg_ocr_page_index"] = ocr_out["page_index"]
            state["scratch"]["ai_reg_label_words"] = labels
            return state

        if "go back to arxiv.org" in t:
            await navigate.arun({"url": "https://arxiv.org"})
            state["scratch"][f"step_{step_no}"] = "navigated back to arXiv home"
            return state

        if 'find "physics and society"' in t and "physics and society" in t:
            # Deterministic category page
            await navigate.arun({"url": "https://arxiv.org/list/physics.soc-ph/recent"})
            state["scratch"][f"step_{step_no}"] = "opened physics.soc-ph category"
            return state

        if 'note that the tag for this category is "physics.soc-ph"' in t:
            state["scratch"]["physics_society_tag"] = "physics.soc-ph"
            return state

        if 'enter "physics.soc-ph"' in t and "all fields" in t:
            state["scratch"][f"step_{step_no}"] = 'noted query="physics.soc-ph", field="All fields"'
            return state

        if "enter 2016-08-11 and 2016-08-12" in t and "submission date (original)" in t:
            url = (
                "https://arxiv.org/search/advanced?"
                "advanced=&terms-0-operator=AND&terms-0-term=physics.soc-ph&terms-0-field=all"
                "&classification-physics_archives=all&classification-include_cross_list=include"
                "&date-filter_by=submitted_date_first&date-from_date=2016-08-11&date-to_date=2016-08-12"
                "&abstracts=show&size=50&order=-announced_date_first"
            )
            await navigate.arun({"url": url})
            page = await extract_text.arun({})
            state["scratch"]["phys_search_page_text"] = page
            state["scratch"]["phys_abs_links"] = parse_arxiv_abs_links(page)
            return state

        if "search for instances of the six words" in t and "phase transition from egalitarian to hierarchical societies" in t:
            labels = state["scratch"].get("ai_reg_label_words", [])
            page = state["scratch"].get("phys_search_page_text", "")
            if not page:
                page = await extract_text.arun({})
                state["scratch"]["phys_search_page_text"] = page

            # naive: find which label word appears in the page text
            low = page.lower()
            matched = None
            for w in labels:
                if w in low:
                    matched = w
                    break
            state["scratch"]["phys_matched_word"] = matched
            return state

        # ------------- Task-specific patterns: Finding Nemo / USGS trace -------------
        if t.startswith("search the web for"):
            # e.g., Search the web for “finding nemo main character”.
            q = re.search(r"search the web for\s+[“\"](.+?)[”\"]\s*$", step_text, flags=re.IGNORECASE)
            query = q.group(1) if q else step_text
            res = await search_tool.ainvoke(query)
            state["scratch"][f"search_{step_no}"] = {"query": query, "results": res}
            return state

        if "click result" in t or "click" in t:
            # Minimal deterministic behavior: navigate to the first URL in the most recent search results if present
            # (Annotator says "click result for ...", so we approximate by opening the top hit.)
            # DuckDuckGoSearchResults returns text; we'll extract first URL-like token.
            last_search_key = None
            for k in sorted(state["scratch"].keys(), reverse=True):
                if k.startswith("search_"):
                    last_search_key = k
                    break
            if last_search_key:
                blob = state["scratch"][last_search_key]["results"]
                # find first url in blob
                m = re.search(r"(https?://\S+)", str(blob))
                if m:
                    await navigate.arun({"url": m.group(1).rstrip(").,")})
                    state["scratch"][f"step_{step_no}_clicked"] = m.group(1)
                    return state
            state["scratch"][f"step_{step_no}_clicked"] = "no URL found to click"
            return state

        if "note the results" in t or "note the place" in t or "note the zip code" in t:
            page = await extract_text.arun({})
            state["scratch"][f"page_{step_no}"] = page
            return state

        # ------------- Finalization -------------
        # If the step looks like it concludes the answer, or if it's the last step, finalize.
        is_last_step = (step_no == len(state["steps"]))
        if "this is the answer" in t or "indicating that" in t or is_last_step:
            state["final_answer"] = await llm_finalize(llm, state["task"]["Question"], state["scratch"])
            return state

        # Default: record unhandled step
        state["scratch"][f"unhandled_step_{step_no}"] = step_text
        return state

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
        init_state: TraceState = {"task": task_json, "steps": steps, "step_idx": 0, "scratch": {}}
        out = await app.ainvoke(init_state, config=config)
        return {"mode": "record", "task_id": task_id, "num_steps": len(steps), "final_answer": out.get("final_answer"), "scratch_keys": list((out.get("scratch") or {}).keys())}

    # replay: load last checkpointed state and return it
    st = await app.aget_state(config)
    return {"mode": "replay", "task_id": task_id, "restored_state": st.values}


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
