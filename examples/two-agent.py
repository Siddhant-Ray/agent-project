from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import httpx


# -----------------------------------------------------------------------------
# Minimal two-agent prototype
# - Agent 1: QA agent
# - Agent 2: Memory manager agent
#
# Design goals:
# - single-file prototype
# - use vLLM OpenAI-compatible HTTP server
# - minimal external deps: only httpx
# - document-based QA with selective verbatim memory preservation
# - no summarization / no compaction
# -----------------------------------------------------------------------------


# ==============================
# Utility helpers
# ==============================


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


_sentence_splitter = re.compile(r"(?<=[.!?])\s+")


def split_into_chunks(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    """
    Very small chunker based on word count.
    It keeps exact chunk text once created and never rewrites later.
    """
    text = normalize_text(text)
    if not text:
        return []

    sentences = _sentence_splitter.split(text)
    words_per_sentence = [s.split() for s in sentences]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent_words in words_per_sentence:
        if not sent_words:
            continue
        if current_len + len(sent_words) <= chunk_size or not current:
            current.extend(sent_words)
            current_len += len(sent_words)
        else:
            chunks.append(" ".join(current))
            tail = current[-overlap:] if overlap > 0 else []
            current = tail + sent_words
            current_len = len(current)

    if current:
        chunks.append(" ".join(current))

    return chunks


_word_re = re.compile(r"[A-Za-z0-9_']+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(text)]


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "to", "of", "in", "on", "for",
    "with", "at", "by", "from", "is", "are", "was", "were", "be", "been", "being", "as",
    "that", "this", "these", "those", "it", "its", "into", "about", "what", "which", "who",
    "when", "where", "why", "how", "do", "does", "did", "can", "could", "would", "should",
    "i", "you", "he", "she", "they", "we", "them", "their", "our", "your", "his", "her",
    "me", "my", "mine", "ours", "yours", "not", "no", "yes", "than", "there", "here",
}


def content_terms(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


# ==============================
# vLLM client
# ==============================


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        api_key: str = "token-abc123",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        extra_body: Optional[Dict] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if extra_body:
            payload.update(extra_body)

        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ==============================
# Core memory objects
# ==============================


@dataclass(frozen=True)
class MemoryBlock:
    block_id: str
    text: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        return (
            f'<BLOCK id="{self.block_id}" source="{self.source}">\n'
            f"{self.text}\n"
            f"</BLOCK>"
        )


@dataclass
class ConversationTurn:
    question: str
    answer: str


class DocumentStore:
    def __init__(self, doc_text: str, chunk_size: int = 120, overlap: int = 30) -> None:
        self.doc_text = normalize_text(doc_text)
        self.blocks: List[MemoryBlock] = []
        self._build(chunk_size=chunk_size, overlap=overlap)

    def _build(self, chunk_size: int, overlap: int) -> None:
        chunks = split_into_chunks(self.doc_text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
            block_id = f"doc_{idx:04d}_{digest}"
            self.blocks.append(
                MemoryBlock(
                    block_id=block_id,
                    text=chunk,
                    source="document",
                    metadata={"chunk_index": str(idx)},
                )
            )

    def lexical_search(self, query: str, k: int = 8) -> List[Tuple[float, MemoryBlock]]:
        q_terms = content_terms(query)
        q_set = set(q_terms)
        scored: List[Tuple[float, MemoryBlock]] = []

        for block in self.blocks:
            b_terms = content_terms(block.text)
            if not b_terms:
                continue
            b_set = set(b_terms)
            overlap = len(q_set & b_set)
            if overlap == 0:
                continue

            tf_bonus = sum(min(3, b_terms.count(t)) for t in q_set)
            density_bonus = overlap / max(1, math.sqrt(len(b_terms)))
            score = overlap * 3.0 + tf_bonus * 0.5 + density_bonus
            scored.append((score, block))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# ==============================
# Agent 2: memory manager
# ==============================


class MemoryManagerAgent:
    """
    HiAgent-like in spirit, but kept minimal for a two-agent setting.

    Important constraints:
    - keeps only verbatim blocks
    - no summaries
    - outputs selected block IDs and ordering
    - can reuse prior selected blocks from previous turns if still relevant
    """

    def __init__(self, llm: VLLMClient, max_keep: int = 6) -> None:
        self.llm = llm
        self.max_keep = max_keep
        self.last_selected_ids: List[str] = []

    def _build_candidate_view(
        self,
        question: str,
        retrieved: Sequence[MemoryBlock],
        previous_selected: Sequence[MemoryBlock],
    ) -> str:
        lines = [f"Current question: {question}", "", "Candidate blocks:"]

        seen = set()
        ordered_candidates = list(previous_selected) + list(retrieved)
        for block in ordered_candidates:
            if block.block_id in seen:
                continue
            seen.add(block.block_id)
            preview = block.text[:400].replace("\n", " ")
            lines.append(f"- id={block.block_id} | text={preview}")
        return "\n".join(lines)

    def select_blocks(
        self,
        question: str,
        retrieved: Sequence[MemoryBlock],
        previous_selected: Sequence[MemoryBlock],
    ) -> List[str]:
        candidate_view = self._build_candidate_view(question, retrieved, previous_selected)

        system_prompt = (
            "You are a memory manager for a document QA system. "
            "You must select verbatim text blocks to keep for the QA agent. "
            "Do not summarize, rewrite, merge, or invent text. "
            "Prefer blocks that directly answer the question or resolve likely follow-up references. "
            "Also keep a previously selected block if it still appears useful for continuity. "
            f"Return JSON only with one field: selected_ids, a list of at most {self.max_keep} block IDs in the exact order they should appear."
        )

        user_prompt = candidate_view + "\n\nReturn JSON now."

        raw = self.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=250,
        )

        try:
            data = self._extract_json(raw)
            selected_ids = data.get("selected_ids", [])
            if not isinstance(selected_ids, list):
                return []
            clean_ids: List[str] = []
            seen = set()
            valid_ids = {b.block_id for b in list(previous_selected) + list(retrieved)}
            for item in selected_ids:
                if isinstance(item, str) and item in valid_ids and item not in seen:
                    clean_ids.append(item)
                    seen.add(item)
                if len(clean_ids) >= self.max_keep:
                    break
            return clean_ids
        except Exception:
            # Deterministic fallback: keep highest-overlap retrieved blocks.
            return [b.block_id for b in list(previous_selected)[:2] + list(retrieved)[: self.max_keep]][: self.max_keep]

    @staticmethod
    def _extract_json(text: str) -> Dict:
        text = text.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")
        return json.loads(match.group(0))


# ==============================
# Agent 1: QA agent
# ==============================


class QAAgent:
    def __init__(self, llm: VLLMClient) -> None:
        self.llm = llm

    def answer(
        self,
        question: str,
        selected_blocks: Sequence[MemoryBlock],
        conversation_history: Sequence[ConversationTurn],
    ) -> str:
        history_tail = conversation_history[-3:]
        history_text = "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}" for turn in history_tail
        )

        block_text = "\n\n".join(block.render() for block in selected_blocks)

        system_prompt = (
            "You are a document QA assistant. "
            "Answer only from the provided blocks. "
            "If the answer is not supported by the blocks, say you do not have enough evidence in the selected document blocks. "
            "When possible, cite supporting block IDs in brackets like [doc_0001_xxx]."
        )

        user_prompt = (
            "Recent conversation (for reference only):\n"
            f"{history_text if history_text else '(empty)'}\n\n"
            "Selected document blocks:\n"
            f"{block_text if block_text else '(none)'}\n\n"
            f"Question: {question}"
        )

        return self.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )


# ==============================
# Orchestrator
# ==============================


class TwoAgentDocQA:
    def __init__(
        self,
        llm: VLLMClient,
        doc_text: str,
        retrieval_k: int = 8,
        memory_k: int = 6,
    ) -> None:
        self.store = DocumentStore(doc_text)
        self.memory_manager = MemoryManagerAgent(llm, max_keep=memory_k)
        self.qa_agent = QAAgent(llm)
        self.retrieval_k = retrieval_k
        self.history: List[ConversationTurn] = []
        self.block_index: Dict[str, MemoryBlock] = {b.block_id: b for b in self.store.blocks}
        self.current_selected_ids: List[str] = []

    def ask(self, question: str) -> Dict[str, object]:
        retrieved = [b for _, b in self.store.lexical_search(question, k=self.retrieval_k)]
        previous_selected = [self.block_index[bid] for bid in self.current_selected_ids if bid in self.block_index]

        selected_ids = self.memory_manager.select_blocks(
            question=question,
            retrieved=retrieved,
            previous_selected=previous_selected,
        )

        selected_blocks = [self.block_index[bid] for bid in selected_ids if bid in self.block_index]
        answer = self.qa_agent.answer(question, selected_blocks, self.history)

        self.current_selected_ids = selected_ids
        self.history.append(ConversationTurn(question=question, answer=answer))

        return {
            "question": question,
            "answer": answer,
            "selected_block_ids": selected_ids,
            "selected_blocks": [b.text for b in selected_blocks],
            "retrieved_block_ids": [b.block_id for b in retrieved],
        }


# ==============================
# Demo / CLI
# ==============================


DEMO_DOC = """
Project Aurora is an internal retrieval system for enterprise document search.
The first prototype was built in 2023 and used only BM25 retrieval.
In March 2024, the team added a reranker based on a cross-encoder model.
The reranker improved top-5 answer accuracy by 11 percent on the internal benchmark.
The production rollout happened in July 2024 for the finance and legal verticals.
Latency initially increased by 180 milliseconds, but a caching layer reduced most of that overhead.
The current system stores document metadata, access control tags, and chunk-level embeddings.
Only authorized documents may be shown to end users.
A follow-up project, Aurora Memory, was proposed in January 2025.
Aurora Memory keeps only selected verbatim evidence blocks across multi-turn QA sessions.
The main motivation was to improve non-prefix KV cache reuse while avoiding quality loss from summarization.
One experiment used Llama 3.1 8B served through vLLM with online HTTP serving.
In those tests, the memory manager kept document blocks that were directly relevant to the active question.
The memory manager did not rewrite or compress document text.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-agent doc QA prototype over vLLM HTTP serving")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "token-abc123"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--doc-path", default=None, help="Path to a plain text document")
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--memory-k", type=int, default=6)
    return parser.parse_args()


def load_document(doc_path: Optional[str]) -> str:
    if doc_path is None:
        return DEMO_DOC
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    args = parse_args()
    llm = VLLMClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    doc_text = load_document(args.doc_path)
    app = TwoAgentDocQA(
        llm=llm,
        doc_text=doc_text,
        retrieval_k=args.retrieval_k,
        memory_k=args.memory_k,
    )

    print("Loaded document blocks:", len(app.store.blocks))
    print("Enter questions. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Q> ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"exit", "quit"}:
            break

        result = app.ask(question)
        print("\nSelected block IDs:")
        for bid in result["selected_block_ids"]:
            print(" -", bid)
        print("\nAnswer:")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
