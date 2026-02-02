"""
dynamic_multi_agent_orchestrator.py

- Workers may REQUEST branching (structured JSON).
- Runner DOES NOT interrupt immediately.
- Runner finishes the current plan (so you always get join/writer/evaluator).
- After the plan finishes, if any worker requested branching, runner replans.

Other properties:
- ONE orchestrator emits WorkflowSpec: entry, nodes, edges, stop, error_policy
- multiple branches with optional tools inside branches
- strict max depth per branch
- NO consensus node
- looping ONLY via evaluate -> refine -> orchestrator replan (evaluator fail)
- saves trace to traces/seed<seed>_<trace_num>.txt (auto-increment)

Python 3.10+
"""

from __future__ import annotations

import ast
import json
import math
import os
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import argparse

from utils import (
    _is_plan_start,
    _short_node_label,
    _parse_branch_start_payload,
    build_plan_flow_visualizations,
    State,
)

# =============================================================================
# 0) Utilities
# =============================================================================

class Dot:
    def __init__(self, obj: Any):
        self._obj = obj

    def __getattr__(self, name: str) -> Any:
        if isinstance(self._obj, dict) and name in self._obj:
            val = self._obj[name]
            return Dot(val) if isinstance(val, dict) else val
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"Dot({self._obj!r})"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def next_trace_path(dirpath: str = "traces", prefix: str = "", suffix: str = ".txt") -> str:
    ensure_dir(dirpath)
    max_n = -1
    for fn in os.listdir(dirpath):
        if not fn.endswith(suffix):
            continue
        stem = fn[: -len(suffix)]
        if prefix and not stem.startswith(prefix):
            continue
        stem2 = stem[len(prefix):] if prefix else stem
        if stem2.isdigit():
            max_n = max(max_n, int(stem2))
    return os.path.join(dirpath, f"{prefix}{max_n + 1}{suffix}")


# =============================================================================
# 1) Backends (REPLACE THESE)
# =============================================================================

ORCHESTRATOR_SYSTEM = (
    "You are a workflow ORCHESTRATOR. "
    "Return ONLY valid JSON for WorkflowSpec with keys: "
    "entry,nodes,edges,stop,error_policy. No markdown."
)


def llm_call(system: str, prompt: str, temperature: float = 0.2) -> str:
    """
    Replace with real LLM. Stub behavior:
    - Orchestrator emits a plan JSON (seeded).
    - Worker/writer produce short text.
    """
    if system == ORCHESTRATOR_SYSTEM:
        return orchestrator_stub_emit_plan(prompt)

    head = f"[{system}]"
    body = prompt.strip().replace("\n", " ")
    return f"{head} {body[:750]}..."


def web_search(query: str) -> List[Dict[str, str]]:
    return [
        {"title": "Example result 1", "url": "https://example.com/1", "snippet": f"Top snippet for: {query}"},
        {"title": "Example result 2", "url": "https://example.com/2", "snippet": f"More about: {query}"},
    ]


def run_python(code: str) -> Dict[str, Any]:
    safe_builtins = {"range": range, "len": len, "sum": sum, "min": min, "max": max, "print": print}
    glb = {"__builtins__": safe_builtins, "math": math}
    loc: Dict[str, Any] = {}
    try:
        exec(code, glb, loc)  # noqa: S102
        return {"ok": True, "locals": {k: str(v)[:200] for k, v in loc.items()}}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


# =============================================================================
# 2) Calculator (safe AST evaluator)
# =============================================================================

_ALLOWED_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
    ast.USub: lambda a: -a,
}


def safe_calc(expr: str) -> float:
    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return float(_ALLOWED_OPS[type(n.op)](_eval(n.operand)))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return float(_ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right)))
        raise ValueError("Disallowed expression")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


# =============================================================================
# 3) State + Trace
# =============================================================================


def _log(state: State, node_id: str, node_type: str, ok: bool, ms: int, **fields: Any) -> None:
    entry = {"node": node_id, "type": node_type, "ok": ok, "ms": ms}
    entry.update(fields)
    state.trace.append(entry)


# =============================================================================
# 4) Plan parsing + validation
# =============================================================================

REQUIRED_PLAN_KEYS = ["entry", "nodes", "edges", "stop", "error_policy"]


def parse_json_plan(raw: str) -> Dict[str, Any]:
    try:
        plan = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Invalid JSON from orchestrator: {e}")
    for k in REQUIRED_PLAN_KEYS:
        if k not in plan:
            raise ValueError(f"Plan missing key: {k}")
    if not isinstance(plan["nodes"], list) or not isinstance(plan["edges"], list):
        raise ValueError("nodes/edges must be lists")
    return plan


def index_nodes(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for n in plan["nodes"]:
        if "id" not in n or "type" not in n:
            raise ValueError("Each node needs id and type")
        if n["id"] in idx:
            raise ValueError(f"Duplicate node id: {n['id']}")
        idx[n["id"]] = n
    idx["__END__"] = {"id": "__END__", "type": "end", "params": {}}
    if plan["entry"] not in idx:
        raise ValueError("entry not found in nodes")
    return idx


def build_outgoing(plan: Dict[str, Any], nodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    outgoing: Dict[str, List[Dict[str, Any]]] = {}
    for e in plan["edges"]:
        for k in ["from", "to", "condition"]:
            if k not in e:
                raise ValueError("edge missing from/to/condition")
        if e["from"] not in nodes:
            raise ValueError(f"edge.from unknown: {e['from']}")
        if e["to"] not in nodes:
            raise ValueError(f"edge.to unknown: {e['to']}")
        outgoing.setdefault(e["from"], []).append(e)
    return outgoing


# =============================================================================
# 5) Condition DSL
# =============================================================================

def eval_condition(cond: str, state: State) -> bool:
    cond = cond.strip()
    if cond == "otherwise":
        return True

    env = {
        "true": True,
        "false": False,
        "state": Dot({
            "flags": state.flags,
            "memory": state.memory,
            "stop": state.stop,
        }),
    }
    return bool(eval(cond, {"__builtins__": {}}, env))  # noqa: S307


# =============================================================================
# 6) Node executors
# =============================================================================

def exec_llm_orchestrator(node: Dict[str, Any], state: State) -> Any:
    state.memory.setdefault("search_query", state.goal)
    state.memory.setdefault("calc_expr", "2*(3+4)")
    state.memory.setdefault("python_code", "x=1+1\n")
    return {"seeded": True}


def exec_web_search(node: Dict[str, Any], state: State) -> Any:
    q = str(state.memory.get("search_query", state.goal))
    results = web_search(q)
    scope = node.get("params", {}).get("scope")
    if scope:
        state.memory.setdefault("branch_tools", {}).setdefault(scope, {})["search_results"] = results
    else:
        state.memory["search_results"] = results
    return {"query": q, "n_results": len(results)}


def exec_calculator(node: Dict[str, Any], state: State) -> Any:
    expr = str(state.memory.get("calc_expr", "1+1"))
    val = safe_calc(expr)
    scope = node.get("params", {}).get("scope")
    if scope:
        state.memory.setdefault("branch_tools", {}).setdefault(scope, {})["calc_result"] = val
    else:
        state.memory["calc_result"] = val
    return {"expr": expr, "result": val}


def exec_code_exec(node: Dict[str, Any], state: State) -> Any:
    code = str(state.memory.get("python_code", "x=2+2\n"))
    out = run_python(code)
    if not out.get("ok"):
        raise RuntimeError(out.get("error", "code_exec failed"))
    scope = node.get("params", {}).get("scope")
    if scope:
        state.memory.setdefault("branch_tools", {}).setdefault(scope, {})["code_output"] = out
    else:
        state.memory["code_output"] = out
    return {"ok": True, "locals": out.get("locals", {})}


def _maybe_emit_branch_request(state: State, name: str, scope: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Simulator-only: deterministically decide whether a worker requests branching.
    Real implementation: model decides and outputs JSON.
    """
    if not state.memory.get("workers_can_request_branch", True):
        return None

    base_seed = int(state.memory.get("seed", 0))
    h = abs(hash((name, scope, base_seed, state.replan_loops))) % 10_000
    rnd = random.Random(h)

    if rnd.random() < 0.18:
        max_br = int(state.memory.get("max_branches", 4))
        k = rnd.randint(2, max(2, max_br))
        return {
            "type": "branch",
            "k": k,
            "reason": f"{name} wants parallel exploration (simulated).",
            "scope": scope,
        }
    return None


def exec_llm_worker(node: Dict[str, Any], state: State) -> Any:
    name = node.get("params", {}).get("name", node.get("id", "worker"))
    stance = node.get("params", {}).get("stance", "")
    scope = node.get("params", {}).get("scope")

    branch_ctx = state.memory.get("branch_tools", {}).get(scope, {}) if scope else {}
    ctx = {
        "scope": scope,
        "branch_ctx": branch_ctx,
        "replan_loops": state.replan_loops,
        "seed": state.memory.get("seed"),
        "latest_branch_request": (state.memory.get("branch_requests") or [None])[-1],
    }
    prompt = (
        f"You are {name}. Stance: {stance}\n"
        f"Goal: {state.goal}\n"
        f"Context: {json.dumps(ctx)[:2500]}\n"
        "If you believe branching is necessary, output JSON:\n"
        "{\"request\":{\"type\":\"branch\",\"k\":<int>,\"reason\":\"...\"}}\n"
        "Otherwise write your contribution.\n"
    )

    # Simulator: sometimes emit a branch request as JSON
    req = _maybe_emit_branch_request(state, name, scope)
    if req:
        resp = json.dumps({"request": req})
    else:
        resp = llm_call(str(name), prompt, temperature=0.4)

    # Parse request JSON (Option A)
    request_obj: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(resp)
        if isinstance(obj, dict) and isinstance(obj.get("request"), dict):
            request_obj = obj["request"]
    except Exception:
        request_obj = None

    if request_obj:
        state.memory.setdefault("branch_requests", []).append(request_obj)
        # IMPORTANT (Fix 1): do NOT interrupt execution.
        state.flags["replan_requested"] = True
        state.flags["replan_reason"] = f"Worker {name} requested branching"
        # store for visibility
        state.memory.setdefault("worker_outputs", {})[name] = resp
        if scope:
            state.memory.setdefault("branch_outputs", {}).setdefault(scope, {})[name] = resp
        return {"worker": name, "scope": scope, "request": request_obj}

    # Normal worker output
    if scope:
        state.memory.setdefault("branch_outputs", {}).setdefault(scope, {})[name] = resp
    state.memory.setdefault("worker_outputs", {})[name] = resp
    return {"worker": name, "scope": scope, "text": resp[:220]}


def exec_branch(node: Dict[str, Any], state: State) -> Any:
    branch_id = node.get("id", "branch")
    k = int(node.get("params", {}).get("k", 2))
    state.memory.setdefault("branches", {})[branch_id] = {"k": k, "started_at": time.time()}
    return {"branch_id": branch_id, "k": k}


def exec_join(node: Dict[str, Any], state: State) -> Any:
    join_id = node.get("id", "join")
    branches = state.memory.get("branch_outputs", {})
    parts: List[str] = []
    for scope, outs in branches.items():
        for worker_name, text in outs.items():
            parts.append(f"[{scope}::{worker_name}] {text}")
    aggregated = "\n\n".join(parts).strip()
    state.memory["aggregated"] = aggregated
    state.memory.setdefault("draft", aggregated)
    return {"join_id": join_id, "n_parts": len(parts)}


def exec_writer(node: Dict[str, Any], state: State) -> Any:
    aggregated = state.memory.get("aggregated", "")
    prompt = (
        "You are the final writer.\n"
        f"Goal: {state.goal}\n"
        f"Aggregated branch notes:\n{aggregated[:6000]}\n\n"
        "Write a clean final answer. Be structured and concise.\n"
    )
    draft = llm_call("writer", prompt, temperature=0.2)
    state.memory["draft"] = draft
    return {"draft_len": len(draft)}


def exec_evaluate(node: Dict[str, Any], state: State) -> Any:
    draft = str(state.memory.get("draft", "")).strip()
    satisfied = bool(draft) and (len(draft) > 120)
    missing: List[str] = []
    if not satisfied:
        missing.append("Draft seems incomplete/too short (sim heuristic).")
    state.flags["satisfied"] = satisfied
    state.memory["eval_missing"] = missing
    return {"satisfied": satisfied, "missing": missing}


def exec_refine(node: Dict[str, Any], state: State) -> Any:
    missing = state.memory.get("eval_missing", [])
    state.memory["refine_notes"] = {
        "missing": missing,
        "hint": "Replan: adjust branches/tools/workers within constraints.",
    }
    return {"missing": missing}


EXECUTORS = {
    "llm_orchestrator": exec_llm_orchestrator,
    "branch": exec_branch,
    "web_search": exec_web_search,
    "calculator": exec_calculator,
    "code_exec": exec_code_exec,
    "llm_worker": exec_llm_worker,
    "join": exec_join,
    "writer": exec_writer,
    "evaluate": exec_evaluate,
    "refine": exec_refine,
    "end": lambda n, s: None,
}


# =============================================================================
# 7) Error handling policies
# =============================================================================

def get_error_policy(plan: Dict[str, Any], node_type: str) -> Dict[str, Any]:
    return dict(plan.get("error_policy", {}).get(node_type, {}))


def repair_code(state: State, error: str) -> None:
    code = str(state.memory.get("python_code", ""))
    prompt = (
        "Fix the python code based on the error.\n"
        f"Error: {error}\n"
        f"Code:\n{code}\n"
        "Return ONLY corrected python code (no markdown).\n"
    )
    fixed = llm_call("CodeRepair", prompt, temperature=0.2)
    state.memory["python_code"] = fixed


# =============================================================================
# 8) Runner: interpret plan (fanout)
# =============================================================================

def _exec_single_node(
    plan: Dict[str, Any],
    nodes: Dict[str, Dict[str, Any]],
    state: State,
    node_id: str,
    retries: Dict[str, int],
) -> None:
    state.check_budget()

    node = nodes[node_id]
    ntype = node["type"]
    executor = EXECUTORS.get(ntype)
    if executor is None:
        raise ValueError(f"No executor for node type: {ntype}")

    t_start = time.time()
    try:
        result = executor(node, state)
        ms = int((time.time() - t_start) * 1000)
        _log(state, node_id, ntype, True, ms, result=str(result)[:750])
    except Exception as e:
        ms = int((time.time() - t_start) * 1000)
        _log(state, node_id, ntype, False, ms, error=repr(e))

        pol = get_error_policy(plan, ntype)
        action = pol.get("on_error", "abort")
        max_retries = int(pol.get("max_retries", 0))
        retries[node_id] = retries.get(node_id, 0) + 1

        if action == "retry" and retries[node_id] <= max_retries:
            return _exec_single_node(plan, nodes, state, node_id, retries)
        if action == "skip_node":
            return
        if action == "repair_and_retry" and ntype == "code_exec" and retries[node_id] <= max_retries:
            repair_code(state, repr(e))
            return _exec_single_node(plan, nodes, state, node_id, retries)
        raise


def _pick_next(outgoing: Dict[str, List[Dict[str, Any]]], current: str, state: State) -> str:
    next_node = "__END__"
    for e in outgoing.get(current, []):
        if eval_condition(str(e["condition"]), state):
            next_node = str(e["to"])
            break
    return next_node


def _run_branch_subpath(
    plan: Dict[str, Any],
    nodes: Dict[str, Dict[str, Any]],
    outgoing: Dict[str, List[Dict[str, Any]]],
    state: State,
    start: str,
    stop_at: str,
    depth_limit: int,
    retries: Dict[str, int],
) -> None:
    cur = start
    depth = 0
    while cur != "__END__" and cur != stop_at:
        if depth >= depth_limit:
            _log(state, "__BRANCH__", "branch_depth_cap", True, 0, result=f"Stopped at depth_limit={depth_limit} before {cur}")
            return
        _exec_single_node(plan, nodes, state, cur, retries)
        depth += 1
        cur = _pick_next(outgoing, cur, state)


def run_plan(plan: Dict[str, Any], state: State) -> State:
    nodes = index_nodes(plan)
    outgoing = build_outgoing(plan, nodes)
    state.stop = plan.get("stop", {})

    current = plan["entry"]
    retries: Dict[str, int] = {}

    while current != "__END__":
        ntype = nodes[current]["type"]

        if ntype == "branch":
            _exec_single_node(plan, nodes, state, current, retries)

            starts: List[str] = []
            for e in outgoing.get(current, []):
                if eval_condition(str(e["condition"]), state):
                    starts.append(str(e["to"]))

            depth_limit = int(nodes[current].get("params", {}).get("max_branch_depth", 4))
            join_id = str(nodes[current].get("params", {}).get("join_id", "join"))

            for s in starts:
                _log(state, "__BRANCH__", "branch_start", True, 0, result=f"start={s} stop_at={join_id} depth={depth_limit}")
                _run_branch_subpath(plan, nodes, outgoing, state, s, join_id, depth_limit, retries)

            current = join_id
            continue

        _exec_single_node(plan, nodes, state, current, retries)
        current = _pick_next(outgoing, current, state)

    return state


# =============================================================================
# 9) Orchestrator: prompt + fetch plan
# =============================================================================

def orchestrator_prompt(
    goal: str,
    tool_catalog: List[str],
    *,
    seed: int,
    max_branches: int,
    max_branch_depth: int,
    branch_request: Optional[Dict[str, Any]] = None,
    last_error: Optional[str] = None,
    last_plan_raw: Optional[str] = None,
    refine_notes: Optional[Dict[str, Any]] = None,
) -> str:
    schema = {
        "entry": "node_id",
        "nodes": [{"id": "node_id", "type": "tool_type", "params": {}}],
        "edges": [{"from": "node_id", "to": "node_id", "condition": "expr|otherwise"}],
        "stop": {"max_steps": 200, "max_seconds": 30, "max_replan_loops": 3},
        "error_policy": {"tool_type": {"on_error": "retry|skip_node|repair_and_retry|abort", "max_retries": 2}},
    }

    msg = (
        f"Seed: {seed}\n"
        f"Goal: {goal}\n"
        f"Available node types: {tool_catalog}\n"
        f"Constraints:\n"
        f"- branches: 1..{max_branches}\n"
        f"- max depth per branch (excluding branch/join): {max_branch_depth}\n"
        f"- branches may include optional tools per branch: web_search/calculator/code_exec\n"
        f"- no consensus node\n"
        f"- plan should be a DAG; looping happens outside via replan after evaluate/refine\n\n"
        f"JSON schema example (shape only): {json.dumps(schema)}\n"
    )
    if branch_request:
        msg += f"\nWorker branch request: {json.dumps(branch_request)[:1500]}\n"
    if refine_notes:
        msg += f"\nRefine notes: {json.dumps(refine_notes)[:1500]}\n"
    if last_error:
        msg += f"\nPrevious error: {last_error}\n"
    if last_plan_raw:
        msg += f"\nPrevious bad plan output: {last_plan_raw[:1200]}\n"
    return msg


def get_plan_from_orchestrator(
    goal: str,
    *,
    seed: int,
    max_attempts: int = 3,
    max_branches: int = 3,
    max_branch_depth: int = 4,
    branch_request: Optional[Dict[str, Any]] = None,
    refine_notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tool_catalog = [
        "llm_orchestrator",
        "branch",
        "web_search",
        "calculator",
        "code_exec",
        "llm_worker",
        "join",
        "writer",
        "evaluate",
        "refine",
        "end",
    ]

    last_error = None
    last_raw = None

    for _ in range(max_attempts):
        prompt = orchestrator_prompt(
            goal,
            tool_catalog,
            seed=seed,
            max_branches=max_branches,
            max_branch_depth=max_branch_depth,
            branch_request=branch_request,
            last_error=last_error,
            last_plan_raw=last_raw,
            refine_notes=refine_notes,
        )
        raw = llm_call(ORCHESTRATOR_SYSTEM, prompt, temperature=0.1)
        last_raw = raw
        try:
            plan = parse_json_plan(raw)
            nodes = index_nodes(plan)
            _ = build_outgoing(plan, nodes)
            return plan
        except Exception as e:
            last_error = repr(e)

    raise RuntimeError(f"Failed to obtain valid plan after {max_attempts} attempts. Last error: {last_error}")


# =============================================================================
# 10) Orchestrator stub plan emitter (seeded + responds to worker branch request)
# =============================================================================

def orchestrator_stub_emit_plan(prompt: str) -> str:
    seed = 0
    max_branches = 3
    max_branch_depth = 4
    requested_k: Optional[int] = None

    for line in prompt.splitlines():
        if line.startswith("Seed:"):
            try:
                seed = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        if "branches:" in line and ".." in line:
            try:
                rhs = line.split("..", 1)[1].strip()
                max_branches = int(rhs)
            except Exception:
                pass
        if "max depth per branch" in line:
            try:
                max_branch_depth = int(line.rsplit(":", 1)[1].strip())
            except Exception:
                pass
        if line.startswith("Worker branch request:"):
            try:
                payload = line.split(":", 1)[1].strip()
                obj = json.loads(payload)
                if isinstance(obj, dict) and obj.get("type") == "branch":
                    requested_k = int(obj.get("k", 0)) or None
            except Exception:
                pass

    goal = ""
    for line in prompt.splitlines():
        if line.startswith("Goal:"):
            goal = line.split(":", 1)[1].strip()
            break

    rnd = random.Random(seed)

    if requested_k is not None:
        b = max(1, min(max_branches, requested_k))
    else:
        b = rnd.randint(1, max(1, max_branches))

    nodes: List[Dict[str, Any]] = [
        {"id": "router", "type": "llm_orchestrator", "params": {}},
        {"id": "fanout", "type": "branch", "params": {"k": b, "max_branch_depth": max_branch_depth, "join_id": "join"}},
    ]
    edges: List[Dict[str, Any]] = [
        {"from": "router", "to": "fanout", "condition": "otherwise"},
    ]

    stance_pool = ["Practical", "Skeptical", "Creative", "Performance", "Systems"]

    for i in range(b):
        scope = f"fanout:{i}"

        tools: List[Tuple[str, str]] = []
        if rnd.random() < 0.55:
            tools.append((f"{scope}:search", "web_search"))
        if rnd.random() < 0.35:
            tools.append((f"{scope}:calc", "calculator"))
        if rnd.random() < 0.25:
            tools.append((f"{scope}:code", "code_exec"))

        remaining = max(1, max_branch_depth - len(tools))
        n_workers = rnd.randint(1, min(3, remaining))

        chain_ids: List[str] = []
        for nid, ntype in tools:
            nodes.append({"id": nid, "type": ntype, "params": {"scope": scope}})
            chain_ids.append(nid)

        for w in range(n_workers):
            wid = f"{scope}:w{w}"
            stance = stance_pool[(i + w + seed) % len(stance_pool)]
            nodes.append({"id": wid, "type": "llm_worker", "params": {"name": wid, "stance": stance, "scope": scope}})
            chain_ids.append(wid)

        if not chain_ids:
            wid = f"{scope}:w0"
            stance = stance_pool[(i + seed) % len(stance_pool)]
            nodes.append({"id": wid, "type": "llm_worker", "params": {"name": wid, "stance": stance, "scope": scope}})
            chain_ids.append(wid)

        edges.append({"from": "fanout", "to": chain_ids[0], "condition": "otherwise"})
        for a, b2 in zip(chain_ids, chain_ids[1:]):
            edges.append({"from": a, "to": b2, "condition": "otherwise"})
        edges.append({"from": chain_ids[-1], "to": "join", "condition": "otherwise"})

    nodes += [
        {"id": "join", "type": "join", "params": {}},
        {"id": "writer", "type": "writer", "params": {}},
        {"id": "evaluator", "type": "evaluate", "params": {}},
        {"id": "refine", "type": "refine", "params": {}},
    ]
    edges += [
        {"from": "join", "to": "writer", "condition": "otherwise"},
        {"from": "writer", "to": "evaluator", "condition": "otherwise"},
        {"from": "evaluator", "to": "__END__", "condition": "state.flags.satisfied == true"},
        {"from": "evaluator", "to": "refine", "condition": "otherwise"},
        {"from": "refine", "to": "__END__", "condition": "otherwise"},
    ]

    plan = {
        "entry": "router",
        "nodes": nodes,
        "edges": edges,
        "stop": {"max_steps": 500, "max_seconds": 30, "max_replan_loops": 3},
        "error_policy": {
            "llm_orchestrator": {"on_error": "retry", "max_retries": 1},
            "branch": {"on_error": "retry", "max_retries": 1},
            "web_search": {"on_error": "skip_node"},
            "calculator": {"on_error": "skip_node"},
            "code_exec": {"on_error": "repair_and_retry", "max_retries": 2},
            "llm_worker": {"on_error": "retry", "max_retries": 1},
            "join": {"on_error": "retry", "max_retries": 1},
            "writer": {"on_error": "retry", "max_retries": 1},
            "evaluate": {"on_error": "retry", "max_retries": 1},
            "refine": {"on_error": "retry", "max_retries": 1},
        },
    }
    return json.dumps(plan)


# =============================================================================
# 11) Top-level loop: evaluator failure OR queued worker branch-request => replan
# =============================================================================

def run_dynamic_workflow(
    goal: str,
    *,
    seed: int,
    max_branches: int = 4,
    max_branch_depth: int = 4,
    allow_replan: bool = True,
    max_replan_loops: int = 3,
) -> State:
    state = State(goal=goal)
    state.memory["goal"] = goal
    state.memory["seed"] = seed
    state.memory["max_branches"] = max_branches
    state.memory["max_branch_depth"] = max_branch_depth
    state.memory["workers_can_request_branch"] = True

    refine_notes: Optional[Dict[str, Any]] = None
    branch_request: Optional[Dict[str, Any]] = None

    while True:
        # reset per-iteration flags
        state.flags.pop("replan_requested", None)
        state.flags.pop("replan_reason", None)

        plan = get_plan_from_orchestrator(
            goal,
            seed=seed + state.replan_loops,  # vary topology across replans
            max_attempts=3,
            max_branches=max_branches,
            max_branch_depth=max_branch_depth,
            branch_request=branch_request,
            refine_notes=refine_notes,
        )
        state.stop = plan.get("stop", {})

        try:
            run_plan(plan, state)
        except Exception as e:
            _log(state, "__RUNNER__", "runner", False, 0, error=repr(e))
            if not allow_replan or state.replan_loops >= max_replan_loops:
                raise
            state.replan_loops += 1
            refine_notes = {"missing": ["execution_error"], "error": repr(e)}
            branch_request = None
            continue

        # --- Fix 1: if ANY worker requested branching, replan AFTER finishing the plan ---
        if state.flags.get("replan_requested", False):
            _log(state, "__RUNNER__", "replan_queued", True, 0, result=state.flags.get("replan_reason", "worker requested replan"))
            if not allow_replan or state.replan_loops >= max_replan_loops:
                return state
            state.replan_loops += 1
            branch_request = (state.memory.get("branch_requests") or [None])[-1]
            refine_notes = {"reason": "worker_requested_branch", "branch_request": branch_request}
            continue

        # satisfied?
        if state.flags.get("satisfied", False):
            return state

        # evaluator failed -> refine ran and we replan
        if not allow_replan or state.replan_loops >= max_replan_loops:
            return state

        state.replan_loops += 1
        refine_notes = state.memory.get("refine_notes", {"missing": ["unknown"]})
        branch_request = None


# =============================================================================
# 12) Trace saving
# =============================================================================

def format_trace(state: State) -> str:
    lines: List[str] = []
    lines.append(f"GOAL: {state.goal}")
    lines.append(f"SEED: {state.memory.get('seed')}  STEPS: {len(state.trace)}  REPLAN_LOOPS: {state.replan_loops}")
    lines.append(f"Max Branches: {state.memory.get('max_branches')}  Max Branch Depth: {state.memory.get('max_branch_depth')}")
    lines.append("")
    for i, t in enumerate(state.trace):
        ok = "OK" if t.get("ok") else "ERR"
        node = str(t.get("node"))
        typ = str(t.get("type"))
        ms = int(t.get("ms", 0))
        payload = str(t.get("result") or t.get("error") or "")
        lines.append(f"{i:04d} [{ok}] {node:<22} :: {typ:<18} {ms:>5}ms  {payload[:260]}")
    lines.append("")
    lines.append("BRANCH_REQUESTS:")
    lines.append(json.dumps(state.memory.get("branch_requests", []), indent=2)[:6000])
    lines.append("")
    lines.append("FINAL_DRAFT:")
    lines.append(str(state.memory.get("draft", "")))
    lines.append("")
    lines.append(build_plan_flow_visualizations(state))
    return "\n".join(lines)

def save_trace(state: State, dirpath: str = "traces") -> str:
    base = next_trace_path(dirpath=dirpath)
    seed = state.memory.get("seed", 0)
    path = os.path.join(dirpath, f"seed{seed}_" + os.path.basename(base))
    txt = format_trace(state)
    ensure_dir(dirpath)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic Workflow Orchestrator Demo")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducibility")
    return parser.parse_args()


# =============================================================================
# 13) Demo
# =============================================================================

def main() -> None:
    goal = "One orchestrator emits a plan with multiple branches (workers + optional tools). Loop only if evaluator fails."
    
    args = parse_args()
    seed = args.seed
    rnd = random.Random(seed)
    max_branches = rnd.randint(2, 10)
    max_branch_depth = rnd.randint(2, 6)
    max_replan_loops = rnd.randint(1, 5)

    state = run_dynamic_workflow(
        goal,
        seed=seed,
        max_branches=max_branches,
        max_branch_depth=max_branch_depth,
        allow_replan=True,
        max_replan_loops=max_replan_loops,
    )
    out = save_trace(state, dirpath="traces")
    print(f"Saved trace to: {out}")
    print("\n=== FINAL DRAFT ===")
    print(state.memory.get("draft", "(no draft)"))


if __name__ == "__main__":
    main()
