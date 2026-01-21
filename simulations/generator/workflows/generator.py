"""
dynamic_multi_agent_orchestrator.py

End-to-end reference implementation where:
- An LLM ORCHESTRATOR produces a WorkflowSpec (nodes, edges, stop rules, error policies).
- A generic runner validates + executes that spec.
- Tools available as nodes: web_search, calculator, code_exec, llm_worker, consensus, evaluate, refine.
- Orchestrator decides branching/loops/consensus/failure handling dynamically.
- On parse/validation failures, runner asks orchestrator to re-emit a fixed plan.
- On unrecoverable execution failures or evaluator dissatisfaction, runner can replan.

This file is backend-agnostic: swap in your real LLM + web search + sandbox.

Python 3.10+
"""

from __future__ import annotations

import ast
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

class Dot:
    """Wrap dicts so you can access keys as attributes: d.x instead of d['x']."""
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        if isinstance(self._obj, dict) and name in self._obj:
            val = self._obj[name]
            return Dot(val) if isinstance(val, dict) else val
        raise AttributeError(name)

    def __repr__(self):
        return f"Dot({self._obj!r})"


# =============================================================================
# 0) Backends (REPLACE THESE)
# =============================================================================

def llm_call(system: str, prompt: str, temperature: float = 0.2) -> str:
    """
    Replace with: vLLM (OpenAI-compatible), OpenAI API, etc.

    IMPORTANT:
    - In production, force JSON output via function calling / JSON schema.
    - This stub returns a plausible WorkflowSpec JSON for demonstration.
    """
    # NOTE: This is a simple deterministic stub that emits a workflow depending on keywords.
    goal = prompt.lower()
    needs_search = any(w in goal for w in ["latest", "cite", "citation", "source", "web", "paper", "arxiv", "news"])
    needs_code = any(w in goal for w in ["simulate", "python", "code", "benchmark", "plot", "compute by code"])
    needs_calc = any(w in goal for w in ["calculate", "compute", "probability", "latency", "throughput", "50^5"])

    # If task seems complex/uncertain, include consensus.
    needs_consensus = any(w in goal for w in ["compare", "tradeoff", "argue", "debate", "consensus", "alternatives"])

    # Orchestrator emits nodes/edges dynamically:
    nodes = [
        {"id": "router", "type": "llm_orchestrator", "params": {}},
    ]

    edges = []

    # optional tools
    if needs_search:
        nodes.append({"id": "search", "type": "web_search", "params": {}})
        edges.append({"from": "router", "to": "search", "condition": "state.flags.needs_search == true"})
    if needs_calc:
        nodes.append({"id": "calc", "type": "calculator", "params": {}})
        edges.append({"from": "router", "to": "calc", "condition": "state.flags.needs_calc == true"})
    if needs_code:
        nodes.append({"id": "code", "type": "code_exec", "params": {}})
        edges.append({"from": "router", "to": "code", "condition": "state.flags.needs_code == true"})

    # writers / workers
    if needs_consensus:
        nodes += [
            {"id": "w1", "type": "llm_worker", "params": {"name": "Worker-1", "stance": "Practical, implementable."}},
            {"id": "w2", "type": "llm_worker", "params": {"name": "Worker-2", "stance": "Skeptical, find failure modes."}},
            {"id": "w3", "type": "llm_worker", "params": {"name": "Worker-3", "stance": "Creative, propose variants."}},
            {"id": "consensus", "type": "consensus", "params": {}},
        ]
    else:
        nodes += [{"id": "writer", "type": "llm_worker", "params": {"name": "Writer", "stance": "Produce final answer."}}]

    nodes += [
        {"id": "evaluator", "type": "evaluate", "params": {}},
        {"id": "refine", "type": "refine", "params": {}},
    ]

    # Default edge from router if no tool conditions trigger
    if needs_consensus:
        edges.append({"from": "router", "to": "w1", "condition": "otherwise"})
    else:
        edges.append({"from": "router", "to": "writer", "condition": "otherwise"})

    # tools flow into first writer/worker
    if needs_search:
        edges.append({"from": "search", "to": ("w1" if needs_consensus else "writer"), "condition": "otherwise"})
    if needs_calc:
        edges.append({"from": "calc", "to": ("w1" if needs_consensus else "writer"), "condition": "otherwise"})
    if needs_code:
        edges.append({"from": "code", "to": ("w1" if needs_consensus else "writer"), "condition": "otherwise"})

    if needs_consensus:
        edges += [
            {"from": "w1", "to": "w2", "condition": "otherwise"},
            {"from": "w2", "to": "w3", "condition": "otherwise"},
            {"from": "w3", "to": "consensus", "condition": "otherwise"},
            {"from": "consensus", "to": "evaluator", "condition": "otherwise"},
        ]
    else:
        edges += [
            {"from": "writer", "to": "evaluator", "condition": "otherwise"},
        ]

    # Loop: if evaluator not satisfied -> refine -> router
    edges += [
        {"from": "evaluator", "to": "__END__", "condition": "state.flags.satisfied == true"},
        {"from": "evaluator", "to": "refine", "condition": "otherwise"},
        {"from": "refine", "to": "router", "condition": "state.flags.loop_count < state.stop.max_loops"},
        {"from": "refine", "to": "__END__", "condition": "otherwise"},
    ]

    plan = {
        "entry": "router",
        "nodes": nodes,
        "edges": edges,
        "stop": {"max_steps": 80, "max_seconds": 25, "max_loops": 3},
        "error_policy": {
            "llm_orchestrator": {"on_error": "retry", "max_retries": 2},
            "llm_worker": {"on_error": "retry", "max_retries": 2},
            "web_search": {"on_error": "skip_node"},
            "calculator": {"on_error": "skip_node"},
            "code_exec": {"on_error": "repair_and_retry", "max_retries": 2},
            "consensus": {"on_error": "retry", "max_retries": 1},
            "evaluate": {"on_error": "retry", "max_retries": 1},
            "refine": {"on_error": "retry", "max_retries": 1},
        },
    }
    return json.dumps(plan)


def web_search(query: str) -> List[Dict[str, str]]:
    """
    Replace with Tavily/SerpAPI/Playwright/DuckDuckGo, etc.
    """
    # Stub results
    return [
        {"title": "Example result 1", "url": "https://example.com/1", "snippet": f"Top snippet for: {query}"},
        {"title": "Example result 2", "url": "https://example.com/2", "snippet": f"More about: {query}"},
    ]


def run_python(code: str) -> Dict[str, Any]:
    """
    WARNING: Not a secure sandbox. For real safety, run in a container / separate process.
    """
    safe_builtins = {"range": range, "len": len, "sum": sum, "min": min, "max": max, "print": print}
    glb = {"__builtins__": safe_builtins, "math": math}
    loc: Dict[str, Any] = {}
    try:
        exec(code, glb, loc)  # noqa: S102
        # avoid returning huge objects
        return {"ok": True, "locals": {k: str(v)[:200] for k, v in loc.items()}}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


# =============================================================================
# 1) Calculator (safe AST evaluator)
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
        if isinstance(n, ast.Num):  # old python ast
            return float(n.n)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return float(_ALLOWED_OPS[type(n.op)](_eval(n.operand)))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return float(_ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right)))
        raise ValueError("Disallowed expression")
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


# =============================================================================
# 2) State + Trace
# =============================================================================

@dataclass
class State:
    goal: str
    memory: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    stop: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    steps: int = 0
    t0: float = field(default_factory=time.time)
    loop_count: int = 0

    def check_budget(self) -> None:
        self.steps += 1
        max_steps = int(self.stop.get("max_steps", 100))
        max_seconds = float(self.stop.get("max_seconds", 60))
        if self.steps > max_steps:
            raise RuntimeError(f"Budget exceeded: max_steps={max_steps}")
        if (time.time() - self.t0) > max_seconds:
            raise RuntimeError(f"Budget exceeded: max_seconds={max_seconds}")


# =============================================================================
# 3) Plan parsing + validation
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
# 4) Condition DSL
# =============================================================================

def eval_condition(cond: str, state: State) -> bool:
    cond = cond.strip()
    if cond == "otherwise":
        return True

    # Provide dot-access objects for state.flags/state.memory/state.stop
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
# 5) Node executors
# =============================================================================

def _log(state: State, node_id: str, node_type: str, ok: bool, ms: int, **fields: Any) -> None:
    entry = {"node": node_id, "type": node_type, "ok": ok, "ms": ms}
    entry.update(fields)
    state.trace.append(entry)

def exec_llm_orchestrator(node: Dict[str, Any], state: State) -> Any:
    """
    This is a *node inside the orchestrator-generated plan*.
    It can set routing flags or update tool inputs in memory.
    """
    goal = state.goal.lower()
    # Example: decide tool needs; in real life, this could itself call LLM.
    state.flags["needs_search"] = any(w in goal for w in ["latest", "cite", "citation", "source", "web", "paper", "news"])
    state.flags["needs_code"] = any(w in goal for w in ["simulate", "python", "code", "benchmark", "plot"])
    state.flags["needs_calc"] = any(w in goal for w in ["calculate", "compute", "probability", "latency", "throughput", "50^5"])

    # Also seed tool inputs if missing
    state.memory.setdefault("search_query", state.goal)
    state.memory.setdefault("calc_expr", "2*(3+4)")
    state.memory.setdefault("python_code", "x=1+1\n")

    return {"flags": dict(state.flags), "seeded": True}

def exec_web_search(node: Dict[str, Any], state: State) -> Any:
    q = str(state.memory.get("search_query", state.goal))
    results = web_search(q)
    state.memory["search_results"] = results
    return {"query": q, "results": results}

def exec_calculator(node: Dict[str, Any], state: State) -> Any:
    expr = str(state.memory.get("calc_expr", "1+1"))
    val = safe_calc(expr)
    state.memory["calc_result"] = val
    return {"expr": expr, "result": val}

def exec_code_exec(node: Dict[str, Any], state: State) -> Any:
    code = str(state.memory.get("python_code", "x=2+2\n"))
    out = run_python(code)
    state.memory["code_output"] = out
    if not out.get("ok"):
        raise RuntimeError(out.get("error", "code_exec failed"))
    return out

def exec_llm_worker(node: Dict[str, Any], state: State) -> Any:
    name = node.get("params", {}).get("name", "Worker")
    stance = node.get("params", {}).get("stance", "")

    ctx = {
        "search_results": state.memory.get("search_results"),
        "calc_result": state.memory.get("calc_result"),
        "code_output": state.memory.get("code_output"),
        "loop_count": state.loop_count,
    }
    prompt = (
        f"You are {name}. Stance: {stance}\n"
        f"Goal: {state.goal}\n"
        f"Context: {json.dumps(ctx)[:2000]}\n"
        f"Write your contribution. Be concise.\n"
    )
    resp = llm_call(name, prompt, temperature=0.4)

    # Store worker output
    workers = state.memory.setdefault("worker_outputs", {})
    workers[name] = resp

    # If it's the single writer, keep draft updated too
    if name.lower() in ["writer", "final_writer", "answer_writer"]:
        state.memory["draft"] = resp

    return {"worker": name, "text": resp}

def exec_consensus(node: Dict[str, Any], state: State) -> Any:
    outs = state.memory.get("worker_outputs", {})
    prompt = (
        "Merge the worker outputs into a single best answer.\n"
        f"Goal: {state.goal}\n"
        f"Worker outputs (JSON): {json.dumps(outs)[:6000]}\n"
        "Return a single merged answer.\n"
    )
    merged = llm_call("Consensus", prompt, temperature=0.2)
    state.memory["draft"] = merged
    return {"merged": merged, "n_workers": len(outs)}

def exec_evaluate(node: Dict[str, Any], state: State) -> Any:
    draft = str(state.memory.get("draft", ""))
    prompt = (
        "Evaluate if the draft fully satisfies the goal.\n"
        f"Goal: {state.goal}\n"
        f"Draft: {draft[:4000]}\n"
        "Return JSON: {\"satisfied\": true/false, \"missing\": [..]}.\n"
    )
    raw = llm_call("Evaluator", prompt, temperature=0.0)

    # Try parse; if parse fails, use heuristic fallback
    satisfied = False
    missing: List[str] = []
    try:
        obj = json.loads(raw)
        satisfied = bool(obj.get("satisfied", False))
        missing = list(obj.get("missing", []))
    except Exception:
        satisfied = bool(draft.strip()) and state.loop_count >= 1

    state.flags["satisfied"] = satisfied
    state.memory["eval_missing"] = missing
    return {"satisfied": satisfied, "missing": missing, "raw": raw}

def exec_refine(node: Dict[str, Any], state: State) -> Any:
    state.loop_count += 1
    missing = state.memory.get("eval_missing", [])
    draft = state.memory.get("draft", "")

    prompt = (
        "Refine the plan for the next iteration.\n"
        f"Goal: {state.goal}\n"
        f"Missing: {missing}\n"
        f"Current draft: {str(draft)[:1500]}\n"
        "Decide next actions: update search_query/calc_expr/python_code, "
        "or request consensus, or re-run tools.\n"
        "Return JSON with optional keys: search_query, calc_expr, python_code.\n"
    )
    raw = llm_call("Refiner", prompt, temperature=0.2)

    # Best-effort parse: update tool inputs
    try:
        obj = json.loads(raw)
        if isinstance(obj.get("search_query"), str):
            state.memory["search_query"] = obj["search_query"]
        if isinstance(obj.get("calc_expr"), str):
            state.memory["calc_expr"] = obj["calc_expr"]
        if isinstance(obj.get("python_code"), str):
            state.memory["python_code"] = obj["python_code"]
    except Exception:
        # fallback: do nothing
        pass

    state.flags["loop_count"] = state.loop_count
    return {"loop_count": state.loop_count, "raw": raw}

EXECUTORS = {
    "llm_orchestrator": exec_llm_orchestrator,
    "web_search": exec_web_search,
    "calculator": exec_calculator,
    "code_exec": exec_code_exec,
    "llm_worker": exec_llm_worker,
    "consensus": exec_consensus,
    "evaluate": exec_evaluate,
    "refine": exec_refine,
    "end": lambda n, s: None,
}


# =============================================================================
# 6) Error handling policies
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
# 7) Runner: interpret plan
# =============================================================================

def run_plan(plan: Dict[str, Any], state: State) -> State:
    nodes = index_nodes(plan)
    outgoing = build_outgoing(plan, nodes)
    state.stop = plan.get("stop", {})

    current = plan["entry"]
    retries: Dict[str, int] = {}

    while current != "__END__":
        state.check_budget()

        node = nodes[current]
        ntype = node["type"]
        executor = EXECUTORS.get(ntype)
        if executor is None:
            raise ValueError(f"No executor for node type: {ntype}")

        t_start = time.time()
        try:
            result = executor(node, state)
            ms = int((time.time() - t_start) * 1000)
            _log(state, current, ntype, True, ms, result=str(result)[:400])
        except Exception as e:
            ms = int((time.time() - t_start) * 1000)
            _log(state, current, ntype, False, ms, error=repr(e))

            pol = get_error_policy(plan, ntype)
            action = pol.get("on_error", "abort")
            max_retries = int(pol.get("max_retries", 0))
            retries[current] = retries.get(current, 0) + 1

            if action == "retry" and retries[current] <= max_retries:
                continue

            if action == "skip_node":
                # just proceed to routing from this node (it may have "otherwise" edges)
                pass
            elif action == "repair_and_retry" and retries[current] <= max_retries and ntype == "code_exec":
                repair_code(state, repr(e))
                continue
            else:
                raise

        # pick next node
        next_node = "__END__"
        for e in outgoing.get(current, []):
            if eval_condition(str(e["condition"]), state):
                next_node = str(e["to"])
                break
        current = next_node

    return state


# =============================================================================
# 8) Orchestrator: generate plan, validate, re-ask if needed
# =============================================================================

ORCHESTRATOR_SYSTEM = (
    "You are a workflow ORCHESTRATOR. "
    "Return ONLY valid JSON for WorkflowSpec with keys: "
    "entry,nodes,edges,stop,error_policy. No markdown."
)

def orchestrator_prompt(goal: str, tool_catalog: List[str], last_error: Optional[str] = None, last_plan_raw: Optional[str] = None) -> str:
    schema = {
        "entry": "node_id",
        "nodes": [{"id": "node_id", "type": "tool_type", "params": {}}],
        "edges": [{"from": "node_id", "to": "node_id", "condition": "expr|otherwise"}],
        "stop": {"max_steps": 80, "max_seconds": 25, "max_loops": 3},
        "error_policy": {"tool_type": {"on_error": "retry|skip_node|repair_and_retry|abort", "max_retries": 2}},
    }
    msg = (
        f"Goal: {goal}\n"
        f"Available tool node types: {tool_catalog}\n\n"
        "Decide dynamically:\n"
        "- what tools to use\n"
        "- whether consensus/fanout is needed\n"
        "- whether evaluate/refine looping is needed\n"
        "- failure handling (esp. code_exec repair)\n\n"
        "Condition language:\n"
        "- use 'otherwise' as default\n"
        "- use expressions like: state.flags.needs_search == true\n"
        "- use loop cap like: state.flags.loop_count < state.stop.max_loops\n\n"
        f"JSON schema example (shape only): {json.dumps(schema)}\n"
    )
    if last_error:
        msg += f"\nPrevious error: {last_error}\n"
    if last_plan_raw:
        msg += f"\nPrevious bad plan output: {last_plan_raw[:1200]}\n"
    return msg

def get_plan_from_orchestrator(goal: str, max_attempts: int = 3) -> Dict[str, Any]:
    tool_catalog = [
        "llm_orchestrator",
        "web_search",
        "calculator",
        "code_exec",
        "llm_worker",
        "consensus",
        "evaluate",
        "refine",
        "end",
    ]

    last_error = None
    last_raw = None
    for _ in range(max_attempts):
        prompt = orchestrator_prompt(goal, tool_catalog, last_error=last_error, last_plan_raw=last_raw)
        raw = llm_call(ORCHESTRATOR_SYSTEM, prompt, temperature=0.1)
        last_raw = raw
        try:
            plan = parse_json_plan(raw)
            # Basic structural validation: all edge refs exist
            nodes = index_nodes(plan)
            _ = build_outgoing(plan, nodes)
            return plan
        except Exception as e:
            last_error = repr(e)

    raise RuntimeError(f"Failed to obtain valid plan after {max_attempts} attempts. Last error: {last_error}")


# =============================================================================
# 9) Top-level run: plan -> execute -> optional replan
# =============================================================================

def run_dynamic_workflow(goal: str, allow_replan: bool = True, max_replans: int = 2) -> State:
    state = State(goal=goal)
    state.memory["goal"] = goal

    replans = 0
    while True:
        plan = get_plan_from_orchestrator(goal)
        state.stop = plan.get("stop", {})

        try:
            run_plan(plan, state)
        except Exception as e:
            # unrecoverable execution error -> ask orchestrator for new plan using current state
            state.trace.append({"node": "__RUNNER__", "type": "runner", "ok": False, "ms": 0, "error": repr(e)})
            if not allow_replan or replans >= max_replans:
                raise
            replans += 1
            # Keep looping: orchestrator might choose different tool strategy next time
            continue

        # If satisfied or cannot/shouldn't replan further, exit
        if state.flags.get("satisfied", False):
            return state
        if not allow_replan or replans >= max_replans:
            return state

        # If evaluator says not satisfied, let the orchestrator replan at top-level too
        replans += 1


# =============================================================================
# 10) Demo
# =============================================================================

def pretty_print_trace(state: State, limit: int = 50) -> None:
    print(f"\n=== TRACE (steps={len(state.trace)}, loops={state.loop_count}) ===")
    for i, t in enumerate(state.trace[:limit]):
        ok = "OK" if t.get("ok") else "ERR"
        print(f"{i:02d} [{ok}] {t.get('node'):>12} :: {t.get('type'):<14} {t.get('ms',0):>4}ms  "
              f"{(t.get('result') or t.get('error') or '')[:120]}")
    if len(state.trace) > limit:
        print(f"... ({len(state.trace)-limit} more)")

def main() -> None:
    goal = "Design an LLM-driven multi-agent workflow system with branching, loops, consensus, and tool-failure recovery."
    state = run_dynamic_workflow(goal, allow_replan=True, max_replans=1)
    pretty_print_trace(state)

    print("\n=== FINAL DRAFT ===")
    print(state.memory.get("draft", "(no draft produced)"))

if __name__ == "__main__":
    main()
# -----------------------------