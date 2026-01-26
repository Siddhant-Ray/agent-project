from typing import DefaultDict, Any, Dict, List, Optional, Tuple
from collections import defaultdict
import re
import time
from dataclasses import dataclass, field


@dataclass
class State:
    goal: str
    memory: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    stop: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    steps: int = 0
    t0: float = field(default_factory=time.time)
    replan_loops: int = 0  # replans after evaluator failure or worker-branch requests

    def check_budget(self) -> None:
        self.steps += 1
        max_steps = int(self.stop.get("max_steps", 500))
        max_seconds = float(self.stop.get("max_seconds", 60))
        if self.steps > max_steps:
            raise RuntimeError(f"Budget exceeded: max_steps={max_steps}")
        if (time.time() - self.t0) > max_seconds:
            raise RuntimeError(f"Budget exceeded: max_seconds={max_seconds}")

def _is_plan_start(t: Dict[str, Any]) -> bool:
    # A new plan begins when the runner executes the plan's router node
    return t.get("node") == "router" and t.get("type") == "llm_orchestrator" and t.get("ok") is True

def _short_node_label(node_id: str) -> str:
    # Keep labels compact in the visualization
    # Example: fanout:7:w2 -> w2 (or fanout:7:w2 if you prefer)
    return node_id

def _parse_branch_start_payload(payload: str) -> Optional[Dict[str, str]]:
    """
    payload looks like: "start=fanout:0:w0 stop_at=join depth=6"
    We'll parse start + stop_at.
    """
    m = re.search(r"start=([^\s]+)\s+stop_at=([^\s]+)", payload)
    if not m:
        return None
    return {"start": m.group(1), "stop_at": m.group(2)}

def build_plan_flow_visualizations(state: State) -> str:
    """
    Builds a per-plan visualization from the EXECUTED TRACE (not the static plan).
    Shows:
      - execution order (with branch blocks)
      - inferred edges (prev -> next)
      - adjacency list
    """
    # Split trace into plans by detecting router execution
    plan_indices: List[int] = []
    for i, t in enumerate(state.trace):
        if _is_plan_start(t):
            plan_indices.append(i)

    if not plan_indices:
        return "FLOW_VISUALIZATION:\n(no plan starts detected)\n"

    plan_indices.append(len(state.trace))  # sentinel end

    out_lines: List[str] = []
    out_lines.append("FLOW_VISUALIZATION (per plan, execution-derived)")
    out_lines.append("Legend: A -> B means B executed immediately after A within the same context.\n")

    for p in range(len(plan_indices) - 1):
        start_i = plan_indices[p]
        end_i = plan_indices[p + 1]
        chunk = state.trace[start_i:end_i]

        out_lines.append(f"--- PLAN {p} (trace idx {start_i}..{end_i-1}) ---")

        # We'll track branch blocks using __BRANCH__ markers
        # When we see "__BRANCH__ :: branch_start", we start a new branch context
        current_branch: Optional[str] = None
        branch_stop_at: Optional[str] = None

        # Order list with indentation for branch contexts
        order_lines: List[str] = []
        edges: List[Tuple[str, str]] = []
        adjacency: DefaultDict[str, List[str]] = defaultdict(list)

        prev_node_global: Optional[str] = None
        prev_node_branch: Optional[str] = None

        def add_edge(a: Optional[str], b: str) -> None:
            if not a:
                return
            edges.append((a, b))
            if b not in adjacency[a]:
                adjacency[a].append(b)

        for t in chunk:
            node = str(t.get("node"))
            typ = str(t.get("type"))
            payload = str(t.get("result") or t.get("error") or "")

            if node == "__BRANCH__" and typ == "branch_start":
                info = _parse_branch_start_payload(payload)
                if info:
                    current_branch = info["start"]  # identify branch by its start node
                    branch_stop_at = info["stop_at"]
                else:
                    current_branch = "unknown_branch"
                    branch_stop_at = None
                order_lines.append(f"  [BRANCH START] {current_branch}  (stop_at={branch_stop_at})")
                prev_node_branch = None
                continue

            if node == "__BRANCH__" and typ == "branch_depth_cap":
                order_lines.append(f"    [BRANCH DEPTH CAP] {payload}")
                continue

            # Normal node
            if current_branch is None:
                # mainline (outside branch)
                order_lines.append(f"{_short_node_label(node)}")
                add_edge(prev_node_global, node)
                prev_node_global = node
            else:
                # inside a branch context until we hit join or a new branch_start
                order_lines.append(f"    {_short_node_label(node)}")
                add_edge(prev_node_branch, node)
                prev_node_branch = node

                # Optional: if we hit stop_at (usually join), end branch context.
                # In your runner, branch subpath stops before join (join executed in mainline),
                # so this typically won't trigger—but keeping it makes it robust.
                if branch_stop_at and node == branch_stop_at:
                    current_branch = None
                    branch_stop_at = None
                    prev_node_branch = None

        # Print execution order
        out_lines.append("Execution order:")
        out_lines.extend(["  " + s for s in order_lines])

        # Print inferred edges (compact)
        out_lines.append("\nInferred edges (execution-adjacent):")
        if edges:
            for a, b in edges:
                out_lines.append(f"  {a} -> {b}")
        else:
            out_lines.append("  (none)")

        # Print adjacency list (connectivity)
        out_lines.append("\nConnectivity (adjacency list):")
        if adjacency:
            for a in sorted(adjacency.keys()):
                out_lines.append(f"  {a}: {', '.join(adjacency[a])}")
        else:
            out_lines.append("  (none)")
        out_lines.append("")  # blank line between plans

    return "\n".join(out_lines)
