from __future__ import annotations

import hashlib
import random

from overall_cost import compute_overall_compaction_curve, compute_overall_prefix_curve


def _stable_seed(*parts: object) -> int:
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _prefix_quality_points(num_turns: int) -> list[float]:
    """
    Starting quality is 100.

    For n turns total (including round 0):
    - up to n/4: linear drop of 1 point per round
    - n/4 to n/2: exponential drop at 5% per round
    - n/2 to 3n/4: exponential drop at 10% per round
    - 3n/4 to n: exponential drop at 20% per round

    Floors at 0.
    """
    if num_turns <= 0:
        return []

    q1 = num_turns // 4
    q2 = num_turns // 2
    q3 = (3 * num_turns) // 4

    qualities = []
    quality = 100.0

    for turn in range(num_turns):
        if turn == 0:
            qualities.append(quality)
            continue

        if turn <= q1:
            quality = max(0.0, quality - 1.0)
        elif turn <= q2:
            quality = max(0.0, quality * 0.95)
        elif turn <= q3:
            quality = max(0.0, quality * 0.90)
        else:
            quality = max(0.0, quality * 0.80)

        qualities.append(quality)

    return qualities


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _max_fractional_change(rate_percent: float) -> float:
    """
    Fine-grained interpolation for the compaction-quality model.

    Anchor points:
      0<r<=10   -> around 0.30
      r=30      -> 0.10
      r=50      -> 0.05
      r=80      -> 0.02
      r=100     -> 0.01

    Interpolate linearly between anchors instead of using hard jumps.
    """
    if rate_percent <= 0.0:
        return 0.0

    if rate_percent <= 10.0:
        return 0.30

    if rate_percent <= 30.0:
        return _lerp(rate_percent, 10.0, 30.0, 0.30, 0.10)

    if rate_percent <= 50.0:
        return _lerp(rate_percent, 30.0, 50.0, 0.10, 0.05)

    if rate_percent <= 80.0:
        return _lerp(rate_percent, 50.0, 80.0, 0.05, 0.02)

    if rate_percent <= 100.0:
        return _lerp(rate_percent, 80.0, 100.0, 0.02, 0.01)

    return 0.01


def _compaction_quality_points(
    num_turns: int,
    compaction_fraction: float,
    seed_key: str,
) -> list[float]:
    """
    Starting quality is 100.

    Quality can go up or down every round, within a compaction-rate-dependent
    percentage band of the previous round.

    Floors at 0 and caps at 100.
    If compaction rate is 0, quality is 0 throughout.
    """
    if num_turns <= 0:
        return []

    rate_percent = compaction_fraction * 100.0
    if rate_percent <= 0.0:
        return [0.0 for _ in range(num_turns)]

    max_change_fraction = _max_fractional_change(rate_percent)
    rng = random.Random(_stable_seed("cost_quality", seed_key, f"{compaction_fraction:.6f}"))

    qualities = []
    quality = 100.0

    for turn in range(num_turns):
        if turn == 0:
            qualities.append(quality)
            continue

        change_fraction = rng.uniform(-max_change_fraction, max_change_fraction)
        quality = min(100.0, max(0.0, quality * (1.0 + change_fraction)))
        qualities.append(quality)

    return qualities


def compute_cost_quality_prefix_curve(
    csv_path: str,
    target_session_id: str,
    model_name: str,
    gpu_name: str,
) -> dict:
    base = compute_overall_prefix_curve(csv_path, target_session_id, model_name, gpu_name)
    qualities = _prefix_quality_points(len(base["points"]))

    points = []
    for p, q in zip(base["points"], qualities):
        pt = dict(p)
        pt["overall_cost"] = pt["cumulative_cost"]
        pt["f1_score"] = q
        points.append(pt)

    result = dict(base)
    result["metric"] = "cost_quality"
    result["mode"] = "cost_quality_prefix"
    result["points"] = points
    return result


def compute_cost_quality_compaction_curve(
    csv_path: str,
    target_session_id: str,
    compaction_fraction: float,
    model_name: str,
    gpu_name: str,
) -> dict:
    base = compute_overall_compaction_curve(
        csv_path,
        target_session_id,
        compaction_fraction,
        model_name,
        gpu_name,
    )
    seed_key = f"{target_session_id}|{model_name}|{gpu_name}|{len(base['points'])}"
    qualities = _compaction_quality_points(
        len(base["points"]),
        compaction_fraction,
        seed_key,
    )

    points = []
    for p, q in zip(base["points"], qualities):
        pt = dict(p)
        pt["overall_cost"] = pt["cumulative_cost"]
        pt["f1_score"] = q
        points.append(pt)

    result = dict(base)
    result["metric"] = "cost_quality"
    result["mode"] = "cost_quality_compaction"
    result["points"] = points
    return result


def _average_quality(points: list[dict]) -> float:
    if not points:
        return 0.0
    return sum(float(p.get("f1_score", 0.0)) for p in points) / len(points)


def compute_avg_quality_prefix_point(
    csv_path: str,
    target_session_id: str,
    model_name: str,
    gpu_name: str,
) -> dict:
    base = compute_cost_quality_prefix_curve(csv_path, target_session_id, model_name, gpu_name)
    total_cost = float(base.get("total_overall_cost", 0.0))
    avg_f1 = _average_quality(base.get("points", []))
    result = dict(base)
    result["metric"] = "avg_quality_total_cost"
    result["mode"] = "avg_quality_prefix"
    result["points"] = [{
        "turn": len(base.get("points", [])) - 1,
        "overall_cost": total_cost,
        "avg_f1_score": avg_f1,
    }]
    result["avg_f1_score"] = avg_f1
    return result


def compute_avg_quality_compaction_point(
    csv_path: str,
    target_session_id: str,
    compaction_fraction: float,
    model_name: str,
    gpu_name: str,
) -> dict:
    base = compute_cost_quality_compaction_curve(
        csv_path,
        target_session_id,
        compaction_fraction,
        model_name,
        gpu_name,
    )
    total_cost = float(base.get("total_overall_cost", 0.0))
    avg_f1 = _average_quality(base.get("points", []))
    result = dict(base)
    result["metric"] = "avg_quality_total_cost"
    result["mode"] = "avg_quality_compaction"
    result["points"] = [{
        "turn": len(base.get("points", [])) - 1,
        "overall_cost": total_cost,
        "avg_f1_score": avg_f1,
    }]
    result["avg_f1_score"] = avg_f1
    return result
