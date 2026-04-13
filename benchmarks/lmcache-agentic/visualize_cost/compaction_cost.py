from typing import Dict
from prefix_cache_cost import load_session_rows, token_cost, UNCACHED_INPUT_COST_PER_M


def compute_compaction_curve(
    csv_path: str,
    target_session_id: str,
    compaction_fraction: float,
    uncached_price_per_m: float = UNCACHED_INPUT_COST_PER_M,
) -> Dict:
    if not (0.0 <= compaction_fraction <= 1.0):
        raise ValueError("compaction_fraction must be between 0.0 and 1.0")

    rows = load_session_rows(csv_path, target_session_id)
    if not rows:
        raise ValueError(f"No rows found for session_id={target_session_id!r}")

    first_prefix = rows[0]["prefix_length"]
    cumulative_cost = token_cost(first_prefix, uncached_price_per_m)

    points = [
        {
            "turn": 0,
            "label": "round0",
            "incremental_cost": cumulative_cost,
            "cumulative_cost": cumulative_cost,
        }
    ]

    for idx, row in enumerate(rows, start=1):
        prefix_len = row["prefix_length"]
        extra_len = row["next_turn_added_length"]
        compacted_prefix_len = prefix_len * compaction_fraction
        prefix_cost = token_cost(compacted_prefix_len, uncached_price_per_m)
        extra_cost = token_cost(extra_len, uncached_price_per_m)
        turn_cost = prefix_cost + extra_cost
        cumulative_cost += turn_cost
        points.append(
            {
                "turn": idx,
                "label": row["turn_indices"],
                "compaction_fraction": compaction_fraction,
                "compacted_prefix_tokens": compacted_prefix_len,
                "new_tokens": extra_len,
                "incremental_cost": turn_cost,
                "cumulative_cost": cumulative_cost,
            }
        )

    return {
        "session_id": target_session_id,
        "mode": "compaction_no_cache",
        "compaction_fraction": compaction_fraction,
        "points": points,
        "total_input_cost": cumulative_cost,
    }
