import csv
import ast
from typing import Dict, List

SHORT_UNCACHED_INPUT_COST_PER_M = 2.50
SHORT_CACHED_INPUT_COST_PER_M = 0.25
LONG_UNCACHED_INPUT_COST_PER_M = 5.00
LONG_CACHED_INPUT_COST_PER_M = 0.50
LONG_CONTEXT_THRESHOLD_TOKENS = 272_000


def token_cost(num_tokens: float, price_per_million: float) -> float:
    return (num_tokens / 1_000_000) * price_per_million


def parse_turn_indices(turn_indices_str: str) -> tuple[int, int]:
    a, b = ast.literal_eval(turn_indices_str)
    return int(a), int(b)


def load_session_rows(csv_path: str, target_session_id: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["session_id"] != target_session_id:
                continue
            start_turn, end_turn = parse_turn_indices(row["turn_indices"])
            rows.append(
                {
                    "session_id": row["session_id"],
                    "turn_indices": row["turn_indices"],
                    "start_turn": start_turn,
                    "end_turn": end_turn,
                    "prefix_length": int(row["prefix_length"]),
                    "next_turn_added_length": int(row["next_turn_added_length"]),
                }
            )
    rows.sort(key=lambda r: r["start_turn"])
    return rows


def gpt54_input_prices(total_input_tokens: float) -> Dict[str, float | str]:
    if total_input_tokens > LONG_CONTEXT_THRESHOLD_TOKENS:
        return {
            "pricing_tier": "long",
            "uncached_price_per_m": LONG_UNCACHED_INPUT_COST_PER_M,
            "cached_price_per_m": LONG_CACHED_INPUT_COST_PER_M,
        }
    return {
        "pricing_tier": "short",
        "uncached_price_per_m": SHORT_UNCACHED_INPUT_COST_PER_M,
        "cached_price_per_m": SHORT_CACHED_INPUT_COST_PER_M,
    }


def compute_prefix_cache_curve(csv_path: str, target_session_id: str) -> Dict:
    rows = load_session_rows(csv_path, target_session_id)
    if not rows:
        raise ValueError(f"No rows found for session_id={target_session_id!r}")

    first_prefix = rows[0]["prefix_length"]
    round0_prices = gpt54_input_prices(first_prefix)
    cumulative_cost = token_cost(first_prefix, round0_prices["uncached_price_per_m"])

    points = [
        {
            "turn": 0,
            "label": "round0",
            "total_input_tokens": first_prefix,
            "pricing_tier": round0_prices["pricing_tier"],
            "uncached_price_per_m": round0_prices["uncached_price_per_m"],
            "cached_price_per_m": round0_prices["cached_price_per_m"],
            "incremental_cost": cumulative_cost,
            "cumulative_cost": cumulative_cost,
        }
    ]

    for idx, row in enumerate(rows, start=1):
        prefix_len = row["prefix_length"]
        extra_len = row["next_turn_added_length"]
        total_input_tokens = prefix_len + extra_len
        prices = gpt54_input_prices(total_input_tokens)
        cached_cost = token_cost(prefix_len, prices["cached_price_per_m"])
        uncached_cost = token_cost(extra_len, prices["uncached_price_per_m"])
        turn_cost = cached_cost + uncached_cost
        cumulative_cost += turn_cost
        points.append(
            {
                "turn": idx,
                "label": row["turn_indices"],
                "cached_prefix_tokens": prefix_len,
                "new_tokens": extra_len,
                "total_input_tokens": total_input_tokens,
                "pricing_tier": prices["pricing_tier"],
                "uncached_price_per_m": prices["uncached_price_per_m"],
                "cached_price_per_m": prices["cached_price_per_m"],
                "incremental_cost": turn_cost,
                "cumulative_cost": cumulative_cost,
            }
        )

    return {
        "session_id": target_session_id,
        "mode": "prefix_cache",
        "pricing_model": "gpt-5.4",
        "long_context_threshold_tokens": LONG_CONTEXT_THRESHOLD_TOKENS,
        "points": points,
        "total_input_cost": cumulative_cost,
    }
