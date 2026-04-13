import csv
import ast


# GPT-5.4 standard pricing per 1M input tokens
UNCACHED_INPUT_COST_PER_M = 2.50
CACHED_INPUT_COST_PER_M = 0.25


def token_cost(num_tokens: int, price_per_million: float) -> float:
    return (num_tokens / 1_000_000) * price_per_million


def parse_turn_indices(turn_indices_str: str) -> tuple[int, int]:
    # Example: "(3, 4)"
    a, b = ast.literal_eval(turn_indices_str)
    return int(a), int(b)


def load_session_rows(csv_path: str, target_session_id: str) -> list[dict]:
    rows = []

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


def compute_gpt54_prefix_cache_cost(
    csv_path: str,
    target_session_id: str,
    uncached_price_per_m: float = UNCACHED_INPUT_COST_PER_M,
    cached_price_per_m: float = CACHED_INPUT_COST_PER_M,
) -> dict:
    rows = load_session_rows(csv_path, target_session_id)

    if not rows:
        raise ValueError(f"No rows found for session_id={target_session_id!r}")

    first_prefix = rows[0]["prefix_length"]

    round0_uncached_cost = token_cost(first_prefix, uncached_price_per_m)
    total_cost = round0_uncached_cost

    breakdown = [
        {
            "kind": "round0_initial_prefix",
            "tokens": first_prefix,
            "price_per_million": uncached_price_per_m,
            "cost": round0_uncached_cost,
        }
    ]

    for row in rows:
        prefix_len = row["prefix_length"]
        extra_len = row["next_turn_added_length"]

        cached_cost = token_cost(prefix_len, cached_price_per_m)
        uncached_cost = token_cost(extra_len, uncached_price_per_m)

        total_cost += cached_cost + uncached_cost

        breakdown.append(
            {
                "kind": "turn",
                "turn_indices": row["turn_indices"],
                "prefix_length": prefix_len,
                "next_turn_added_length": extra_len,
                "cached_cost": cached_cost,
                "uncached_cost": uncached_cost,
                "turn_total_cost": cached_cost + uncached_cost,
            }
        )

    return {
        "session_id": target_session_id,
        "num_rows": len(rows),
        "round0_prefix_length": first_prefix,
        "total_input_cost": total_cost,
        "breakdown": breakdown,
    }


def print_cost_report(result: dict) -> None:
    print(f"Session: {result['session_id']}")
    print(f"Rows in session: {result['num_rows']}")
    print()

    first = result["breakdown"][0]
    print(
        f"Round 0: uncached input {first['tokens']} tokens "
        f"-> ${first['cost']:.6f}"
    )

    print()
    print("Per turn:")
    for item in result["breakdown"][1:]:
        print(
            f"  {item['turn_indices']}: "
            f"cached {item['prefix_length']} -> ${item['cached_cost']:.6f}, "
            f"uncached {item['next_turn_added_length']} -> ${item['uncached_cost']:.6f}, "
            f"turn total -> ${item['turn_total_cost']:.6f}"
        )

    print()
    print(f"Total input cost: ${result['total_input_cost']:.6f}")


if __name__ == "__main__":
    csv_path = "lcp_analysis.csv"
    session_id = "swebench__astropy__astropy-12907__minimax"

    result = compute_gpt54_prefix_cache_cost(csv_path, session_id)
    print_cost_report(result)