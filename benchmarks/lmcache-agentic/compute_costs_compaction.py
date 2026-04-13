import csv
import ast


# Change these if needed
UNCACHED_INPUT_COST_PER_M = 2.50  # dollars per 1M input tokens


def token_cost(num_tokens: float, price_per_million: float) -> float:
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


def compute_session_compaction_cost(
    csv_path: str,
    target_session_id: str,
    compaction_fraction: float,
    uncached_price_per_m: float = UNCACHED_INPUT_COST_PER_M,
) -> dict:
    """
    compaction_fraction:
        0.25 means keep 25% of prefix_length each turn
        1.00 means keep 100% of prefix_length
        0.00 means keep none of the prefix
    """
    if not (0.0 <= compaction_fraction <= 1.0):
        raise ValueError("compaction_fraction must be between 0.0 and 1.0")

    rows = load_session_rows(csv_path, target_session_id)
    if not rows:
        raise ValueError(f"No rows found for session_id={target_session_id!r}")

    first_prefix = rows[0]["prefix_length"]

    # Round 0: first prefix as ordinary uncached input
    round0_cost = token_cost(first_prefix, uncached_price_per_m)
    total_cost = round0_cost

    breakdown = [
        {
            "kind": "round0_initial_prefix",
            "tokens": first_prefix,
            "cost": round0_cost,
        }
    ]

    # Later rounds: uncached compacted prefix + uncached extra length
    for row in rows:
        prefix_len = row["prefix_length"]
        extra_len = row["next_turn_added_length"]

        compacted_prefix_len = prefix_len * compaction_fraction

        compacted_prefix_cost = token_cost(compacted_prefix_len, uncached_price_per_m)
        extra_cost = token_cost(extra_len, uncached_price_per_m)
        turn_total_cost = compacted_prefix_cost + extra_cost

        total_cost += turn_total_cost

        breakdown.append(
            {
                "kind": "turn",
                "turn_indices": row["turn_indices"],
                "prefix_length": prefix_len,
                "compacted_prefix_length": compacted_prefix_len,
                "next_turn_added_length": extra_len,
                "compacted_prefix_cost": compacted_prefix_cost,
                "extra_cost": extra_cost,
                "turn_total_cost": turn_total_cost,
            }
        )

    return {
        "session_id": target_session_id,
        "compaction_fraction": compaction_fraction,
        "num_rows": len(rows),
        "round0_prefix_length": first_prefix,
        "total_input_cost": total_cost,
        "breakdown": breakdown,
    }


def print_cost_report(result: dict) -> None:
    print(f"Session: {result['session_id']}")
    print(f"Compaction fraction: {result['compaction_fraction']:.2%}")
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
            f"compacted prefix {item['compacted_prefix_length']:.2f} -> ${item['compacted_prefix_cost']:.6f}, "
            f"extra {item['next_turn_added_length']} -> ${item['extra_cost']:.6f}, "
            f"turn total -> ${item['turn_total_cost']:.6f}"
        )

    print()
    print(f"Total input cost: ${result['total_input_cost']:.6f}")


if __name__ == "__main__":
    csv_path = "lcp_analysis.csv"
    session_id = "swebench__astropy__astropy-12907__minimax"
    compaction_fraction = 0.10  # keep 25% of each prefix

    result = compute_session_compaction_cost(
        csv_path=csv_path,
        target_session_id=session_id,
        compaction_fraction=compaction_fraction,
    )
    print_cost_report(result)