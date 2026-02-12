import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import cm


# ============================================================
# 1) Tokenization + Vocabulary
# ============================================================

_word_re = re.compile(r"[A-Za-z0-9']+")

def tokenize_words(text: str, lowercase: bool = True) -> List[str]:
    toks = _word_re.findall(text)
    return [t.lower() for t in toks] if lowercase else toks

def build_vocab(all_texts: Iterable[str], lowercase: bool = True) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for t in all_texts:
        for w in tokenize_words(t, lowercase=lowercase):
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {i: w for w, i in vocab.items()}

def encode(text: str, vocab: Dict[str, int], lowercase: bool = True) -> List[int]:
    return [vocab[w] for w in tokenize_words(text, lowercase=lowercase)]

def decode_subtuple(sub: Tuple[int, ...], inv_vocab: Dict[int, str]) -> str:
    return " ".join(inv_vocab.get(t, f"<{t}>") for t in sub)


# ============================================================
# 2) Longest Common Substring (contiguous) on token sequences
# ============================================================

def longest_common_substring_tokens(a: List[int], b: List[int], min_len: int = 1) -> Optional[Tuple[int, int, int]]:
    """
    Returns (start_a, start_b, length) of the SINGLE longest common contiguous
    token subsequence between token lists a and b.
    DP O(len(a)*len(b)).
    """
    if not a or not b:
        return None

    prev = [0] * (len(b) + 1)
    best_len = 0
    best_end_a = 0
    best_end_b = 0

    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len = cur[j]
                    best_end_a = i
                    best_end_b = j
        prev = cur

    if best_len < min_len:
        return None
    return (best_end_a - best_len, best_end_b - best_len, best_len)

def tok_tuple(seq: List[int], start: int, length: int) -> Tuple[int, ...]:
    return tuple(seq[start:start + length])


# ============================================================
# 3) Node model (each step has input/out nodes)
# ============================================================

@dataclass(frozen=True)
class NodeRef:
    step: int
    kind: str  # "in" or "out"

def timeline_nodes(n_steps: int) -> List[NodeRef]:
    nodes: List[NodeRef] = []
    for k in range(n_steps):
        nodes.append(NodeRef(k, "in"))
        nodes.append(NodeRef(k, "out"))
    return nodes


# ============================================================
# 4) Global pairwise matching "over all groups"
#    For every pair of nodes i<j compute the (single) max LCS substring.
#    Then for each node keep TOP-K incident substrings (descending length).
# ============================================================

def compute_global_pairwise_lcs(
    nodes: List[NodeRef],
    node_seqs: Dict[NodeRef, List[int]],
    min_len: int = 1,
) -> List[Tuple[NodeRef, NodeRef, Tuple[int, ...], int]]:
    """
    Returns list of (node_i, node_j, substring_tuple, length)
    for every pair that has a match >= min_len.
    """
    pairs: List[Tuple[NodeRef, NodeRef, Tuple[int, ...], int]] = []
    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            res = longest_common_substring_tokens(node_seqs[a], node_seqs[b], min_len=min_len)
            if res is None:
                continue
            sa, sb, L = res
            sub = tok_tuple(node_seqs[a], sa, L)
            pairs.append((a, b, sub, L))
    return pairs

def per_node_topk_substrings(
    nodes: List[NodeRef],
    pair_matches: List[Tuple[NodeRef, NodeRef, Tuple[int, ...], int]],
    top_k: int = 3,
) -> Dict[NodeRef, List[Tuple[int, ...]]]:
    """
    For each node, keep TOP-K substring tuples among its incident matches,
    sorted by descending length (ties preserved), then stable-deduped.
    """
    incident: Dict[NodeRef, List[Tuple[Tuple[int, ...], int]]] = {n: [] for n in nodes}
    for a, b, sub, L in pair_matches:
        incident[a].append((sub, L))
        incident[b].append((sub, L))

    keep: Dict[NodeRef, List[Tuple[int, ...]]] = {}

    for n in nodes:
        items = incident[n]
        if not items:
            keep[n] = []
            continue

        # sort by length desc, then deterministic by token tuple
        items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))

        # stable unique by substring tuple while keeping order
        seen = set()
        uniq: List[Tuple[Tuple[int, ...], int]] = []
        for sub, L in items_sorted:
            if sub not in seen:
                seen.add(sub)
                uniq.append((sub, L))

        keep[n] = [sub for sub, _ in uniq[:max(0, top_k)]]
    return keep


# ============================================================
# 5) Find substring position in a node sequence
#    (we map substring to its first occurrence to plot position)
# ============================================================

def find_subtuple_first(seq: List[int], sub: Tuple[int, ...]) -> Optional[int]:
    """Return first start index where sub occurs in seq, else None."""
    if not sub or not seq or len(sub) > len(seq):
        return None
    L = len(sub)
    for i in range(len(seq) - L + 1):
        if tuple(seq[i:i+L]) == sub:
            return i
    return None


# ============================================================
# 6) Colors + plotting (no labels)
# ============================================================

def assign_colors(subs: List[Tuple[int, ...]]) -> Dict[Tuple[int, ...], Tuple[float, float, float, float]]:
    uniq = list(dict.fromkeys(subs))  # stable unique
    cmap = cm.get_cmap("tab20", max(1, len(uniq)))
    return {s: cmap(i) for i, s in enumerate(uniq)}

@dataclass
class DrawnSeg:
    sub: Tuple[int, ...]
    center: Tuple[float, float]

def plot_token_bars_no_labels(
    steps: List[Tuple[str, str]],
    per_node_top_k: int = 3,
    min_token_len: int = 1,
    lowercase: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    row_gap: float = 0.24,
    return_matches: bool = False,
):
    # Build vocab
    all_texts = [t for step in steps for t in step]
    vocab = build_vocab(all_texts, lowercase=lowercase)
    inv_vocab = invert_vocab(vocab)

    # Encode sequences
    n = len(steps)
    nodes = timeline_nodes(n)
    node_seqs: Dict[NodeRef, List[int]] = {}
    for k, (inp, outp) in enumerate(steps):
        node_seqs[NodeRef(k, "in")] = encode(inp, vocab, lowercase=lowercase)
        node_seqs[NodeRef(k, "out")] = encode(outp, vocab, lowercase=lowercase)

    # Global pairwise longest matches (single LCS per pair)
    pair_matches = compute_global_pairwise_lcs(nodes, node_seqs, min_len=min_token_len)

    # Keep TOP-K per node (Approach B)
    keep = per_node_topk_substrings(nodes, pair_matches, top_k=per_node_top_k)

    # Colors across all kept substrings
    all_kept: List[Tuple[int, ...]] = []
    for nd in nodes:
        all_kept.extend(keep.get(nd, []))
    color_of = assign_colors(all_kept)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 1)

    # Dynamic y-limits
    top = 0.93
    total_height = top + row_gap * (n - 1) + 0.25
    ax.set_ylim(-total_height, 1)

    # Layout constants
    x_in, w = 0.05, 0.42
    x_out = 0.53
    h_node = 0.18

    def draw_frame(x, y_top):
        ax.add_patch(Rectangle((x, y_top - h_node), w, h_node, fill=False, linewidth=1.2, edgecolor="black"))

    drawn: Dict[NodeRef, List[DrawnSeg]] = {}

    def draw_bars_in_node(node: NodeRef, x: float, y_top: float):
        subs = keep.get(node, [])
        seq = node_seqs[node]
        T = max(1, len(seq))

        if not subs:
            drawn[node] = []
            return

        pad_x = 0.02
        pad_y = 0.02
        inner_x = x + pad_x
        inner_w = w - 2 * pad_x
        inner_y_top = y_top - pad_y

        n_lanes = len(subs)
        lane_gap = 0.012
        lane_h = (h_node - 2 * pad_y - lane_gap * (n_lanes - 1)) / n_lanes

        segs: List[DrawnSeg] = []
        for i, sub in enumerate(subs):
            lane_top = inner_y_top - i * (lane_h + lane_gap)
            lane_y0 = lane_top - lane_h

            # White bar
            ax.add_patch(Rectangle(
                (inner_x, lane_y0),
                inner_w,
                lane_h,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=1,
            ))

            start = find_subtuple_first(seq, sub)
            if start is None:
                continue

            L = len(sub)
            rel_x0 = start / T
            rel_x1 = (start + L) / T

            seg_x0 = inner_x + inner_w * rel_x0
            seg_w = max(0.002, inner_w * (rel_x1 - rel_x0))

            c = color_of.get(sub, (0.7, 0.7, 0.7, 1.0))
            ax.add_patch(Rectangle(
                (seg_x0, lane_y0),
                seg_w,
                lane_h,
                facecolor=c,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            ))

            segs.append(DrawnSeg(sub=sub, center=(seg_x0 + seg_w / 2, lane_y0 + lane_h / 2)))

        drawn[node] = segs

    # Draw rows
    for k in range(n):
        y = top - k * row_gap
        draw_frame(x_in, y)
        draw_frame(x_out, y)
        draw_bars_in_node(NodeRef(k, "in"), x_in, y)
        draw_bars_in_node(NodeRef(k, "out"), x_out, y)

    # Arrows between identical substrings in timeline order
    occ: Dict[Tuple[int, ...], List[Tuple[float, float]]] = {}
    for nd in nodes:
        for seg in drawn.get(nd, []):
            occ.setdefault(seg.sub, []).append(seg.center)

    for sub, centers in occ.items():
        if len(centers) < 2:
            continue
        c = color_of.get(sub, (0, 0, 0, 1))
        for i in range(len(centers) - 1):
            (x0, y0) = centers[i]
            (x1, y1) = centers[i + 1]
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.8,
                color=c,
                alpha=0.9,
                connectionstyle="arc3,rad=-0.15",
                zorder=3,
            ))

    if return_matches:
        # return keep + vocab decode + pairwise match list so printing can find "where else it matched"
        return fig, ax, keep, inv_vocab, pair_matches
    return fig, ax


# ============================================================
# 7) Printing helpers
# ============================================================

def print_topk_per_step_io(
    keep: Dict[NodeRef, List[Tuple[int, ...]]],
    inv_vocab: Dict[int, str],
    k: int = 3,
) -> None:
    max_step = max((nr.step for nr in keep.keys()), default=-1)
    for step in range(max_step + 1):
        for kind in ("in", "out"):
            nr = NodeRef(step, kind)
            subs = keep.get(nr, [])[:k]
            print(f"\nstep {step:02d} {kind}:")
            if not subs:
                print("  (no matches kept)")
                continue
            for idx, sub in enumerate(subs, 1):
                print(f"  {idx}. len={len(sub)} | {decode_subtuple(sub, inv_vocab)}")


def build_match_index(
    pair_matches: List[Tuple[NodeRef, NodeRef, Tuple[int, ...], int]]
) -> Dict[Tuple[int, ...], List[Tuple[NodeRef, NodeRef]]]:
    """
    substring -> list of node-pairs (a,b) for which this substring was the (single)
    longest match. (One per pair.)
    """
    idx: Dict[Tuple[int, ...], List[Tuple[NodeRef, NodeRef]]] = {}
    for a, b, sub, _L in pair_matches:
        idx.setdefault(sub, []).append((a, b))
    return idx


def other_inputs_where_matched(
    sub: Tuple[int, ...],
    src_in: NodeRef,
    sub_to_pairs: Dict[Tuple[int, ...], List[Tuple[NodeRef, NodeRef]]],
) -> List[NodeRef]:
    """
    Return all OTHER input nodes that have a recorded pair_match with src_in
    using exactly 'sub'.
    """
    out: List[NodeRef] = []
    seen = set()

    for (a, b) in sub_to_pairs.get(sub, []):
        if a == src_in:
            other = b
        elif b == src_in:
            other = a
        else:
            continue

        if other.kind != "in":
            continue

        if other not in seen:
            seen.add(other)
            out.append(other)

    out.sort(key=lambda nr: nr.step)
    return out


def print_input_topk_where_else_matched(
    keep: Dict[NodeRef, List[Tuple[int, ...]]],
    pair_matches: List[Tuple[NodeRef, NodeRef, Tuple[int, ...], int]],
    inv_vocab: Dict[int, str],
    k: int = 3,
) -> None:
    """
    For every input node:
      for each of its top-k kept substrings:
        print all (input i, input j, substring) where input j is another input
        that matched input i on that substring (as the pairwise longest match).
    """
    sub_to_pairs = build_match_index(pair_matches)
    max_step = max((nr.step for nr in keep.keys()), default=-1)

    for step in range(max_step + 1):
        src_in = NodeRef(step, "in")
        subs = keep.get(src_in, [])[:k]

        print(f"\n=== INPUT {step:02d}: top-{k} substrings and other INPUT matches ===")
        if not subs:
            print("  (no matches kept)")
            continue

        for idx, sub in enumerate(subs, 1):
            decoded = decode_subtuple(sub, inv_vocab)
            others = other_inputs_where_matched(sub, src_in, sub_to_pairs)

            if not others:
                print(f"  {idx}. (input {step}, <none>, len={len(sub)}) | {decoded}")
                continue

            for other in others:
                # Your requested format (plus decoded text for readability)
                print(f"  {idx}. (input {step}, input {other.step}, len={len(sub)}) | {decoded}")


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    steps = [
        (
            "You are a helpful agent who needs to perform a task. What tasks can you perform?",
            "I can do math. I can do coding."
        ),
        (
            "Previous agent said I can do math and coding. Can you do coding?",
            "I can do coding."
        ),
        (
            "Earlier output stated I can do coding. Can you also do math?",
            "I can do math."
        ),
        (
            "So far we have I can do math. I can do things. I can do coding. Can you repeat your capabilities?",
            "I can do math and coding and other things which you did not ask."
        ),
        (
            "Given I can do math and coding was said before, can you confirm coding again?",
            "I can do coding."
        ),
        (
            "We saw earlier that I can do math. Can you confirm math again? I can do other things. I can do coding.",
            "I can do math and other things."
        ),
        (
            "So far we have I can do coding. I can do things. I can do math. Please focus on coding tasks now.",
            "I can do coding and other things."
        ),
        (
            "Prior outputs said I can do coding. I can do math. Can you restate both?",
            "I can do math and coding."
        ),
        (
            "Given that I can do math and coding has appeared multiple times, can you handle math tasks?",
            "I can do math and analysis."
        ),
    ]

    fig, ax, keep, inv_vocab, pair_matches = plot_token_bars_no_labels(
        steps,
        per_node_top_k=3,
        min_token_len=1,
        lowercase=True,
        figsize=(14, 10),
        row_gap=0.24,
        return_matches=True,
    )

    plt.savefig("stepwise_io_alignment.pdf", dpi=300, bbox_inches="tight")

    # 1) Print top-k kept substrings per step input/output node
    # print_topk_per_step_io(keep, inv_vocab, k=3)

    # 2) For every input, for its top-k substrings, print where else (other inputs) it matched
    print_input_topk_where_else_matched(keep, pair_matches, inv_vocab, k=3)
