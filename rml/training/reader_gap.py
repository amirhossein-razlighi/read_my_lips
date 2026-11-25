"""Training-label fallback of train_vis2gemma.py (gap_of):
DEPLOYABLE gap extraction from a reader's
full-window n-best (no ground truth): each hypothesis is Levenshtein-aligned
to the whisper context (left + right context tokens, the ___ removed); the
hypothesis tokens aligned inside the ___ position are that hypothesis' gap
filling. Used only for rows without --keep-meta metadata.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_gap_score import align  # noqa: E402
from label_scoring import norm_tokens  # noqa: E402

def gap_of(ctx: str, hyp: str, lg: str) -> str:
    left, _, right = ctx.partition("___")
    L, R = norm_tokens(left, lg), norm_tokens(right, lg)
    ref = L + R
    h = norm_tokens(hyp, lg)
    ops = align(ref, h)
    a = len(L)
    idx = [t for t, (op, ri, _) in enumerate(ops) if ri is not None]
    lt = max((t for t in idx if ops[t][1] < a), default=-1)
    rt = min((t for t in idx if ops[t][1] >= a), default=len(ops))
    # tokens between the last left-context op and the first right-context op
    # that are insertions = the gap; boundary substitutions stay context
    return " ".join(h[j] for op, _, j in ops[lt + 1:rt] if op == "I")
