import re

def _bigrams(text):
    toks = re.findall(r"\w+", text.lower())
    return set(zip(toks, toks[1:])) if len(toks) > 1 else set()

def support_score(answer: str, contexts: list[str]) -> float:
    """Naive baseline: bigram coverage of answer by concatenated contexts."""
    a = _bigrams(answer)
    if not a:
        return 0.0
    ctx = " ".join(contexts)
    c = _bigrams(ctx)
    return len(a & c) / max(1, len(a))  # 0.0–1.0
