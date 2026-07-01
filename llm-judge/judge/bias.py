"""Bias handling — named, mitigated in code, and measured.

position bias: judge each pair in BOTH orders (A,B) and (B,A). If the winner is
   consistent after accounting for the swap, the pair is order-stable. The flip
   rate = fraction of pairs whose verdict changed with order. Mitigation: require
   agreement; treat disagreement as a tie.
verbosity bias: re-judge a correct answer that has been padded with filler. If the
   padded version wins, the judge is length-biased.
"""
from .core import judge_pairwise, judge_pointwise


def position_bias_pairwise(question, answer_a, answer_b, system_prompt="(none)"):
    """Run both orders. Returns consensus winner + whether it flipped."""
    fwd = judge_pairwise(question, answer_a, answer_b, system_prompt)  # A=a, B=b
    rev = judge_pairwise(question, answer_b, answer_a, system_prompt)  # A=b, B=a

    # translate reverse verdict back to original labels
    rev_winner_orig = {"A": "b", "B": "a", "tie": "tie"}[rev["winner"]]
    fwd_winner_orig = {"A": "a", "B": "b", "tie": "tie"}[fwd["winner"]]

    flipped = fwd_winner_orig != rev_winner_orig
    # mitigation: require agreement, else tie
    consensus = fwd_winner_orig if not flipped else "tie"
    return {"forward": fwd_winner_orig, "reverse": rev_winner_orig,
            "flipped": flipped, "consensus": consensus,
            "tokens": fwd["tokens"] + rev["tokens"]}


PAD = (" To elaborate further, it is worth noting that this topic has many "
       "nuances and considerations that thoughtful practitioners often discuss "
       "at length in great and exhaustive detail.")


def verbosity_probe(case):
    """Score the answer, then score a padded (longer but no new info) version.
    A length-unbiased judge should not raise the score for padding."""
    base = judge_pointwise(case)
    padded = dict(case)
    padded["model_output"] = case["model_output"] + PAD * 3
    padded_v = judge_pointwise(padded)
    delta = None
    if base["ok"] and padded_v["ok"]:
        delta = round(padded_v["overall"] - base["overall"], 3)
    return {"base_overall": base.get("overall"),
            "padded_overall": padded_v.get("overall"),
            "score_inflation_from_padding": delta}
