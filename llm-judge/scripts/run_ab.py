"""A/B comparison of two configs using position-bias-corrected pairwise judging.
   python -m scripts.run_ab suites/ab_pairs.json
Each pair: {input, a, b}. Reports win rate for A and B, flip rate, and a winner."""
import json
import sys
from judge.bias import position_bias_pairwise


def run_ab(path):
    data = json.load(open(path))
    name_a, name_b = data.get("config_a", "A"), data.get("config_b", "B")
    a_wins = b_wins = ties = flips = 0
    rows = []

    for pair in data["pairs"]:
        r = position_bias_pairwise(pair["input"], pair["a"], pair["b"],
                                   data.get("system_prompt", "(none)"))
        if r["flipped"]:
            flips += 1
        if r["consensus"] == "a":
            a_wins += 1
        elif r["consensus"] == "b":
            b_wins += 1
        else:
            ties += 1
        rows.append({"input": pair["input"][:50], **r})

    n = len(data["pairs"])
    win_rate_a = round(a_wins / n, 3)
    win_rate_b = round(b_wins / n, 3)
    if a_wins > b_wins:
        winner = name_a
    elif b_wins > a_wins:
        winner = name_b
    else:
        winner = "tie"

    report = {
        "config_a": name_a, "config_b": name_b, "n_pairs": n,
        "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
        "win_rate_a": win_rate_a, "win_rate_b": win_rate_b,
        "position_flip_rate": round(flips / n, 3),
        "winner": winner, "rows": rows,
    }
    return report


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "suites/ab_pairs.json"
    rep = run_ab(path)
    print(json.dumps(rep, indent=2))
    json.dump(rep, open("reports/ab_report.json", "w"), indent=2)
