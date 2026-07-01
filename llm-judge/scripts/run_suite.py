"""Run a suite -> structured report. python -m scripts.run_suite suites/k8s_suite.json"""
import json
import statistics as st
import sys
from judge.core import judge_pointwise
from judge.rubric import RUBRIC


def run_suite(path):
    suite = json.load(open(path))
    sys_prompt = suite.get("system_prompt", "(none)")
    results, parse_failures = [], 0

    for case in suite["cases"]:
        case = dict(case)
        case.setdefault("system_prompt", sys_prompt)
        verdict = judge_pointwise(case)
        if not verdict["ok"]:
            parse_failures += 1
        results.append({"id": case["id"], "label": case.get("label"),
                        "verdict": verdict})

    scored = [r for r in results if r["verdict"]["ok"]]
    overalls = [r["verdict"]["overall"] for r in scored]
    pass_rate = sum(1 for o in overalls if o >= 4.0) / len(overalls) if overalls else 0

    per_criterion = {}
    for crit in RUBRIC:
        vals = [r["verdict"]["scores"].get(crit) for r in scored
                if crit in r["verdict"]["scores"]]
        if vals:
            per_criterion[crit] = round(st.mean(vals), 2)

    report = {
        "suite": suite.get("name", path),
        "n_cases": len(suite["cases"]),
        "parse_failures": parse_failures,
        "pass_rate@4": round(pass_rate, 3),
        "mean_overall": round(st.mean(overalls), 3) if overalls else None,
        "score_spread": round(max(overalls) - min(overalls), 3) if len(overalls) > 1 else 0,
        "mean_per_criterion": per_criterion,
        "cases": [{"id": r["id"], "label": r["label"],
                   "overall": r["verdict"].get("overall"),
                   "scores": r["verdict"].get("scores")} for r in results],
    }
    return report


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "suites/k8s_suite.json"
    rep = run_suite(path)
    print(json.dumps(rep, indent=2))
    json.dump(rep, open("reports/suite_report.json", "w"), indent=2)
