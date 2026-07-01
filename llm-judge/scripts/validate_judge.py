"""Validate the judge itself.
   python -m scripts.validate_judge

Three checks:
  1. Adversarial probes — verbose-but-wrong & confidently-wrong should score LOW
     on correctness/faithfulness; terse-but-correct should score HIGH. Reports
     whether the judge was fooled.
  2. Test-retest — re-judge each case N times at temp=0; flip rate of the
     pass/fail decision.
  3. Gold agreement — each suite case has a human "label"; we map label->expected
     pass/fail and report agreement rate + Cohen's kappa.
"""
import json
import statistics as st

from judge.core import judge_pointwise
from judge.bias import verbosity_probe

SUITE = "suites/k8s_suite.json"
RETEST_N = 3

# human gold: does this answer DESERVE to pass (overall >= 4)?
GOLD_PASS = {
    "good": True, "terse_but_correct": True,
    "verbose_but_wrong": False, "confidently_wrong": False,
}


def cohens_kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    return round((po - pe) / (1 - pe), 3) if pe != 1 else 1.0


def main():
    suite = json.load(open(SUITE))
    sysp = suite.get("system_prompt", "(none)")

    judge_pass, gold_pass, fooled = [], [], []
    adversarial = []

    for case in suite["cases"]:
        c = dict(case)
        c.setdefault("system_prompt", sysp)
        v = judge_pointwise(c)
        if not v["ok"]:
            continue
        passed = v["overall"] >= 4.0
        label = case.get("label", "good")
        gp = GOLD_PASS.get(label, True)
        judge_pass.append(int(passed))
        gold_pass.append(int(gp))

        if label in ("verbose_but_wrong", "confidently_wrong"):
            was_fooled = passed  # it should have failed
            fooled.append(int(was_fooled))
            adversarial.append({"id": case["id"], "label": label,
                                "overall": v["overall"],
                                "correctness": v["scores"].get("correctness"),
                                "faithfulness": v["scores"].get("faithfulness"),
                                "fooled": was_fooled})
        elif label == "terse_but_correct":
            adversarial.append({"id": case["id"], "label": label,
                                "overall": v["overall"],
                                "penalized_for_brevity": v["overall"] < 4.0})

    # test-retest on first 'good' + first adversarial case
    retest = {}
    for case in suite["cases"][:3]:
        c = dict(case)
        c.setdefault("system_prompt", sysp)
        decisions = []
        for _ in range(RETEST_N):
            v = judge_pointwise(c)
            if v["ok"]:
                decisions.append(v["overall"] >= 4.0)
        flips = sum(1 for i in range(1, len(decisions))
                    if decisions[i] != decisions[i - 1])
        retest[case["id"]] = {"decisions": decisions,
                              "flip_rate": round(flips / max(1, len(decisions) - 1), 3)}

    # verbosity probe on a known-good case
    good = next(c for c in suite["cases"] if c.get("label") == "good")
    good = dict(good); good.setdefault("system_prompt", sysp)
    vb = verbosity_probe(good)

    agreement = round(sum(1 for x, y in zip(judge_pass, gold_pass)
                          if x == y) / len(judge_pass), 3)
    report = {
        "agreement_with_gold": agreement,
        "cohens_kappa": cohens_kappa(judge_pass, gold_pass),
        "adversarial_fooled_rate": round(st.mean(fooled), 3) if fooled else None,
        "adversarial_detail": adversarial,
        "test_retest": retest,
        "verbosity_probe": vb,
        "n_cases": len(judge_pass),
    }
    print(json.dumps(report, indent=2))
    json.dump(report, open("reports/judge_validation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
