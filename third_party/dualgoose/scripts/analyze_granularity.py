"""[Major-Project-Research addition] Analyse the scalar vs channel-wise decay study against PREDICTIONS_granularity.json.

Reads our runs (results/mqar_probe/*_gran_*, results/long_retrieval/*_granE4_*, *_granE5_*), the authors' published runs
(upstream_results/...), and E6 interventions (results/decay_interventions/). Writes results/granularity_report.md and
results/granularity_report.json. Statistics: exact two-sided permutation test on the mean difference, Fisher exact
two-sided test on lock-in counts, mean rebinding accuracy.
"""
import itertools
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
PRED = json.loads((ROOT / "PREDICTIONS_granularity.json").read_text())
LOCK = PRED["thresholds"]


def load(pattern, base="results"):
    rows = {}
    for f in sorted((ROOT / base).glob(pattern)):
        d = json.loads(f.read_text())
        rows[d["seed"]] = d
    return rows


def permutation_p(a, b):
    """Exact two-sided permutation test on the difference of means."""
    pooled, n = a + b, len(a)
    observed = abs(mean(a) - mean(b))
    total = hits = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        chosen = set(idx)
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        hits += abs(mean(x) - mean(y)) >= observed - 1e-12
        total += 1
    return hits / total


def fisher_p(a_hit, a_n, b_hit, b_n):
    """Two-sided Fisher exact test on a 2x2 table (sum of tables no more likely than the observed one)."""
    hits, n = a_hit + b_hit, a_n + b_n
    def prob(x):
        return math.comb(a_n, x) * math.comb(b_n, hits - x) / math.comb(n, hits)
    observed = prob(a_hit)
    lo, hi = max(0, hits - b_n), min(a_n, hits)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def summary(rows, threshold):
    acc = [r["eval_accuracy"] for _, r in sorted(rows.items())]
    rebind = [r["rebind_accuracy"] for r in rows.values() if r.get("rebind_accuracy") is not None]
    return dict(n=len(acc), mean=mean(acc) if acc else None, std=pstdev(acc) if acc else None,
                lock_in=sum(a >= threshold for a in acc), per_seed=[round(a, 3) for a in acc],
                rebind_mean=mean(rebind) if rebind else None)


def compare(channel, scalar, threshold):
    c, s = summary(channel, threshold), summary(scalar, threshold)
    out = dict(channel=c, scalar=s)
    if c["n"] and s["n"]:
        a = [r["eval_accuracy"] for r in channel.values()]
        b = [r["eval_accuracy"] for r in scalar.values()]
        out.update(mean_difference=c["mean"] - s["mean"], permutation_p=permutation_p(a, b),
                   lock_in_fisher_p=fisher_p(c["lock_in"], c["n"], s["lock_in"], s["n"]))
    return out


def main():
    cells = {
        "E1": dict(threshold=LOCK["mqar_lock_in"], ours="mqar_probe/{arm}_k32_d32l1_gran_seed*.json",
                   upstream="mqar_probe/armed_gdn_k32_d32l1_h10_seed*.json"),
        "E4": dict(threshold=LOCK["long_retrieval_lock_in"], ours="long_retrieval/{arm}_L64p4_d32_granE4_seed*.json",
                   upstream="long_retrieval/armed_gdn_L64p4_d32_h5_d4c_seed*.json"),
        "E5": dict(threshold=LOCK["long_retrieval_lock_in"], ours="long_retrieval/{arm}_L64p4_d32_granE5_seed*.json",
                   upstream="long_retrieval/armed_gdn_L64p4_d32_h10ck_seed*.json"),
    }
    report = {}
    for name, cell in cells.items():
        channel = load(cell["ours"].format(arm="armed_gdn"))
        scalar = load(cell["ours"].format(arm="armed_gdn_scalar"))
        upstream = load(cell["upstream"], base="upstream_results")
        result = compare(channel, scalar, cell["threshold"])
        up = summary(upstream, cell["threshold"])
        ours = result["channel"]
        result["reproduction"] = dict(upstream=up, ours_channel=ours)
        if ours["n"] and up["n"]:
            result["reproduction"].update(
                mean_difference=ours["mean"] - up["mean"],
                lock_in_fisher_p=fisher_p(ours["lock_in"], ours["n"], up["lock_in"], up["n"]))
        report[name] = result

    def ok(name):
        r = report[name].get("reproduction", {})
        if "mean_difference" not in r:
            return None
        return abs(r["mean_difference"]) <= .05 if name == "E1" else r["lock_in_fisher_p"] >= .05
    verdicts = {"R0_reproduction": {k: ok(k) for k in cells}}
    e1, e4, e5 = report["E1"], report["E4"], report["E5"]
    verdicts["G1_capacity"] = (dict(mean_difference=e1["mean_difference"], p=e1["permutation_p"],
                                    confirmed=e1["mean_difference"] > 0 and e1["permutation_p"] < LOCK["alpha"])
                               if "mean_difference" in e1 else None)
    verdicts["G2_distractors"] = (dict(lock_in=(e4["channel"]["lock_in"], e4["scalar"]["lock_in"]),
                                       fisher_p=e4["lock_in_fisher_p"],
                                       direction_holds=e4["channel"]["lock_in"] >= e4["scalar"]["lock_in"])
                                  if "lock_in_fisher_p" in e4 else None)
    verdicts["G3_collision"] = (dict(lock_in=(e5["channel"]["lock_in"], e5["scalar"]["lock_in"]),
                                     fisher_p=e5["lock_in_fisher_p"],
                                     direction_holds=e5["channel"]["lock_in"] > e5["scalar"]["lock_in"],
                                     significant=e5["lock_in_fisher_p"] < LOCK["alpha"])
                                if "lock_in_fisher_p" in e5 else None)
    interventions = {}
    for f in sorted((ROOT / "results" / "decay_interventions").glob("*.json")):
        d = json.loads(f.read_text())
        tag = "E1" if "_gran_" in f.name else "E4" if "_granE4_" in f.name else "E5" if "_granE5_" in f.name else "other"
        for k, v in d["accuracy"].items():
            interventions.setdefault(tag, {}).setdefault(k, []).append(v)
    verdicts["G4_mechanism_inputs"] = {t: {k: round(mean(v), 4) for k, v in d.items()} for t, d in interventions.items()}
    out = dict(predictions=PRED["predictions"], verdicts=verdicts, cells=report)
    (ROOT / "results" / "granularity_report.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = ["# Scalar vs channel-wise decay in the Anatomy harness", "",
             "Predictions frozen in PREDICTIONS_granularity.json before training.", "", "## Verdicts", ""]
    lines += [f"- **{k}**: `{json.dumps(v)}`" for k, v in verdicts.items()]
    lines += ["", "## Cells", "", "| Cell | Arm | n | mean | lock-in | rebind | per-seed |", "|---|---|---:|---:|---:|---:|---|"]
    for name, r in report.items():
        for arm, s in (("channel (armed_gdn)", r["channel"]), ("scalar (armed_gdn_scalar)", r["scalar"]),
                       ("published armed_gdn", r["reproduction"]["upstream"])):
            if s["n"]:
                rb = f"{s['rebind_mean']:.3f}" if s["rebind_mean"] is not None else "-"
                lines.append(f"| {name} | {arm} | {s['n']} | {s['mean']:.3f} | {s['lock_in']}/{s['n']} | {rb} | {s['per_seed']} |")
    (ROOT / "results" / "granularity_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
