"""Generate Paper A CSV/LaTeX summaries from audited result ledgers."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from paper_a.tools.validate_result_ledgers import RESULTS, validate
except ModuleNotFoundError:
    from validate_result_ledgers import RESULTS, validate


GENERATED = Path(__file__).resolve().parents[1] / "manuscript" / "generated"
METRICS = ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR100"]
EXPECTED_CORE_SEEDS = {42, 123, 2024}


def _accepted_main_rows() -> list[dict[str, str]]:
    with (RESULTS / "main_results.csv").open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["status"] == "ACCEPTED"]


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    keys = ("dataset", "detector", "backbone", "method", "placement")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summaries: list[dict[str, str]] = []
    for group_key, group_rows in sorted(groups.items()):
        summary = dict(zip(keys, group_key))
        seeds = sorted({int(row["seed"]) for row in group_rows})
        summary["seeds"] = ";".join(str(seed) for seed in seeds)
        summary["n"] = str(len(group_rows))
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows if row[metric].strip()]
            summary[f"{metric}_mean"] = f"{statistics.mean(values):.6f}" if values else ""
            summary[f"{metric}_std"] = f"{statistics.stdev(values):.6f}" if len(values) > 1 else ""
        summaries.append(summary)
    return summaries


def _format_metric(row: dict[str, str], metric: str) -> str:
    mean = row[f"{metric}_mean"]
    if not mean:
        return "--"
    standard_deviation = row[f"{metric}_std"]
    mean_percent = 100.0 * float(mean)
    if standard_deviation:
        return f"{mean_percent:.2f} $\\pm$ {100.0 * float(standard_deviation):.2f}"
    return f"{mean_percent:.2f}"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def build() -> dict[str, int | str]:
    validate()
    summaries = aggregate(_accepted_main_rows())
    GENERATED.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "detector", "backbone", "method", "placement", "seeds", "n"] + [
        item for metric in METRICS for item in (f"{metric}_mean", f"{metric}_std")
    ]
    csv_path = GENERATED / "main_results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)

    tex_path = GENERATED / "main_results_rows.tex"
    complete_summaries = [
        row
        for row in summaries
        if {int(seed) for seed in row["seeds"].split(";") if seed} == EXPECTED_CORE_SEEDS
    ]
    if not complete_summaries:
        tex_path.write_text(
            r"\newcommand{\generatedmainresultrows}{Pending & -- & -- & 0 & -- & -- & -- & -- \\}" + "\n",
            encoding="ascii",
        )
    else:
        lines = []
        for row in complete_summaries:
            values = [_latex_escape(row["dataset"]), _latex_escape(row["method"]), _latex_escape(row["placement"]), row["n"]]
            values.extend(_format_metric(row, metric) for metric in ("AP", "AP50", "AP75", "APS"))
            lines.append(" & ".join(values) + r" \\")
        tex_path.write_text(
            "\\newcommand{\\generatedmainresultrows}{%\n"
            + "\n".join(lines)
            + "\n}\n",
            encoding="utf-8",
        )
    return {
        "status": "PASS",
        "accepted_rows": sum(int(row["n"]) for row in summaries),
        "summary_rows": len(summaries),
        "headline_groups": len(complete_summaries),
        "incomplete_groups": len(summaries) - len(complete_summaries),
    }


if __name__ == "__main__":
    print(build())
