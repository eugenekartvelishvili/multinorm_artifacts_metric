from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_summary_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    summary = data.get("summary", {})
    det = summary.get("deterministic_metrics_avg", {}) or {}
    judge = summary.get("llm_judge_avg", {}) or {}
    timing = summary.get("timing", {}) or {}

    rows = [
        {
            "group": "Deterministic",
            "metric": "Answer nonempty",
            "value": fmt_num(det.get("answer_nonempty")),
            "comment": "Доля непустых ответов",
        },
        {
            "group": "Deterministic",
            "metric": "Doc match",
            "value": fmt_num(det.get("doc_match")),
            "comment": "Попадание в ожидаемый документ",
        },
        {
            "group": "Deterministic",
            "metric": "Section match",
            "value": fmt_num(det.get("section_match")),
            "comment": "Попадание в ожидаемый раздел",
        },
        {
            "group": "Deterministic",
            "metric": "Key point coverage",
            "value": fmt_num(det.get("key_point_coverage")),
            "comment": "Покрытие ключевых пунктов ответа",
        },
        {
            "group": "Deterministic",
            "metric": "Source count",
            "value": fmt_num(det.get("source_count")),
            "comment": "Достаточность числа источников",
        },
        {
            "group": "Deterministic",
            "metric": "Timing score",
            "value": fmt_num(det.get("timing_score")),
            "comment": "Нормированная скорость ответа",
        },
        {
            "group": "Deterministic",
            "metric": "Main score",
            "value": fmt_num(det.get("main_score")),
            "comment": "Основная агрегированная метрика",
        },
        {
            "group": "LLM judge",
            "metric": "Label score",
            "value": fmt_num(judge.get("label_score")),
            "comment": "Средняя качественная оценка судьи",
        },
        {
            "group": "LLM judge",
            "metric": "Relevance",
            "value": fmt_num(judge.get("relevance")),
            "comment": "Насколько ответ релевантен вопросу",
        },
        {
            "group": "LLM judge",
            "metric": "Completeness",
            "value": fmt_num(judge.get("completeness")),
            "comment": "Насколько ответ полный",
        },
        {
            "group": "LLM judge",
            "metric": "Groundedness",
            "value": fmt_num(judge.get("groundedness")),
            "comment": "Насколько ответ опирается на контекст",
        },
        {
            "group": "LLM judge",
            "metric": "Doc alignment",
            "value": fmt_num(judge.get("doc_alignment")),
            "comment": "Соответствие ответа найденному документу",
        },
        {
            "group": "LLM judge",
            "metric": "Overall",
            "value": fmt_num(judge.get("overall")),
            "comment": "Общая качественная оценка судьи",
        },
        {
            "group": "Timing",
            "metric": "Avg total time",
            "value": fmt_num(timing.get("avg_total_time")),
            "comment": "Среднее время ответа, сек",
        },
        {
            "group": "Meta",
            "metric": "Count",
            "value": str(summary.get("count", "")),
            "comment": "Число оценённых вопросов",
        },
    ]
    return rows


def make_by_type_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    summary = data.get("summary", {})
    by_type = summary.get("by_question_type", {}) or {}

    rows: list[dict[str, Any]] = []
    for question_type, metrics in by_type.items():
        rows.append(
            {
                "question_type": question_type,
                "count": metrics.get("count", ""),
                "main_score": fmt_num(metrics.get("main_score")),
                "doc_match": fmt_num(metrics.get("doc_match")),
                "section_match": fmt_num(metrics.get("section_match")),
                "key_point_coverage": fmt_num(metrics.get("key_point_coverage")),
                "judge_overall": fmt_num(metrics.get("judge_overall")),
                "judge_label_score": fmt_num(metrics.get("judge_label_score")),
            }
        )

    # сортировка: по count desc, потом по имени
    rows.sort(key=lambda x: (-int(x["count"]) if str(x["count"]).isdigit() else 0, x["question_type"]))
    return rows


def make_results_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    results = data.get("results", []) or []

    rows: list[dict[str, Any]] = []
    for item in results:
        parsed = item.get("parsed", {}) or {}
        det = item.get("deterministic_metrics", {}) or {}
        judge = item.get("llm_judge", {}) or {}
        flags = item.get("suspicious_flags", {}) or {}

        rows.append(
            {
                "id": item.get("id", ""),
                "question_type": item.get("question_type", ""),
                "question": item.get("question", ""),
                "expected_doc": item.get("expected_doc", ""),
                "sources_count": len(parsed.get("sources", []) or []),
                "timing_total": fmt_num(parsed.get("timing_total")),
                "doc_match": fmt_num(det.get("doc_match")),
                "section_match": fmt_num(det.get("section_match")),
                "key_point_coverage": fmt_num(det.get("key_point_coverage")),
                "main_score": fmt_num(det.get("main_score")),
                "judge_label": judge.get("label", ""),
                "judge_overall": fmt_num(judge.get("overall")),
                "judge_comment": judge.get("comment", ""),
                "judge_too_harsh": flags.get("judge_too_harsh", ""),
                "judge_doc_conflict": flags.get("judge_doc_conflict", ""),
                "judge_answer_conflict": flags.get("judge_answer_conflict", ""),
            }
        )

    rows.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else 10**9)
    return rows


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    if not rows:
        return "_Нет данных_"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        vals = [v.replace("\n", " ").replace("|", "/") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def build_presentation_md(
    summary_rows: list[dict[str, Any]],
    by_type_rows: list[dict[str, Any]],
    results_rows: list[dict[str, Any]],
) -> str:
    summary_headers = ["group", "metric", "value", "comment"]
    by_type_headers = [
        "question_type",
        "count",
        "main_score",
        "doc_match",
        "section_match",
        "key_point_coverage",
        "judge_overall",
        "judge_label_score",
    ]
    # для презентации лучше укороченная таблица по вопросам
    results_headers = [
        "id",
        "question_type",
        "main_score",
        "judge_label",
        "judge_overall",
        "timing_total",
        "question",
    ]

    parts = [
        "# Итоговые таблицы по оценке RAG-системы",
        "",
        "## 1. Сводная таблица метрик",
        markdown_table(summary_rows, summary_headers),
        "",
        "## 2. Метрики по типам вопросов",
        markdown_table(by_type_rows, by_type_headers),
        "",
        "## 3. Результаты по отдельным вопросам",
        markdown_table(results_rows, results_headers),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build presentation-ready tables from evaluation_results.json")
    parser.add_argument(
        "--input",
        default="evaluation_results.json",
        help="Путь к evaluation_results.json",
    )
    parser.add_argument(
        "--outdir",
        default="metrics_tables",
        help="Папка для сохранения таблиц",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)

    summary_rows = make_summary_table(data)
    by_type_rows = make_by_type_table(data)
    results_rows = make_results_table(data)

    write_csv(
        outdir / "summary_table.csv",
        summary_rows,
        ["group", "metric", "value", "comment"],
    )

    write_csv(
        outdir / "by_type_table.csv",
        by_type_rows,
        [
            "question_type",
            "count",
            "main_score",
            "doc_match",
            "section_match",
            "key_point_coverage",
            "judge_overall",
            "judge_label_score",
        ],
    )

    write_csv(
        outdir / "results_table.csv",
        results_rows,
        [
            "id",
            "question_type",
            "question",
            "expected_doc",
            "sources_count",
            "timing_total",
            "doc_match",
            "section_match",
            "key_point_coverage",
            "main_score",
            "judge_label",
            "judge_overall",
            "judge_comment",
            "judge_too_harsh",
            "judge_doc_conflict",
            "judge_answer_conflict",
        ],
    )

    md_text = build_presentation_md(summary_rows, by_type_rows, results_rows)
    (outdir / "presentation_tables.md").write_text(md_text, encoding="utf-8")

    print(f"Готово. Таблицы сохранены в: {outdir.resolve()}")
    print("Созданы файлы:")
    print(" - summary_table.csv")
    print(" - by_type_table.csv")
    print(" - results_table.csv")
    print(" - presentation_tables.md")


if __name__ == "__main__":
    main()