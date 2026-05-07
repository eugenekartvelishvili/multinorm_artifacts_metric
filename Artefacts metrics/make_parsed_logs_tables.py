from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    if not rows:
        return "_Нет данных_"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        vals = [fmt(row.get(h, "")) for h in headers]
        vals = [v.replace("\n", " ").replace("|", "/") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def flatten_sessions(parsed_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for idx, s in enumerate(parsed_logs, start=1):
        classification = s.get("classification", {}) or {}
        search = s.get("search", {}) or {}
        context = s.get("context", {}) or {}
        answer = s.get("answer", {}) or {}
        timing = s.get("timing", {}) or {}
        bitrix = s.get("bitrix", {}) or {}
        consistency = s.get("consistency", {}) or {}

        row = {
            "session_index": idx,
            "question": s.get("question", ""),
            "bitrix_message": bitrix.get("message", ""),
            "question_type": classification.get("question_type", ""),
            "answer_mode": classification.get("answer_mode", ""),
            "decompose": classification.get("decompose", ""),
            "confidence": classification.get("confidence", ""),
            "rewrite_entity": search.get("entity", ""),
            "rewrite_attribute": search.get("attribute", ""),
            "search_strategy": search.get("strategy", ""),
            "search_text": search.get("search_text", ""),
            "search_mode": search.get("mode", ""),
            "search_level": json.dumps(search.get("level", []), ensure_ascii=False),
            "search_limit": search.get("limit", ""),
            "pass1_hit_count": search.get("pass1_hit_count", ""),
            "pass2_section_hit_count": search.get("pass2_section_hit_count", ""),
            "selected_doc_id": (search.get("selected_doc", {}) or {}).get("doc_id", ""),
            "selected_doc_title": (search.get("selected_doc", {}) or {}).get("doc_title", ""),
            "anchor_number": (search.get("anchor", {}) or {}).get("number", ""),
            "anchor_main_section_id": (search.get("anchor", {}) or {}).get("main_section_id", ""),
            "top_section_number": (search.get("top_section", {}) or {}).get("number", ""),
            "context_chars": context.get("chars", ""),
            "context_blocks": context.get("blocks", ""),
            "final_context_chars": context.get("final_context_chars", ""),
            "sources_count": len(context.get("final_sources", []) or context.get("sources", []) or []),
            "answer_template": answer.get("template", ""),
            "answer_context_chars": answer.get("context_chars", ""),
            "raw_response_chars": answer.get("raw_response_chars", ""),
            "final_response_chars": answer.get("final_response_chars", ""),
            "time_classify": timing.get("classify", ""),
            "time_rewrite": timing.get("rewrite", ""),
            "time_search_fetch": timing.get("search+fetch", ""),
            "time_answer": timing.get("answer", ""),
            "time_total": timing.get("total", ""),
            "http_calls_count": len(s.get("http_calls", []) or []),
            "question_mismatch": consistency.get("question_mismatch", ""),
        }
        rows.append(row)

    return rows


def flatten_hits(parsed_logs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = []

    for idx, s in enumerate(parsed_logs, start=1):
        hits = ((s.get("search", {}) or {}).get(key, [])) or []
        question = s.get("question", "")

        for hit in hits:
            rows.append({
                "session_index": idx,
                "question": question,
                "rank": hit.get("rank", ""),
                "score": hit.get("score", ""),
                "level": hit.get("level", ""),
                "number": hit.get("number", ""),
                "doc_id": hit.get("doc_id", ""),
                "title": hit.get("title", ""),
            })

    return rows


def flatten_hybrid_candidates(parsed_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for idx, s in enumerate(parsed_logs, start=1):
        question = s.get("question", "")
        candidates = ((s.get("search", {}) or {}).get("hybrid_candidates", [])) or []

        for rank, c in enumerate(candidates, start=1):
            rows.append({
                "session_index": idx,
                "question": question,
                "rank_in_log": rank,
                "score": c.get("score", ""),
                "dense_score": c.get("dense_score", ""),
                "sparse_score": c.get("sparse_score", ""),
                "sum_score": c.get("sum_score", ""),
                "title": c.get("title", ""),
            })

    return rows


def flatten_max_score_candidates(parsed_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for idx, s in enumerate(parsed_logs, start=1):
        question = s.get("question", "")
        candidates = ((s.get("search", {}) or {}).get("max_score_candidates", [])) or []

        for rank, c in enumerate(candidates, start=1):
            rows.append({
                "session_index": idx,
                "question": question,
                "rank_in_log": rank,
                "score": c.get("score", ""),
                "dense_score": c.get("dense_score", ""),
                "sparse_score": c.get("sparse_score", ""),
                "sum_score": c.get("sum_score", ""),
                "title": c.get("title", ""),
            })

    return rows


def flatten_fetched_blocks(parsed_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for idx, s in enumerate(parsed_logs, start=1):
        question = s.get("question", "")
        blocks = ((s.get("search", {}) or {}).get("fetched_blocks", [])) or []

        for block in blocks:
            rows.append({
                "session_index": idx,
                "question": question,
                "level": block.get("level", ""),
                "number": block.get("number", ""),
                "text_len": block.get("text_len", ""),
                "score": block.get("score", ""),
                "source": block.get("source", ""),
            })

    return rows


def flatten_sources(parsed_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for idx, s in enumerate(parsed_logs, start=1):
        question = s.get("question", "")
        context = s.get("context", {}) or {}

        sources = context.get("final_sources", []) or context.get("sources", []) or []

        for rank, src in enumerate(sources, start=1):
            rows.append({
                "session_index": idx,
                "question": question,
                "source_rank": rank,
                "source": src,
            })

    return rows


def build_presentation_md(
    sessions_rows: list[dict[str, Any]],
    pass1_rows: list[dict[str, Any]],
    pass2_rows: list[dict[str, Any]],
    sources_rows: list[dict[str, Any]],
) -> str:
    sessions_headers = [
        "session_index",
        "question_type",
        "confidence",
        "search_strategy",
        "selected_doc_title",
        "context_chars",
        "time_total",
        "question",
    ]

    pass_headers = [
        "session_index",
        "rank",
        "score",
        "level",
        "number",
        "title",
    ]

    sources_headers = [
        "session_index",
        "source_rank",
        "source",
    ]

    parts = [
        "# Таблицы по распарсенным логам",
        "",
        "## 1. Сессии / запросы",
        markdown_table(sessions_rows, sessions_headers),
        "",
        "## 2. Pass1 hits",
        markdown_table(pass1_rows[:30], pass_headers),
        "",
        "## 3. Pass2 hits",
        markdown_table(pass2_rows[:30], pass_headers),
        "",
        "## 4. Источники",
        markdown_table(sources_rows[:30], sources_headers),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSV/MD tables from parsed_logs.json")
    parser.add_argument(
        "--input",
        default="parsed_logs.json",
        help="Путь к parsed_logs.json",
    )
    parser.add_argument(
        "--outdir",
        default="parsed_logs_tables",
        help="Папка для таблиц",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    parsed_logs = load_json(input_path)

    sessions_rows = flatten_sessions(parsed_logs)
    pass1_rows = flatten_hits(parsed_logs, "pass1_hits")
    pass2_rows = flatten_hits(parsed_logs, "pass2_section_hits")
    hybrid_rows = flatten_hybrid_candidates(parsed_logs)
    max_score_rows = flatten_max_score_candidates(parsed_logs)
    fetched_blocks_rows = flatten_fetched_blocks(parsed_logs)
    sources_rows = flatten_sources(parsed_logs)

    write_csv(
        outdir / "sessions_table.csv",
        sessions_rows,
        [
            "session_index",
            "question",
            "bitrix_message",
            "question_type",
            "answer_mode",
            "decompose",
            "confidence",
            "rewrite_entity",
            "rewrite_attribute",
            "search_strategy",
            "search_text",
            "search_mode",
            "search_level",
            "search_limit",
            "pass1_hit_count",
            "pass2_section_hit_count",
            "selected_doc_id",
            "selected_doc_title",
            "anchor_number",
            "anchor_main_section_id",
            "top_section_number",
            "context_chars",
            "context_blocks",
            "final_context_chars",
            "sources_count",
            "answer_template",
            "answer_context_chars",
            "raw_response_chars",
            "final_response_chars",
            "time_classify",
            "time_rewrite",
            "time_search_fetch",
            "time_answer",
            "time_total",
            "http_calls_count",
            "question_mismatch",
        ],
    )

    write_csv(
        outdir / "pass1_hits_table.csv",
        pass1_rows,
        ["session_index", "question", "rank", "score", "level", "number", "doc_id", "title"],
    )

    write_csv(
        outdir / "pass2_hits_table.csv",
        pass2_rows,
        ["session_index", "question", "rank", "score", "level", "number", "doc_id", "title"],
    )

    write_csv(
        outdir / "hybrid_candidates_table.csv",
        hybrid_rows,
        ["session_index", "question", "rank_in_log", "score", "dense_score", "sparse_score", "sum_score", "title"],
    )

    write_csv(
        outdir / "max_score_candidates_table.csv",
        max_score_rows,
        ["session_index", "question", "rank_in_log", "score", "dense_score", "sparse_score", "sum_score", "title"],
    )

    write_csv(
        outdir / "fetched_blocks_table.csv",
        fetched_blocks_rows,
        ["session_index", "question", "level", "number", "text_len", "score", "source"],
    )

    write_csv(
        outdir / "sources_table.csv",
        sources_rows,
        ["session_index", "question", "source_rank", "source"],
    )

    md_text = build_presentation_md(sessions_rows, pass1_rows, pass2_rows, sources_rows)
    (outdir / "presentation_tables.md").write_text(md_text, encoding="utf-8")

    print(f"Готово. Таблицы сохранены в: {outdir.resolve()}")
    print("Созданы файлы:")
    print(" - sessions_table.csv")
    print(" - pass1_hits_table.csv")
    print(" - pass2_hits_table.csv")
    print(" - hybrid_candidates_table.csv")
    print(" - max_score_candidates_table.csv")
    print(" - fetched_blocks_table.csv")
    print(" - sources_table.csv")
    print(" - presentation_tables.md")


if __name__ == "__main__":
    main()