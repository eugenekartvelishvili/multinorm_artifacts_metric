from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from datasets import Dataset

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall

from langchain_ollama import ChatOllama, OllamaEmbeddings


# =========================
# HARD DISABLE PROXY
# =========================
for key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(key, None)

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"


# =========================
# CONFIG
# =========================
DEFAULT_PARSED_PATH = Path("parsed_logs_29.json")
DEFAULT_GOLDEN_PATH = Path("golden_dataset_good.json")
DEFAULT_OUTPUT_CSV = Path("ragas_eval_results.csv")
DEFAULT_OUTPUT_JSON = Path("ragas_eval_results.json")

DEFAULT_OLLAMA_LLM = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_EMBED = "bge-m3"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

ALLOWED_01 = [round(x / 10, 1) for x in range(11)]


# =========================
# IO
# =========================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# TEXT HELPERS
# =========================
def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"[«»\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_match(text: str | None) -> str:
    text = normalize_text(text)
    text = re.sub(r"[(){}$begin:math:display$$end:math:display$,.:;!?/\\|+=*_~@#$%^&–—-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_get(obj: Any, *keys: str, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def truncate_text(text: str, max_chars: int = 12000) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def snap_01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    v = max(0.0, min(1.0, v))
    return min(ALLOWED_01, key=lambda x: abs(x - v))


# =========================
# EXTRACTORS
# =========================
def extract_answer(log_item: dict[str, Any]) -> str:
    answer = log_item.get("answer")

    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    if isinstance(answer, dict):
        val = (
            answer.get("text")
            or answer.get("final_response_preview")
            or answer.get("raw_response_preview")
            or ""
        )
        if str(val).strip():
            return str(val).strip()

    parsed = log_item.get("parsed")
    if isinstance(parsed, dict):
        val = parsed.get("answer") or ""
        if str(val).strip():
            return str(val).strip()

    return ""


def extract_contexts(log_item: dict[str, Any]) -> list[str]:
    contexts: list[str] = []

    context_text = safe_get(log_item, "context", "text")
    if isinstance(context_text, str) and context_text.strip():
        contexts.append(context_text.strip())

    answer_context_text = log_item.get("answer_context_text")
    if isinstance(answer_context_text, str) and answer_context_text.strip():
        contexts.append(answer_context_text.strip())

    fetched_blocks = safe_get(log_item, "search", "fetched_blocks", default=[])
    if isinstance(fetched_blocks, list) and fetched_blocks:
        for block in fetched_blocks:
            if not isinstance(block, dict):
                continue
            source = str(block.get("source") or "").strip()
            text_len = block.get("text_len")
            if source:
                if text_len is not None:
                    contexts.append(f"{source}\n[text_len={text_len}]")
                else:
                    contexts.append(source)

    final_sources = safe_get(log_item, "context", "final_sources", default=[]) or []
    sources = safe_get(log_item, "context", "sources", default=[]) or []

    if isinstance(final_sources, list):
        contexts.extend([str(x).strip() for x in final_sources if str(x).strip()])
    if isinstance(sources, list):
        contexts.extend([str(x).strip() for x in sources if str(x).strip()])

    seen = set()
    uniq = []
    for item in contexts:
        norm = normalize_text(item)
        if norm and norm not in seen:
            seen.add(norm)
            uniq.append(item)

    if not uniq:
        answer = extract_answer(log_item)
        if answer:
            uniq = [answer]

    return [truncate_text(x, 4000) for x in uniq[:20]]


def extract_total_time(log_item: dict[str, Any]) -> float | None:
    val = safe_get(log_item, "timing", "total")
    if val is None:
        val = safe_get(log_item, "parsed", "timing_total")
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


# =========================
# ALIGNMENT
# =========================
def build_golden_index(golden_items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for item in golden_items:
        q = normalize_for_match(item.get("question"))
        if not q:
            continue
        index.setdefault(q, []).append(item)
    return index


def align_items(
    parsed_logs: list[dict[str, Any]],
    golden_items: list[dict[str, Any]],
    limit: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    golden_subset = golden_items[:limit] if limit is not None and limit > 0 else golden_items
    golden_index = build_golden_index(golden_subset)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_golden_keys: set[tuple[Any, str]] = set()

    for log_item in parsed_logs:
        question = normalize_for_match(log_item.get("question"))
        if not question:
            continue

        candidates = golden_index.get(question, [])
        if not candidates:
            continue

        chosen = None
        for cand in candidates:
            unique_key = (
                cand.get("id"),
                normalize_for_match(cand.get("question")),
            )
            if unique_key not in used_golden_keys:
                chosen = cand
                used_golden_keys.add(unique_key)
                break

        if chosen is None:
            continue

        pairs.append((log_item, chosen))

    return pairs


# =========================
# DATASET BUILD
# =========================
def build_rows(
    parsed_logs: list[dict[str, Any]],
    golden_items: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    pairs = align_items(parsed_logs, golden_items, limit)

    rows: list[dict[str, Any]] = []
    for log_item, gold_item in pairs:
        question = str(log_item.get("question") or "").strip()
        answer = extract_answer(log_item)
        contexts = extract_contexts(log_item)

        ground_truth = str(
            gold_item.get("gold_answer")
            or gold_item.get("answer")
            or ""
        ).strip()

        if not question or not answer or not ground_truth:
            continue

        rows.append({
            "id": gold_item.get("id"),
            "question": question,
            "question_type": gold_item.get("question_type"),
            "answer": truncate_text(answer, 6000),
            "contexts": contexts,
            "ground_truth": truncate_text(ground_truth, 6000),
            "expected_doc": gold_item.get("expected_doc"),
            "expected_sections": gold_item.get("expected_sections"),
            "expected_key_points": gold_item.get("expected_key_points"),
            "timing_total": extract_total_time(log_item),
        })

    return rows


# =========================
# SIMPLE OLLAMA JUDGE FOR ANSWER RELEVANCY
# =========================
def call_ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int = 180,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "")).strip()


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Не удалось извлечь JSON:\n{text}")


def answer_relevancy_prompt(question: str, answer: str) -> str:
    return f"""
Оцени, насколько ответ релевантен вопросу.

Вопрос:
{question}

Ответ:
{answer}

Верни только JSON:
{{
  "answer_relevancy": 0.0
}}

Где:
0.0 = ответ вообще не по вопросу
0.5 = частично по вопросу
1.0 = полностью по вопросу

Разрешены промежуточные значения с шагом 0.1.
""".strip()


def compute_answer_relevancy_batch(
    rows: list[dict[str, Any]],
    model: str,
    base_url: str,
) -> list[float | None]:
    scores: list[float | None] = []

    for i, row in enumerate(rows, start=1):
        try:
            raw = call_ollama_generate(
                prompt=answer_relevancy_prompt(row["question"], row["answer"]),
                model=model,
                base_url=base_url,
                timeout=120,
            )
            parsed = extract_json_block(raw)
            score = snap_01(parsed.get("answer_relevancy"), default=0.5)
            scores.append(score)
            print(f"[answer_relevancy {i}/{len(rows)}] {score}")
        except Exception as e:
            print(f"[answer_relevancy {i}/{len(rows)}] error: {e}")
            scores.append(None)

    return scores


# =========================
# MAIN
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on parsed logs using local Ollama")
    parser.add_argument("--parsed", default=str(DEFAULT_PARSED_PATH), help="Путь к parsed logs JSON")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_PATH), help="Путь к golden dataset JSON")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Путь к выходному CSV")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Путь к выходному JSON")
    parser.add_argument("--llm-model", default=DEFAULT_OLLAMA_LLM, help="Модель Ollama для judge")
    parser.add_argument("--embed-model", default=DEFAULT_OLLAMA_EMBED, help="Embedding модель Ollama")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Base URL Ollama")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Сколько первых вопросов брать из golden dataset; 0 = все",
    )
    args = parser.parse_args()

    parsed_path = Path(args.parsed)
    golden_path = Path(args.golden)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    parsed_logs = load_json(parsed_path)
    golden_items = load_json(golden_path)

    rows = build_rows(
        parsed_logs=parsed_logs,
        golden_items=golden_items,
        limit=args.limit if args.limit > 0 else None,
    )

    if not rows:
        raise ValueError("Не удалось собрать строки для RAGAS. Проверь questions в parsed и golden.")

    print(f"Сопоставлено вопросов: {len(rows)}")

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    base_llm = ChatOllama(
        model=args.llm_model,
        base_url=args.ollama_base_url,
        temperature=0,
        client_kwargs={
            "trust_env": False,
            "timeout": 180,
        },
    )

    base_embeddings = OllamaEmbeddings(
        model=args.embed_model,
        base_url=args.ollama_base_url,
        client_kwargs={
            "trust_env": False,
            "timeout": 180,
        },
    )

    evaluator_llm = LangchainLLMWrapper(base_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
    ]

    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=2,
    )

    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    ragas_df = ragas_result.to_pandas()

    # добиваем answer_relevancy отдельно
    answer_relevancy_scores = compute_answer_relevancy_batch(
        rows=rows,
        model=args.llm_model,
        base_url=args.ollama_base_url,
    )
    ragas_df["answer_relevancy"] = answer_relevancy_scores

    meta_df = pd.DataFrame([{
        "id": r["id"],
        "question": r["question"],
        "question_type": r["question_type"],
        "expected_doc": r["expected_doc"],
        "expected_sections": json.dumps(r["expected_sections"], ensure_ascii=False),
        "expected_key_points": json.dumps(r["expected_key_points"], ensure_ascii=False),
        "timing_total": r["timing_total"],
        "answer": r["answer"],
        "ground_truth": r["ground_truth"],
        "contexts_joined": "\n---\n".join(r["contexts"]),
    } for r in rows])

    final_df = pd.concat([meta_df.reset_index(drop=True), ragas_df.reset_index(drop=True)], axis=1)

    metric_cols = [
        c for c in [
            "faithfulness",
            "llm_context_recall",
            "factual_correctness",
            "answer_relevancy",
        ]
        if c in final_df.columns
    ]

    if metric_cols:
        final_df["ragas_main_score"] = final_df[metric_cols].mean(axis=1, skipna=True)
    else:
        final_df["ragas_main_score"] = None

    summary = {
        "count": int(len(final_df)),
        "metrics_avg": {},
        "by_question_type": {},
    }

    for col in metric_cols + ["ragas_main_score"]:
        if col in final_df.columns:
            try:
                summary["metrics_avg"][col] = float(final_df[col].mean())
            except Exception:
                pass

    if "question_type" in final_df.columns and metric_cols:
        grouped = (
            final_df.groupby("question_type")[metric_cols + ["ragas_main_score"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary["by_question_type"] = grouped.to_dict(orient="records")

    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    payload = {
        "meta": {
            "parsed_path": str(parsed_path),
            "golden_path": str(golden_path),
            "output_csv": str(output_csv),
            "output_json": str(output_json),
            "limit": args.limit,
            "matched_pairs": len(rows),
            "llm_model": args.llm_model,
            "embed_model": args.embed_model,
            "ollama_base_url": args.ollama_base_url,
            "metrics": metric_cols,
            "note": (
                "RAGAS запущен на локальном Ollama. "
                "Proxy отключен через env cleanup + trust_env=False. "
                "Параллельность ограничена до 1 worker. "
                "answer_relevancy посчитан отдельно через локальный Ollama judge."
            ),
        },
        "summary": summary,
        "results": final_df.to_dict(orient="records"),
    }

    save_json(output_json, payload)

    print("\n=== RAGAS AVG ===")
    for k, v in summary["metrics_avg"].items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()