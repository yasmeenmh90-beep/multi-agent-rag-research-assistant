"""
Evaluates the pipeline on a small held-out question set using RAGAS metrics:
- faithfulness: is the answer supported by the retrieved context?
- context_precision: how much of the retrieved context was actually relevant?

This is what turns "it works well" into a number you can put in a README
or say out loud in an interview. Edit EVAL_QUESTIONS below to match your
actual corpus before running.

Run standalone:
    python -m app.eval.ragas_eval

Or call run_evaluation() directly from other code (e.g. a FastAPI endpoint)
to get the same results back as a plain dict instead of printed output.
"""
from datasets import Dataset
from pathlib import Path
import asyncio
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.run_config import RunConfig

from app.agents.graph import run_query

# Questions matched to the actual ingested corpus (ai_papers: RAG techniques,
# multi_agent: multi-agent LLM/RL systems). ground_truth is optional for
# faithfulness but improves context_precision scoring - left blank here
# since we don't have hand-verified reference answers; add them if you want
# tighter precision scores.
EVAL_QUESTIONS = [
    {"question": "What techniques are used to reduce hallucination in retrieval-augmented generation systems?", "ground_truth": ""},
    {"question": "What are common architectures and frameworks used for building multi-agent LLM systems?", "ground_truth": ""},
    {"question": "How does chunking strategy affect retrieval quality in RAG systems?", "ground_truth": ""},
]


def build_eval_dataset():
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in EVAL_QUESTIONS:
        result = run_query(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(result.get("final_answer", ""))
        rows["contexts"].append([r["content"] for r in result.get("retrieved", [])])
        rows["ground_truth"].append(item.get("ground_truth", ""))

    return Dataset.from_dict(rows)


def run_evaluation() -> dict:
    """
    Runs the full RAGAS eval and returns a plain dict:
        {
          "num_questions": int,
          "faithfulness_avg": float,
          "context_precision_avg": float,
          "per_question": [
            {"question": str, "faithfulness": float, "context_precision": float},
            ...
          ]
        }
    Also writes ragas_results.csv as before, so the CLI behavior is unchanged.
    """
    dataset = build_eval_dataset()

    # RAGAS's evaluate() uses asyncio internally and expects an event loop
    # to already exist in the current thread. FastAPI runs sync `def`
    # endpoints (like /eval/run) in a background worker thread that has
    # no event loop by default, which crashes RAGAS with "There is no
    # current event loop in thread 'AnyIO worker thread'". Creating and
    # setting one here fixes it for whichever thread this function runs in
    # (CLI's main thread already has one, so this is a no-op there).
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    results = evaluate(
        dataset,
        metrics=[faithfulness, context_precision],
        # Default concurrency fires several OpenAI calls at once, which on
        # lower-tier rate limits triggers silent retries with backoff -
        # this looks exactly like a hang (progress bar stuck at 0%) even
        # though nothing crashed. max_workers=2 keeps requests sequential
        # enough to avoid that, at the cost of being a bit slower overall.
        run_config=RunConfig(max_workers=2, timeout=120),
    )
    df = results.to_pandas()

    out_path = Path(__file__).resolve().parent / "ragas_results.csv"
    df.to_csv(out_path, index=False)

    per_question = [
        {
            "question": row["question"],
            "faithfulness": round(float(row["faithfulness"]), 3),
            "context_precision": round(float(row["context_precision"]), 3),
        }
        for _, row in df.iterrows()
    ]

    return {
        "num_questions": len(EVAL_QUESTIONS),
        "faithfulness_avg": round(float(df["faithfulness"].mean()), 3),
        "context_precision_avg": round(float(df["context_precision"].mean()), 3),
        "per_question": per_question,
    }


def main():
    print(f"Running RAGAS evaluation on {len(EVAL_QUESTIONS)} questions...\n")
    result = run_evaluation()

    print("=" * 70)
    print("RESULTS PER QUESTION")
    print("=" * 70)
    for row in result["per_question"]:
        print(f"\nQ: {row['question']}")
        print(f"   faithfulness:      {row['faithfulness']:.2f}")
        print(f"   context_precision: {row['context_precision']:.2f}")

    print("\n" + "=" * 70)
    print("AVERAGES (report these)")
    print("=" * 70)
    print(f"faithfulness:      {result['faithfulness_avg']:.3f}")
    print(f"context_precision: {result['context_precision_avg']:.3f}")

    out_path = Path(__file__).resolve().parent / "ragas_results.csv"
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()