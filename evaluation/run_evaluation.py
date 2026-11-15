import argparse
import json
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluation.evaluator import RAGEvaluator


def print_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(col_widths[idx]) for idx, h in enumerate(headers)) + " |"

    print(sep)
    print(header_str)
    print(sep)
    for row in rows:
        row_str = "| " + " | ".join(str(val).ljust(col_widths[idx]) for idx, val in enumerate(row)) + " |"
        print(row_str)
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="DocuMind RAG Evaluation Benchmark")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM generation evaluation and only benchmark retrieval strategies."
    )
    parser.add_argument(
        "--output",
        default="evaluation/evaluation_report.json",
        help="Path to save the benchmark results JSON."
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("      DocuMind RAG Evaluation Benchmark Suite          ")
    print("=======================================================\n")

    evaluator = RAGEvaluator()
    print("Running comparative retrieval evaluation...")
    report = evaluator.run_full_benchmark(run_generation=not args.skip_llm)

    retrieval_data = report.get("retrieval_comparison", {})
    if retrieval_data:
        print("\n--- RETRIEVAL STRATEGY COMPARISON ---")
        headers = ["Strategy", "Recall@1", "Recall@3", "Recall@5", "MRR", "nDCG@3"]
        rows = []
        for strat, metrics in retrieval_data.items():
            rows.append([
                strat,
                f"{metrics['recall@1']:.2f}",
                f"{metrics['recall@3']:.2f}",
                f"{metrics['recall@5']:.2f}",
                f"{metrics['mrr']:.2f}",
                f"{metrics['ndcg@3']:.2f}",
            ])
        print_table(headers, rows)

    gen_data = report.get("generation_metrics", {})
    if gen_data:
        print("\n--- GENERATION & CITATION QUALITY ---")
        headers = ["Metric", "Score / Rate"]
        rows = [
            ["Citation Accuracy", f"{gen_data.get('citation_accuracy_percentage', 0.0)}%"],
            ["Mean Faithfulness Score", f"{gen_data.get('mean_faithfulness_score', 0.0):.2f} / 1.00"],
            ["Anti-Hallucination Guardrail Rate", f"{gen_data.get('anti_hallucination_block_rate', 0.0)}%"],
            ["Total Test Queries Evaluated", str(gen_data.get("total_evaluated_queries", 0))],
        ]
        print_table(headers, rows)

    output_path = root_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark report successfully exported to: {output_path}\n")


if __name__ == "__main__":
    main()
