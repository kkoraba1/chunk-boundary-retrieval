import csv
import json

import pytest

from boundary_retrieval import analysis


def test_load_results_csv_reads_rows(tmp_path):
    path = tmp_path / "results.csv"
    fields = ["query_id", "retriever", "phase", "efs", "rr"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q1",
                "retriever": "bm25",
                "phase": 0,
                "efs": 0.0,
                "rr": 1.0,
            }
        )

    rows = analysis.load_results_csv(path)

    assert rows == [
        {
            "query_id": "q1",
            "retriever": "bm25",
            "phase": "0",
            "efs": "0.0",
            "rr": "1.0",
        }
    ]


def test_load_results_csv_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("query_id,retriever\nq1,bm25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        analysis.load_results_csv(path)


def test_count_overlap_efs_variation_counts_queries_once():
    rows = [
        {"query_id": "q1", "retriever": "bm25", "efs": 0.0},
        {"query_id": "q1", "retriever": "e5", "efs": 0.0},
        {"query_id": "q1", "retriever": "bm25", "efs": 0.2},
        {"query_id": "q1", "retriever": "e5", "efs": 0.2},
        {"query_id": "q2", "retriever": "bm25", "efs": 0.0},
        {"query_id": "q2", "retriever": "e5", "efs": 0.0},
    ]

    summary = analysis._count_overlap_efs_variation(rows)

    assert summary == {
        "n_queries": 2,
        "n_variable_efs": 1,
        "n_constant_efs": 1,
    }


def test_save_analysis_summary_writes_formatted_json(tmp_path):
    path = tmp_path / "nested" / "analysis_summary.json"
    summary = {"controlled_probe": {"bm25": {"mean_rr_difference": -0.1}}}

    saved = analysis.save_analysis_summary(summary, path)

    assert saved == path
    assert json.loads(path.read_text(encoding="utf-8")) == summary
    assert path.read_text(encoding="utf-8").endswith("\n")
