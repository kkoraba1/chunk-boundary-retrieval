# Measuring the Effect of Shifted Chunk Boundaries on Evidence Retrieval

This study examines how fixed-size chunk boundaries can affect evidence retrieval even when the chunk size stays the same.

The central idea is that the same supporting evidence can fall fully inside one chunk under one boundary placement, but be split across two chunks after the boundaries shift. The experiment keeps the source document, query, chunk width, retriever settings, and evaluation setup fixed while changing the chunk boundary positions.

The study uses annotated grounding spans from Doc2Dial as gold supporting evidence. Documents are split into fixed-size chunks under different boundary phases, and the resulting passages are ranked with BM25 and E5-base-v2.

**Paper:** [Measuring the Effect of Shifted Chunk Boundaries on Evidence Retrieval](paper/paper.pdf)

## Research Question

> How does shifting fixed-size document chunk boundaries affect the retrievability of gold supporting evidence when the document, query, chunk width, retriever, and retrieval settings remain unchanged?

The study is motivated by retrieval-augmented generation (RAG), but it only studies the retrieval stage. It does not evaluate generated answers or try to find one chunking method that is best for every use case.

## Key Results

The main experiment used 94 held-out queries from 46 Doc2Dial documents. Each document was split into 200-word chunks under eight boundary phases.

Mean reciprocal rank was lower at the highest-fragmentation condition than at the lowest-fragmentation condition:

| Retriever | Lowest-EFS RR | Highest-EFS RR | Difference |
| --- | ---: | ---: | ---: |
| BM25 | 0.767 | 0.588 | -0.179 |
| E5-base-v2 | 0.715 | 0.644 | -0.071 |

The 95% query-bootstrap confidence interval was `[-0.238, -0.120]` for BM25 and `[-0.131, -0.010]` for E5-base-v2. When the data were resampled by source document instead of by query, the BM25 interval stayed below zero, while the E5 interval widened to include zero.

The full-index experiment showed the same overall direction for both retrievers. These full-index values are paired averages and do not have bootstrap confidence intervals.

Adding 40 words of overlap reduced mean EFS from `0.051` to less than `0.001`. However, overlap also removed the EFS variation needed for the planned low-versus-high fragmentation retrieval comparison in 92 of the 94 held-out queries. Because of that, the study does not claim that overlap reduced the retrieval effect of boundary placement.

Overall, the results show that chunk boundary placement can be linked to measurable changes in evidence retrieval, but the pattern was not the same for every query.

## Method

| Setting | Value |
| --- | --- |
| Dataset | Doc2Dial v1.0.1 |
| Development set | 472 examples from 111 documents |
| Held-out set | 94 queries from 46 documents |
| Chunk width | 200 source words |
| Primary overlap | 0 words |
| Boundary phases | 0, 25, 50, 75, 100, 125, 150, 175 |
| Retrieval scope | Known source document |
| Retrievers | BM25 and E5-base-v2 |
| Primary metric | Reciprocal Rank |
| Full-index metrics | graded nDCG@5 and oracle-normalized Evidence Coverage@5 |

The experiment uses Doc2Dial grounding annotations as gold evidence. The training split is used for development, and the validation split is used for the held-out evaluation. Each example must contain one continuous evidence region. That region must be shorter than the chunk width and must be able to be split by at least one tested boundary phase.

The main experiment shifts the starting position of a fixed 200-word chunk pattern across eight phases. For nonzero phases, the clipped prefix is kept so that shifting the boundary pattern does not drop any source words.

### Evidence Fragmentation Score

The Evidence Fragmentation Score (EFS) measures how much of the gold evidence is missing from the chunk that contains the largest share of it.

<p align="center">
  <strong>EFS(E, C) = 1 − max<sub>c ∈ C</sub> (|E ∩ c| / |E|)</strong>
</p>

Here, `E` is the gold evidence span and `C` is the set of chunks for one boundary phase.

An EFS of `0` means that at least one chunk contains the full evidence span. A larger EFS means that the evidence is split more strongly across chunk boundaries.

In the controlled probe, the chunk with the greatest evidence overlap is ranked against a fixed set of non-evidence distractors from the same source document. The distractor texts and candidate count stay fixed across phases.

For E5, the score of a fixed distractor does not depend on the other candidates. BM25 works differently. `BM25Okapi` is fit separately to each phase-specific candidate set, so changing the evidence passage can also change collection statistics used by BM25. The controlled probe therefore keeps the distractor texts fixed, but it does not keep every part of the BM25 scoring calculation fixed.

A second experiment ranks all chunks produced for each boundary phase from the known source document. BM25 is used as the sparse retriever, and `intfloat/e5-base-v2` is used as the dense retriever.

A separate overlap experiment compares zero overlap with 40 words of overlap while keeping the chunk width at 200 words.

## Repository Structure

```text
.
├── paper/
│   └── paper.pdf
├── results/
│   ├── dataset_summary.json
│   ├── error_cases.json
│   ├── overlap_ablation_results.csv
│   ├── primary_results.csv
│   └── figures/
│       └── efs_vs_rr.png
├── src/
│   └── boundary_retrieval/
│       ├── __init__.py
│       ├── analysis.py
│       ├── data.py
│       ├── evaluation.py
│       ├── experiment.py
│       ├── retrieval.py
│       └── segmentation.py
├── tests/
│   ├── test_analysis.py
│   ├── test_data.py
│   ├── test_evaluation.py
│   ├── test_experiment.py
│   ├── test_retrieval.py
│   └── test_segmentation.py
├── .gitignore
├── pyproject.toml
├── requirements-lock.txt
└── README.md
```

## Installation and Setup

The experiment was run with Python 3.11.

The study uses `document-chunking-utility` for fixed-size word chunking. Keep both repos inside the same parent directory:

```text
workspace/
├── document-chunking-utility/
└── chunk-boundary-retrieval/
```

After both repositories are in place, enter the retrieval directory:

```bash
cd chunk-boundary-retrieval
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install `document-chunking-utility` version 1.0 from the sibling directory:

```bash
python -m pip install -e ../document-chunking-utility
```

Install the locked environment:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements-lock.txt
```

Then install this repository without changing the dependency versions from the lock file:

```bash
python -m pip install -e . --no-deps
```

The first dense-retrieval run may download E5-base-v2 from Hugging Face if the model is not already cached. The implementation pins `intfloat/e5-base-v2` to revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`.

## Dataset Setup

The dataset is not stored in this repository. **Doc2Dial v1.0.1** can be found at the official release link below:

```text
https://doc2dial.github.io/file/doc2dial_v1.0.1.zip
```

The following command downloads the archive and extracts the Doc2Dial files directly into `data/`:

```bash
python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

url = "https://doc2dial.github.io/file/doc2dial_v1.0.1.zip"

data_dir = Path("data")
archive = data_dir / "doc2dial_v1.0.1.zip"

data_dir.mkdir(parents=True, exist_ok=True)

if not archive.exists():
    urlretrieve(url, archive)

with zipfile.ZipFile(archive) as zip_file:
    zip_file.extractall(data_dir)

print("Extracted Doc2Dial to: data")
PY
```

The expected SHA-256 for the Doc2Dial v1.0.1 archive used in this study is:

```text
94499fa5259f69018d2458cb948552e7a05f424711a95a562bb9e816a515dc23
```

Verify the downloaded archive with:

```bash
printf '%s  %s\n' \
  '94499fa5259f69018d2458cb948552e7a05f424711a95a562bb9e816a515dc23' \
  'data/doc2dial_v1.0.1.zip' | sha256sum -c -
```

## Reproducing the Experiments

The saved outputs from the completed study are already included under `results/`. Fresh runs can be written to a separate `reprod/` directory.

Regenerate the dataset summary:

```bash
mkdir -p reprod

python - <<'PY'
import json
from pathlib import Path

from boundary_retrieval.data import (
    build_dataset_summary,
    build_examples,
    load_dialogues,
    load_docs,
)

data_dir = Path("data")
output_path = Path("reprod/dataset_summary.json")

chunk_size = 200
phases = (0, 25, 50, 75, 100, 125, 150, 175)
archive_sha = "94499fa5259f69018d2458cb948552e7a05f424711a95a562bb9e816a515dc23"

documents = load_docs(data_dir / "doc2dial_doc.json")
train_dialogues = load_dialogues(data_dir / "doc2dial_dial_train.json")
validation_dialogues = load_dialogues(
    data_dir / "doc2dial_dial_validation.json"
)

train_examples = build_examples(
    train_dialogues,
    documents,
    chunk_size,
    phases,
)
validation_examples = build_examples(
    validation_dialogues,
    documents,
    chunk_size,
    phases,
)

summary = build_dataset_summary(
    train_examples,
    validation_examples,
    src_version="1.0.1",
    archive_sha=archive_sha,
    chunk_size=chunk_size,
    phases=phases,
)

output_path.write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)

print(f"Wrote {output_path}")
PY
```

Run the main experiment:

```bash
mkdir -p reprod

python -m boundary_retrieval.experiment \
  primary data reprod/primary_results.csv
```

Run the overlap ablation:

```bash
python -m boundary_retrieval.experiment \
  overlap data reprod/overlap_ablation_results.csv
```

Run the analysis on the freshly reproduced experiment outputs:

```bash
python -m boundary_retrieval.analysis \
  data \
  reprod/primary_results.csv \
  reprod/overlap_ablation_results.csv \
  reprod/analysis
```

This creates:

```text
reprod/analysis/analysis_summary.json
reprod/analysis/error_cases.json
reprod/analysis/figures/efs_vs_rr.png
```

The command rebuilds the numerical summaries used for the controlled probe, document-cluster sensitivity analysis, within-query Spearman analysis, full-index evaluation, and overlap analysis. It also uses the Doc2Dial validation data to reconstruct the qualitative error cases.

## Testing

The repo contains 170 automated tests covering the data pipeline, segmentation, retrieval, evaluation, analysis regeneration, and experiment behavior.

Run the full test suite with:

```bash
python -m pytest -q
```

## Scope and Limitations

This is a focused retrieval study, not a general evaluation of document chunking. It uses one filtered Doc2Dial subset, one main chunk width, one overlap setting, and two retrievers.

Retrieval is limited to the known source document. Because of that, the study does not test document selection across a large corpus. It also does not evaluate downstream language-model answer quality.

The overlap result shows that 40 words of overlap greatly reduced measured fragmentation in this setup. It does not show that 40 words is the best overlap amount or that overlap generally improves retrieval.

The BM25 controlled probe also has an important detail in that the distractor texts stay fixed, but BM25 collection statistics can still change when the phase-specific evidence passage changes.

## Acknowledgments

This study uses:

- **[Doc2Dial v1.0.1](https://doc2dial.github.io/)** for the source documents, queries, and grounding annotations.
- **[Document Chunking Utility v1.0](https://github.com/kkoraba1/document-chunking-utility)** for fixed-size word chunking.
- **BM25** through `rank_bm25.BM25Okapi`.
- **E5-base-v2**, loaded with **Sentence Transformers**, for dense retrieval.