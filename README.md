# KV Cache Internet Project Playground

This repository contains supporting data and measurements for **[An Internet for the KV Cache: Rethinking Classical Infrastructure Boundaries in the LLM Inference Age](https://arxiv.org/abs/2608.01526)** by Siddhant Ray, Nick Feamster, and Junchen Jiang (2026).

Research scripts for studying LLM agent workloads, workflow orchestration, memory reuse, and inference performance. The repository combines synthetic multi-agent workflow generation, GAIA dataset analysis, LangGraph trace recording and replay, a two-agent document question-answering demo, and experiments comparing prefix caching with shorter contexts.

This is a collection of standalone experiments, with no packaged application, pinned dependency manifest, or automated test suite. Some workflows use local stubs; others require a running model server and external tools.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/` | Supporting KV-cache storage and transfer analyses, vendor cost comparisons, and model-family ratio plots for the paper. |
| `simulations/generator/workflows/` | Synthetic workflow generators, branching and replanning logic, trace formatting, and sample traces. |
| `simulations/scheduler/multiqueue.py` | Asyncio simulation of three priority queues with request demotion and variable service times. |
| `scripts/` | Download dataset validation splits and summarize GAIA annotation metadata. |
| `benchmarks/gaia/` | GAIA JSONL data, task examples, and summary CSV. |
| `benchmarks/dummy/` | vLLM prompt-length latency benchmark and an example server launch script. |
| `trace_replay/langgraph/` | Tool-assisted execution of annotated task steps, JSON checkpoints, and a separate SQLite logging variant. |
| `examples/two-agent.py` | Interactive document QA with a memory manager and a QA agent. |
| `plots/` | Prefix-cache versus compaction cost, latency, and KV-cache plots using values embedded in the scripts. |
| `visualize/visuazlize_inp_out.py` | Token-overlap visualization using longest common substrings in example inputs and outputs. |

## Supporting data

The [`data/`](data/) directory contains five Excel workbooks, one CSV, and 16 PNG plots exploring KV-cache storage and transfer relative to prefill computation.

| File or directory within `data/` | Contents |
| --- | --- |
| [`kv_1000token_fp16_only_storage_internet_colored.xlsx`](data/kv_1000token_fp16_only_storage_internet_colored.xlsx) | FP16 KV-cache sizes and prefill computation for 1,000 tokens across 65 models, with model family, release date, architecture, MB/TFLOP ratios, and storage thresholds for one hour, day, month, and year. Includes a `Calculations` sheet with assumptions and formulas. |
| [`kv_1000token_fp8_storage_k_ratios_colored.xlsx`](data/kv_1000token_fp8_storage_k_ratios_colored.xlsx) | Corresponding FP8 KV-cache analysis for 65 models, including storage-duration thresholds and a `Calculations` sheet. |
| [`kv_h100_transfer_break_even_100mbps_effective_colored.xlsx`](data/kv_h100_transfer_break_even_100mbps_effective_colored.xlsx) | H100 transfer break-even analysis at an effective 100 Mbps for 65 models, with KV size per token, prefill TFLOPs, current ratios, and break-even comparisons. The workbook preserves colored check/cross indicators. |
| [`kv_h100_transfer_break_even_100mbps_effective_full.csv`](data/kv_h100_transfer_break_even_100mbps_effective_full.csv) | CSV version of the transfer break-even table, retaining indicator text without Excel formatting. |
| [`b200_1month_minimum_ratios_by_vendor.xlsx`](data/b200_1month_minimum_ratios_by_vendor.xlsx) | B200 GPU and storage pricing comparisons for eight vendor pairs, with one-month minimum ratios in MB/TFLOP, source URLs, notes, and an `Assumptions` sheet. |
| [`b200_minimum_ratios_1h_1d_1m_1y_by_vendor.xlsx`](data/b200_minimum_ratios_1h_1d_1m_1y_by_vendor.xlsx) | B200 vendor comparisons extended to one-hour, one-day, one-month, and one-year storage durations, with supporting calculations and source URLs. |
| [`family_size_grouped_ratio_plots_with_glm/`](data/family_size_grouped_ratio_plots_with_glm/) | 16 plots of current KV-cache-to-prefill ratios versus model release date, grouped by model family and size. Covers Llama, Qwen, Gemma, Mistral, Phi, DeepSeek, Kimi, MiniMax, and GLM. |

Open the `.xlsx` files in a spreadsheet application to inspect the assumptions and formatting; use the CSV for programmatic analysis. Units and normalization differ between the storage and transfer tables, so consult their column headers and calculation sheets before comparing ratios. These files contain saved analysis inputs and results; vendor prices and hardware assumptions are snapshots, not live values.

## Setup

Use Python 3.10 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies for the experiment you want to run:

| Experiment | Python dependencies |
| --- | --- |
| Synthetic workflows, scheduler, existing dataset analysis | Standard library only |
| Dataset download | `datasets` |
| Document QA | `httpx` |
| Compaction benchmark client | `requests transformers` |
| Plots and overlap visualization | `matplotlib numpy` |
| JSON trace recording/replay | `langgraph langchain-openai langchain-community playwright duckduckgo-search requests pymupdf pillow pytesseract beautifulsoup4` |
| SQLite trace logging variant | Trace dependencies above, plus `langgraph-checkpoint-sqlite` |

For example:

```bash
python -m pip install matplotlib numpy
```

Dependency versions are not pinned; compatibility with a particular environment must be checked locally. Live inference experiments require a separately configured vLLM/OpenAI-compatible server. Browser recording also needs Playwright's Chromium (`python -m playwright install chromium`), and PDF OCR needs the Tesseract executable installed on the host.

## Generate synthetic workflows

The generators use stub LLM and web-search backends, so they run without API credentials or a GPU. Version 2 generates seeded plans with workers, optional tools, branches, joins, evaluation, and replanning, then saves a readable execution trace.

```bash
python simulations/generator/workflows/generator_v2.py --seed 42
```

This writes `traces/seed42_0.txt` relative to the current working directory. Repeating a seed can overwrite its trace. Existing sample traces are in `simulations/generator/workflows/traces/`.

The earlier generator prints its trace to the terminal:

```bash
python simulations/generator/workflows/generator.py
```

The workflow `run.sh` wrapper generates 100 seeds but deletes existing `traces/*.txt` first. Use individual Python commands to preserve previous traces. The generators' Python execution helpers are intended for these controlled examples, not untrusted code.

## Run the scheduler simulation

```bash
python simulations/scheduler/multiqueue.py
```

The default simulation runs for about 40 seconds and logs arrivals, queue transitions, and completions. Parameters are constants at the top of the script. Demotion thresholds use elapsed simulation time, rather than each request's age.

## Work with GAIA data

To analyze the included data:

```bash
python scripts/analyse_dataset.py \
  --dataset_name gaia-benchmark/GAIA --save_folder gaia
```

The script reads `benchmarks/gaia/dataset.jsonl`, creates `dataset_stats.csv` if it does not exist, and reports tool counts, step counts, and annotation timing statistics. An existing CSV is reused, so move it aside before regenerating statistics for changed data.

To download the validation split again, install `datasets`, configure your own Hugging Face credentials in the environment if required, and run:

```bash
python scripts/generate_dataset.py \
  --dataset_name gaia-benchmark/GAIA --save_folder gaia
```

For GAIA this selects the `2023_all` configuration and writes `benchmarks/gaia/dataset.jsonl`, replacing the existing file. Dataset access is subject to upstream access requirements. The shell wrappers in `scripts/` contain a hardcoded credential; use the Python commands above and your own environment configuration instead.

## Record and replay annotated tasks

`replay_langgraph.py` maps GAIA annotator steps into a LangGraph workflow using browser navigation, search, PDF OCR, and LLM calls. Its step handlers include task-specific heuristics; it is not a general GAIA solver.

With the trace dependencies installed, inspect a bundled checkpoint without making live model or browser calls:

```bash
JSON_CKPT_DIR=trace_replay/langgraph/checkpoints_json \
python trace_replay/langgraph/replay_langgraph.py \
  --task_json benchmarks/gaia/json1.json --mode replay
```

Replay loads the saved result for the task ID; it does not re-execute the tools. To record against your server:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY
export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
export JSON_CKPT_DIR=checkpoints_json

python trace_replay/langgraph/replay_langgraph.py \
  --task_json benchmarks/gaia/json1.json --mode record
```

Set the model and API key to match your server. Recording writes JSON checkpoints containing step inputs/outputs, tool logs, timings, and scratch state. PDF artifacts are written under `downloads/`; these paths are relative to the working directory. `replay_langgraph_logging.py` is a separate variant using SQLite, configured through `SQLITE_PATH` (default: `trace_checkpoints.sqlite`) and `MAX_LOG_CHARS`.

## Try two-agent document QA

With `httpx` installed and a model server running:

```bash
python examples/two-agent.py \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --api-key EMPTY \
  --doc-path /path/to/document.txt
```

Omit `--doc-path` to use the built-in example document. Enter questions at the prompt and type `exit` to quit. The demo chunks the document, retrieves relevant blocks, and coordinates memory selection with answer generation. `--retrieval-k` and `--memory-k` control the selection sizes. Environment defaults use `VLLM_BASE_URL`, `VLLM_API_KEY`, and `VLLM_MODEL` (distinct from the trace script's `MODEL_NAME`).

## Benchmark prefix caching and shorter contexts

`benchmarks/dummy/bench.py` builds synthetic prompts and measures completion-request latency against a running vLLM server:

```bash
python benchmarks/dummy/bench.py --mode 10k
python benchmarks/dummy/bench.py --mode 500k
```

The default model and tokenizer are `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct`. Override `--base-url`, `--model`, `--tokenizer`, and `--api-key` to match your deployment. The server needs sufficient context capacity and GPU memory; the example `benchmarks/dummy/run.sh` launches vLLM on four GPUs with prefix caching enabled.

- `10k`, `20k`, `50k`, `75k`, `100k`, and `200k` add a unique nonce to reduce cache reuse between invocations.
- `500k` sends a warm-up request, then measures repeated requests with the same prompt.
- JSON output includes per-request latency, token usage, and summary statistics. `--retries` sets the number of measured requests (default: one).

The shorter-context modes approximate compacted input lengths; the benchmark does not summarize a real conversation or measure answer quality. Within a single invocation, repeated baseline requests reuse the same prompt, so increasing `--retries` can introduce cache hits even in an “uncached” mode.

## Generate figures

Run plotting scripts from `plots/` so their relative output path resolves to this repository's `figures/` directory:

```bash
cd plots
python prefix_cache_vs_compaction_cost.py
python prefix_cache_vs_compaction_latency.py
python prefix_cache_vs_compaction_kv.py
```

These write PDFs under `figures/YYYY-MM-DD/`. Their data is hardcoded; they do not read benchmark output or fetch current model pricing. Update the arrays in the scripts for new measurements.

From the repository root, run the standalone token-overlap example with:

```bash
python visualize/visuazlize_inp_out.py
```

## Paper reference

If you use the supporting data, measurements, or code in this repository, please cite the paper:

```bibtex
@misc{ray2026internetkvcache,
  title = {An Internet for the KV Cache: Rethinking Classical Infrastructure Boundaries in the LLM Inference Age},
  author = {Siddhant Ray and Nick Feamster and Junchen Jiang},
  year = {2026},
  eprint = {2608.01526},
  archivePrefix = {arXiv},
  primaryClass = {cs.NI},
  url = {https://arxiv.org/abs/2608.01526}
}
```

## License

Original project code is licensed under the [MIT License](LICENSE). Third-party datasets, downloaded documents, model assets, and content captured in traces remain subject to their respective upstream terms.
