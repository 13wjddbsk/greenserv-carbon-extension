# GreenServ Carbon Extension

**Carbon Emission as an Evaluation Metric for Environmentally Efficient LLM Inference Routing**

Yuna Jung · Claremont McKenna College · Summer Research Project 2026

---

## Overview

This repository extends [GreenServ](https://github.com/TZData1/llm-inference-router) (Ziller et al., 2026), a contextual multi-armed bandit LLM inference routing framework, by attempting to integrate [CodeCarbon](https://github.com/mlco2/codecarbon) to measure carbon emissions alongside GreenServ's existing GPU energy metric.

**Research Question:** Is GPU energy consumption (Wh) a sufficient proxy for carbon emissions (gCO₂) when evaluating the environmental friendliness of LLM inference routing algorithms?

GreenServ routes queries across a pool of 16 language models (0.5B–34B parameters) using LinUCB, optimizing a tradeoff between accuracy and GPU energy. It measures energy with the Zeus library but does not measure carbon. This project adds carbon measurement using CodeCarbon.

---

## Repository Structure

```
greenserv-carbon-extension/
├── README.md
├── scripts/
│   ├── carbon_benchmark.py          # Experiment 1: per-model carbon measurement (10 queries each)
│   ├── live_inference_carbon.py     # Experiment 2a: per-model carbon on full 2,500 GreenServ queries
│   └── routing_carbon_experiment.py # Experiment 2b: live routing with CodeCarbon per algorithm
├── modified_greenserv/
│   ├── energy.py                    # src/metrics/energy.py with CodeCarbonMeasurement class added
│   ├── inference_service.py         # src/services/inference_service.py with CodeCarbon alongside Zeus
│   └── analysis_utils.py            # experiments/shared/analysis_utils.py with carbon calculation
├── results/
│   └── live_carbon_results.csv      # Completed per-model carbon measurements (9/15 models)
└── report/
    └── yuna_srp_report.tex          # Full research report (LaTeX)
```

---

## Experiments

### Experiment 1: Per-Model Carbon Benchmark (`carbon_benchmark.py`)
Runs each of GreenServ's 16 models on 10 general-knowledge queries with CodeCarbon measuring carbon emissions. Establishes baseline carbon per model.

**Key finding:** Carbon scales with parameter count but is also influenced by model architecture — Phi-4-mini (3.8B) emits less carbon than some larger Qwen models.

### Experiment 2a: Live Per-Model Inference (`live_inference_carbon.py`)
Runs each model on GreenServ's full 2,500-query benchmark dataset (MMLU, GSM8K, HellaSwag, WinoGrande, CNN/DailyMail) with CodeCarbon measuring carbon per model. This is the two-phase approach needed to attribute carbon to routing algorithms.

**Status:** Partially completed (9/15 models) at time of report.

### Experiment 2b: Live Routing Experiment (`routing_carbon_experiment.py`)
Runs all 8 GreenServ routing strategies (LinUCB, Thompson Sampling, ε-Greedy NC/C, Random, Largest, Smallest, Accuracy) live on 2,500 queries with CodeCarbon measuring carbon per routing decision.

**Status:** Did not complete within project timeframe due to model loading overhead (see Limitations).

---

## GreenServ Modifications

Three files from GreenServ were modified:

### `src/metrics/energy.py` → `modified_greenserv/energy.py`
Added `CodeCarbonMeasurement` class using `OfflineEmissionsTracker`. Sits alongside existing `ZeusMeasurement` and `PynvmlMeasurement` classes.

### `src/services/inference_service.py` → `modified_greenserv/inference_service.py`
Added `carbon_tracker` alongside Zeus's `energy_measurement`. Both start and stop around `model.generate()` during each inference call.

### `experiments/shared/analysis_utils.py` → `modified_greenserv/analysis_utils.py`
Added derived carbon calculation: converts Zeus's joule measurements to gCO₂ using Southern California EPA eGRID carbon intensity (230 gCO₂/kWh) as a fallback when CodeCarbon is not available during pregenerated replay experiments.

---

## Infrastructure

- **Cluster:** Laguna HPC (USC/CARC), accessed by Claremont McKenna College
- **GPU:** NVIDIA L40S (46 GB VRAM × 2 per node = 92 GB total)
- **CPU:** AMD EPYC 9354 32-core, 128 GB RAM
- **Job scheduler:** SLURM (`sbatch`)
- **Container runtime:** Apptainer (Docker replacement for HPC)
- **Database:** PostgreSQL 15 (via Apptainer), loaded with GreenServ's 40,000 pregenerated inference results

### Docker → Apptainer Migration
GreenServ requires Docker to run PostgreSQL. Laguna does not support Docker. Migration steps:
```bash
# Pull PostgreSQL image via Apptainer
apptainer pull docker://postgres:15

# Initialize database directory in project storage
apptainer exec --bind /path/to/pgdata:/var/lib/postgresql/data \
  postgres_15.sif initdb -D /var/lib/postgresql/data

# Start with writable socket directory
apptainer exec --bind /path/to/pgdata:/var/lib/postgresql/data,/path/to/pgsocket:/var/run/postgresql \
  postgres_15.sif postgres -D /var/lib/postgresql/data -k /var/run/postgresql &
```

---

## Known Limitations

### Model Loading Overhead
The live routing experiment (Experiment 2b) could not complete within the project timeframe. When a routing algorithm switches between models mid-experiment, each model switch requires loading weights from disk (30–120 seconds per load). With 15 models and 2,500 queries, this overhead made the experiment infeasible.

**Two architectural solutions for future work:**
1. Preload all models simultaneously — requires ~150–200 GB VRAM (exceeds 92 GB available)
2. Two-phase carbon attribution — run each model separately on all queries (Experiment 2a), then use GreenServ's pregenerated routing decisions to attribute carbon to algorithms post-hoc

### Inference-Only Measurement
Both Zeus and CodeCarbon measure only inference carbon. Training carbon for the 16 pretrained models is not captured.

### Offline Carbon Intensity
`OfflineEmissionsTracker` uses US national average carbon intensity (~386 gCO₂/kWh). California's grid is cleaner than the national average, so measurements are conservative overestimates.

---

## Results

### Per-Model Carbon (10 queries, Laguna L40S GPU)

| Model | Parameters | Carbon (gCO₂) |
|-------|-----------|----------------|
| Qwen2.5-0.5B | 0.5B | 0.171 |
| Qwen2.5-1.5B | 1.5B | 0.207 |
| Phi-4-mini-4B | 3.8B | 0.213 |
| Qwen2.5-3B | 3B | 0.221 |
| Qwen2.5-7B | 7B | 0.355 |
| Mistral-7B | 7B | 0.373 |
| Gemma-3-1B | 1B | 0.634 |
| Qwen2.5-14B | 14B | 0.646 |
| Phi-4-14B | 14B | 0.925 |
| Gemma-3-4B | 4B | 1.234 |
| Llama-3.1-8B | 8B | 1.807 |
| Gemma-3-12B | 12B | 2.529 |
| Gemma-3-27B | 27B | 4.470 |

### Per-Model Carbon (2,500 queries, Laguna L40S GPU) — Partial Results

| Model | Total Carbon (gCO₂) | Carbon/Query (gCO₂) |
|-------|--------------------|--------------------|
| Qwen2.5-0.5B | 80.63 | 0.0323 |
| Qwen2.5-1.5B | 105.33 | 0.0421 |
| Qwen2.5-3B | 144.47 | 0.0578 |
| Qwen2.5-7B | 283.75 | 0.1135 |
| Qwen2.5-14B | 853.74 | 0.3415 |
| Mistral-7B | 186.09 | 0.0744 |
| Gemma-3-1B | 126.85 | 0.0507 |
| Gemma-3-4B | 193.31 | 0.0773 |
| Gemma-3-12B | 250.84 | 0.1003 |

---

## Setup

### Prerequisites
- Python 3.11+
- NVIDIA GPU (tested on L40S)
- Apptainer (for PostgreSQL on HPC)
- HuggingFace account + token (for gated models: Gemma, Llama)

### Installation
```bash
git clone https://github.com/<your-username>/greenserv-carbon-extension
cd greenserv-carbon-extension

# Clone base GreenServ repo
git clone https://github.com/TZData1/llm-inference-router
cd llm-inference-router
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install codecarbon

# Apply modifications
cp ../modified_greenserv/energy.py src/metrics/energy.py
cp ../modified_greenserv/inference_service.py src/services/inference_service.py
cp ../modified_greenserv/analysis_utils.py experiments/shared/analysis_utils.py

export HF_TOKEN="your_token_here"
export HF_HOME="/path/to/hf_cache"
```

### Running Experiment 1 (Per-Model Benchmark)
```bash
python3 scripts/carbon_benchmark.py
```

### Running Experiment 2a (Live Per-Model Inference)
```bash
# Start PostgreSQL first (see Docker → Apptainer Migration above)
python3 scripts/live_inference_carbon.py
```

---

## Citation

If you use this work, please also cite GreenServ:
```
Ziller, T., et al. GreenServ: Energy-Efficient LLM Inference Routing via Contextual Bandits. 2026.
```

And CodeCarbon:
```
Courty, B., et al. CodeCarbon: Estimate and Track Carbon Emissions from Machine Learning Computing. 2023.
```

---

## Acknowledgments

This research was conducted as part of the CMC Summer Research Program 2026 at the Murty Sunak Quantitative Computing Lab, under the supervision of Professor Jeho Park. Computing resources provided by USC CARC (Laguna HPC cluster). Claude.ai (Anthropic) was used as a coding and debugging assistant with advisor approval.
