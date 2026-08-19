# GreenServ Carbon Extension

**Yuna Jung · Claremont McKenna College Summer Research Program 2026**

Extends [GreenServ](https://github.com/TZData1/llm-inference-router) by integrating [CodeCarbon](https://github.com/mlco2/codecarbon) to measure carbon emissions (gCO₂) alongside GPU energy consumption in LLM inference routing.

**Research question:** Is GPU energy consumption a sufficient proxy for carbon emissions when evaluating the environmental impact of LLM routing algorithms?

---

## Scripts

| File | Description |
|------|-------------|
| `scripts/carbon_benchmark.py` | Measures carbon per model, 10 queries each |
| `scripts/live_inference_carbon.py` | Measures carbon per model, 2,500 GreenServ queries |
| `scripts/routing_carbon_experiment.py` | Live routing experiment with CodeCarbon per algorithm (incomplete — see limitations) |

## Results

Per-model carbon benchmark completed for 13 of 15 models. Live routing experiment did not complete due to model loading overhead — see report for details and proposed two-phase solution.

## Setup

```bash
git clone https://github.com/TZData1/llm-inference-router
pip install -r requirements.txt && pip install codecarbon
export HF_TOKEN="your_token"
```

## Report

See `report/yuna_srp_report_v2.tex` for the full research report.
