#!/usr/bin/env python3
"""
Experiment 2a: Live Per-Model Inference with CodeCarbon
========================================================
Runs each of GreenServ's 15 models on the full 2,500-query GreenServ
benchmark dataset and measures carbon emissions using CodeCarbon's
OfflineEmissionsTracker.

This is Phase 1 of the two-phase carbon attribution approach:
  Phase 1 (this script): measure per-model carbon on 2,500 queries
  Phase 2 (future work): use GreenServ's pregenerated routing decisions
                         to attribute carbon to each routing algorithm

Why two-phase? The live routing experiment (routing_carbon_experiment.py)
encountered model loading overhead that made it infeasible: when a routing
algorithm switches between models, each switch requires loading weights from
disk (30-120 seconds). The two-phase approach avoids this by running each
model separately on all queries.

Hardware: NVIDIA L40S GPU on Laguna HPC (USC/CARC)
Queries: 2,500 from GreenServ's PostgreSQL database
         (MMLU, GSM8K, HellaSwag, WinoGrande, CNN/DailyMail)

Results saved to: live_carbon_results.csv (raw CodeCarbon output)
                  live_carbon_summary.csv (per-model summary)

Prerequisites:
    - PostgreSQL database running with GreenServ's data loaded
    - HF_TOKEN environment variable set
    - HF_HOME pointing to model cache directory

Usage:
    python3 live_inference_carbon.py
"""

import sys
import torch
import pandas as pd
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import OfflineEmissionsTracker
from db.connect import get_connection

sys.path.insert(0, "/project/JehoPark_1895/Yuna_SLM/llm-inference-router")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "/project/JehoPark_1895/Yuna_SLM"  # update for your environment

# GreenServ's 15 models (Yi-34B excluded: requires >92GB VRAM for 2,500 queries)
MODELS = {
    "qwen2.5-0.5b":  "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":    "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b":   "Qwen/Qwen2.5-14B-Instruct",
    "mistral-7b":    "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-3-1b":    "google/gemma-3-1b-it",
    "gemma-3-4b":    "google/gemma-3-4b-it",
    "gemma-3-12b":   "google/gemma-3-12b-it",
    "gemma-3-27b":   "google/gemma-3-27b-it",
    "llama-3.1-8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "phi-4-mini-4b": "microsoft/Phi-4-mini-instruct",
    "phi-4-14b":     "microsoft/phi-4",
}


def load_queries():
    """Load 2,500 benchmark queries from GreenServ's PostgreSQL database."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT query_id, text, dataset FROM queries LIMIT 2500",
        conn
    )
    conn.close()
    logger.info(f"Loaded {len(df)} queries from database")
    return df


def run_live_inference():
    queries_df = load_queries()
    queries = queries_df["text"].tolist()
    all_results = []

    for model_id, model_name in MODELS.items():
        logger.info(f"\nRunning {model_id} on {len(queries)} queries...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            tracker = OfflineEmissionsTracker(
                project_name=model_id,
                country_iso_code="USA",
                save_to_file=True,
                output_file=f"{OUTPUT_DIR}/live_carbon_results.csv"
            )
            tracker.start()

            for i, query in enumerate(queries):
                inputs = tokenizer(
                    query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=256,
                        do_sample=False
                    )
                if i % 100 == 0:
                    logger.info(f"  {model_id}: Query {i}/{len(queries)}")

            emissions_kg = tracker.stop()
            emissions_g = emissions_kg * 1000
            per_query = emissions_g / len(queries)

            all_results.append({
                "model_id":           model_id,
                "total_carbon_g":     emissions_g,
                "carbon_per_query_g": per_query,
                "queries_run":        len(queries)
            })
            logger.info(
                f"  {model_id}: {emissions_g:.4f} gCO2 total, "
                f"{per_query:.6f} gCO2/query"
            )

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"ERROR on {model_id}: {e}")
            all_results.append({
                "model_id":           model_id,
                "total_carbon_g":     None,
                "carbon_per_query_g": None,
                "queries_run":        0
            })

    df = pd.DataFrame(all_results)
    df.to_csv(f"{OUTPUT_DIR}/live_carbon_summary.csv", index=False)
    logger.info("\nDone!")
    logger.info(
        df[["model_id", "total_carbon_g", "carbon_per_query_g"]].to_string()
    )


if __name__ == "__main__":
    run_live_inference()
