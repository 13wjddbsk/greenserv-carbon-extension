#!/usr/bin/env python3
"""
Experiment 2b: Live Routing Experiment with CodeCarbon
=======================================================
Runs all 8 GreenServ routing strategies live on 2,500 benchmark queries
using GreenServ's full feature extraction pipeline, with CodeCarbon
measuring carbon emissions per algorithm.

Goal: Produce an Accuracy vs. Carbon Emissions Pareto plot comparable to
GreenServ's existing Accuracy vs. Energy (Wh) Pareto plot.

8 routing strategies:
  MAB algorithms: LinUCB (C), Thompson Sampling (C), e-Greedy (C), e-Greedy (NC)
  Baselines:      Random, Largest, Smallest, Accuracy

KNOWN LIMITATION: This experiment did not complete within the project
timeframe due to model loading overhead. When a routing algorithm switches
between models, each switch requires loading weights from disk (30-120 sec).
With 15 models and 2,500 queries, this makes the experiment infeasible on
hardware with <150GB VRAM.

Recommended alternative: Two-phase carbon attribution
  Phase 1: Run live_inference_carbon.py to get per-model carbon per query
  Phase 2: Use GreenServ's pregenerated routing decisions to attribute
           carbon to each algorithm post-hoc

Errors encountered during development:
  1. TypeError: LinUCB missing model_ids, context_dimension args
     Fix: pass model_ids and context_dimension to all bandit constructors
  2. AttributeError: 'LinUCB' has no attribute 'select'
     Fix: use algo.select_model(context), not algo.select(context)
  3. OSError: gated repo aya-expanse-32b
     Fix: models.yaml has extra models; hardcode GREENSERV_MODEL_IDS
  4. CUDA OutOfMemoryError between algorithms
     Fix: del model, torch.cuda.empty_cache(), gc.collect() after each algo
  5. RuntimeError: layer not mapped to device
     Fix: device_map="balanced", use_cache=False
  6. sbatch: node config unavailable
     Fix: --gres=gpu:8 → --gres=gpu:2 (max 2 L40S per node)

Hardware: 2x NVIDIA L40S (92GB total VRAM) on Laguna HPC
"""

import sys
import gc
import logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker

sys.path.insert(0, "/project/JehoPark_1895/Yuna_SLM/llm-inference-router")

from db.connect import get_connection
from src.services.feature_service import FeatureService
from src.bandit.linucb import LinUCB
from src.bandit.epsilon_greedy import EpsilonGreedy, LinearEpsilonGreedy
from src.bandit.thompson_sampling import ThompsonSampling
from experiments.shared.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "/project/JehoPark_1895/Yuna_SLM"
LAMBDA = 0.5  # accuracy-energy tradeoff, matching GreenServ default

# ── Configuration ──────────────────────────────────────────────────────────────

# Hardcoded to GreenServ's 16 models only
# (models.yaml contains extra models not used in the paper)
GREENSERV_MODEL_IDS = [
    "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b",
    "mistral-7b",
    "gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b",
    "llama-3.1-8b", "llama-3.2-1b", "llama-3.2-3b",
    "phi-4-mini-4b", "phi-4-14b",
    # "yi-34b"  # excluded: 34B too large for 2x L40S with routing overhead
]

configs = {
    "models":           load_config("models"),
    "feature_extraction": load_config("feature_extraction"),
    "experiments":      load_config("experiments"),
    "baselines":        load_config("baselines"),
}

ALL_MODEL_CONFIGS = configs["models"]["models"]
MODEL_IDS = [m for m in GREENSERV_MODEL_IDS if m in ALL_MODEL_CONFIGS]
MODEL_CONFIGS = {k: ALL_MODEL_CONFIGS[k] for k in MODEL_IDS}
BASELINES = configs["baselines"]

feature_config = configs["feature_extraction"]
feature_service = FeatureService(feature_config)

# ── Algorithm definitions ──────────────────────────────────────────────────────
# Note: all GreenServ bandit classes require model_ids and context_dimension
# as the first two positional arguments

def make_algorithms(model_ids, context_dim):
    return {
        "linucb": LinUCB(
            model_ids=model_ids,
            context_dimension=context_dim,
            alpha=0.1,
            regularization=0.05
        ),
        "epsilon_greedy_nc": EpsilonGreedy(
            model_ids=model_ids,
            context_dimension=context_dim,
            initial_epsilon=1.0,
            decay_factor=0.98,
            min_epsilon=0.01
        ),
        "epsilon_greedy_c": LinearEpsilonGreedy(
            model_ids=model_ids,
            context_dimension=context_dim,
            initial_epsilon=1.0,
            decay_factor=0.985,
            min_epsilon=0.01,
            lambda_=0.25
        ),
        "thompson_sampling": ThompsonSampling(
            model_ids=model_ids,
            context_dimension=context_dim,
            sigma=0.25,
            prior_variance=2
        ),
    }

BASELINES_FIXED = {
    "random":   None,
    "largest":  BASELINES.get("largest_model_id", "yi-34b"),
    "smallest": BASELINES.get("smallest_model_id", "qwen2.5-0.5b"),
    "accuracy": BASELINES.get("accuracy_model_id", "gemma-3-27b"),
}

# ── Helper functions ───────────────────────────────────────────────────────────

def load_queries():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT query_id, text, dataset FROM queries LIMIT 2500", conn
    )
    conn.close()
    logger.info(f"Loaded {len(df)} queries from database")
    return df


def load_pregenerated_rewards():
    """Load accuracy and energy per (query_id, model_id) from GreenServ's DB."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT query_id, model_id, accuracy, energy_consumption "
        "FROM pregenerated_results",
        conn
    )
    conn.close()
    reward_lookup = {}
    for _, row in df.iterrows():
        reward_lookup[(row["query_id"], row["model_id"])] = {
            "accuracy": row["accuracy"],
            "energy":   row["energy_consumption"]
        }
    logger.info(f"Loaded {len(reward_lookup)} pregenerated rewards")
    return reward_lookup


def load_model(model_id):
    model_name = MODEL_CONFIGS[model_id]["name"]
    logger.info(f"  Loading {model_id} ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",      # distribute evenly across both L40S GPUs
        attn_implementation="eager"
    )
    # Warm up as GreenServ does to avoid lazy initialization skewing latency
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=False,
            use_cache=False  # required for multi-GPU layer mapping
        )
    logger.info(f"  {model_id} loaded and warmed up")
    return model, tokenizer


def run_inference(model, tokenizer, query):
    inputs = tokenizer(
        query, return_tensors="pt",
        truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=False,
            use_cache=False
        )
    return tokenizer.decode(
        outputs[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )


def clear_gpu_memory(loaded_models):
    """Free all loaded models from GPU memory between algorithm runs."""
    for model_id, (model, tokenizer) in list(loaded_models.items()):
        del model
        del tokenizer
    loaded_models.clear()
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("  GPU memory cleared")


# ── Main experiment function ───────────────────────────────────────────────────

def run_algorithm(algo_name, algo, queries_df, reward_lookup):
    logger.info(f"\n=== Running algorithm: {algo_name} ===")
    loaded_models = {}
    total_reward = 0
    step_results = []

    # CodeCarbon measures total GPU+CPU+RAM energy for this algorithm's run.
    # Carbon is implicitly weighted by model selection frequency: algorithms
    # that route to large models consume more GPU power, captured by CodeCarbon.
    tracker = EmissionsTracker(
        project_name=algo_name,
        log_level="error",
        save_to_file=True,
        output_file=f"{OUTPUT_DIR}/routing_carbon_results.csv"
    )
    tracker.start()

    for i, row in enumerate(queries_df.itertuples()):
        query_text = row.text
        query_metadata = {"dataset": row.dataset}

        # Extract features using GreenServ's full pipeline (task type,
        # semantic cluster, text complexity) + intercept term
        features = feature_service.extract_features(query_text, query_metadata)
        context = np.array(features["context_vector"] + [1.0])

        # Select model based on algorithm type
        if algo_name == "random":
            chosen_model_id = np.random.choice(MODEL_IDS)
        elif algo_name in BASELINES_FIXED:
            chosen_model_id = BASELINES_FIXED[algo_name]
        else:
            # MAB: select_model() returns model_id string directly
            chosen_model_id = algo.select_model(context)

        # Load model if not already cached
        if chosen_model_id not in loaded_models:
            loaded_models[chosen_model_id] = load_model(chosen_model_id)

        model, tokenizer = loaded_models[chosen_model_id]
        _ = run_inference(model, tokenizer, query_text)

        # Reward from pregenerated results (avoids re-evaluating accuracy)
        reward_data = reward_lookup.get(
            (row.query_id, chosen_model_id),
            {"accuracy": 0.5, "energy": 0.5}
        )
        accuracy = reward_data["accuracy"]
        energy_norm = reward_data["energy"]
        reward = (1 - LAMBDA) * accuracy - LAMBDA * energy_norm

        # Update MAB with observed reward
        if algo is not None and algo_name not in BASELINES_FIXED:
            algo.update(chosen_model_id, reward, context)

        total_reward += reward
        step_results.append({
            "algorithm":    algo_name,
            "query_id":     row.query_id,
            "chosen_model": chosen_model_id,
            "accuracy":     accuracy,
            "reward":       reward
        })

        if i % 100 == 0:
            logger.info(
                f"  {algo_name}: Query {i}/{len(queries_df)}, "
                f"model={chosen_model_id}"
            )

    emissions_kg = tracker.stop()
    emissions_g = emissions_kg * 1000
    mean_accuracy = np.mean([r["accuracy"] for r in step_results])

    logger.info(
        f"  {algo_name}: {emissions_g:.4f} gCO2, "
        f"mean_accuracy={mean_accuracy:.4f}"
    )

    # Clear GPU memory before next algorithm
    clear_gpu_memory(loaded_models)

    return {
        "algorithm":        algo_name,
        "total_carbon_g":   emissions_g,
        "carbon_per_query_g": emissions_g / len(queries_df),
        "mean_accuracy":    mean_accuracy,
        "total_reward":     total_reward,
        "queries_run":      len(queries_df)
    }


def main():
    queries_df = load_queries()
    reward_lookup = load_pregenerated_rewards()

    context_dim = (
        len(feature_service.task_types) +
        feature_service.num_clusters +
        feature_service.num_complexity_bins +
        1  # intercept
    )
    logger.info(f"Context dimension: {context_dim}")

    algorithms = make_algorithms(MODEL_IDS, context_dim)
    all_results = []

    # Run MAB algorithms
    for algo_name, algo in algorithms.items():
        result = run_algorithm(algo_name, algo, queries_df, reward_lookup)
        all_results.append(result)

    # Run baselines
    for baseline_name in BASELINES_FIXED.keys():
        result = run_algorithm(baseline_name, None, queries_df, reward_lookup)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(f"{OUTPUT_DIR}/routing_carbon_summary.csv", index=False)
    logger.info("\n=== All done! ===")
    logger.info(
        df[["algorithm", "total_carbon_g", "mean_accuracy"]].to_string()
    )


if __name__ == "__main__":
    main()
