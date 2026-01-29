#!/bin/bash
# Batch submission script for training SAEs across multiple layers and datasets
# Usage: bash batch_jobs/submit_train_batch.sh

# Configuration
LAYERS=(9 10 11)  #  different layers
DATASETS=("covid")  #  datasets
SEEDS=(42)  #  different random seeds
EXPERIMENT_BASE="exp_sweep"  # Base experiment name
MODEL="scfoundation"  # Model to use (scgpt, scfoundation, geneformer, scgpt_finetuned_*)
exp_num="covid_sweeps"

# SLURM configuration (adjust as needed)
TIME="24:00:00"
MEM="110000"
GPUS="rtx_3090:1"
CPUS="1"

# Counter for job submissions
job_count=0

# Loop through datasets, layers, and seeds
for dataset in "${DATASETS[@]}"; do
    for layer in "${LAYERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # Create job name
            job_name="train_${dataset}_L${layer}_S${seed}"
            
            echo "Submitting job: ${job_name}"
            echo "  Dataset: ${dataset}"
            echo "  Layer: ${layer}"
            echo "  Seed: ${seed}"
            echo "  Experiment: ${exp_num}"
            
            # Submit the job with Hydra overrides
            sbatch --job-name="${job_name}" \
                   --time="${TIME}" \
                   --mem-per-cpu="${MEM}" \
                   --gpus="${GPUS}" \
                   --cpus-per-task="${CPUS}" \
                   --output="logs/train_${dataset}_L${layer}_S${seed}_%j.out" \
                   --error="logs/train_${dataset}_L${layer}_S${seed}_%j.err" \
                   --wrap="python -m scripts.train_sae \
                           scfm=${MODEL} \
                           data=${dataset} \
                           sae.target_layer=${layer} \
                           seed=${seed} \
                           experiment.number=${exp_num} \
                           experiment.group=layer_sweep"
            
            ((job_count++))
            
            # Add a small delay between submissions to avoid timestamp collisions
            sleep 3m
        done
    done
done

echo ""
echo "=========================================="
echo "Submitted ${job_count} training jobs"
echo "=========================================="

