#!/bin/bash
# Batch submission script for training SAEs across multiple k values (BatchTopK sparsity)
# Usage: bash batch_jobs/submit_train_k_sweep.sh

# Configuration
K_VALUES=(1.4 1.7 2.1 3.1 5)  # Different k values for BatchTopK
DATASETS=("covid")  # datasets
SEEDS=(42 43 44)  # different random seeds
EXPERIMENT_BASE="exp_k_sweep"  # Base experiment name
MODEL="scgpt"  # Model to use (scgpt, scfoundation, geneformer, scgpt_finetuned_*)
exp_num="sae_archs"
LAYER=10  # Fixed layer to sweep k values on

# SLURM configuration (adjust as needed)
TIME="4:00:00"
MEM="80000"
GPUS="1"
CPUS="1"

# Counter for job submissions
job_count=0

# Loop through datasets, k values, and seeds
for dataset in "${DATASETS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # Create job name
            job_name="train_${dataset}_k${k}_S${seed}"
            
            echo "Submitting job: ${job_name}"
            echo "  Dataset: ${dataset}"
            echo "  k value: ${k}"
            echo "  Layer: ${LAYER}"
            echo "  Seed: ${seed}"
            echo "  Experiment: ${exp_num}"
            
            # Submit the job with Hydra overrides
            sbatch --job-name="${job_name}" \
                   --time="${TIME}" \
                   --mem-per-cpu="${MEM}" \
                   --gpus="${GPUS}" \
                   --cpus-per-task="${CPUS}" \
                   --output="logs/train_${dataset}_k${k}_S${seed}_%j.out" \
                   --error="logs/train_${dataset}_k${k}_S${seed}_%j.err" \
                   --wrap="python -m scripts.train_sae \
                           scfm=${MODEL} \
                           data=${dataset} \
                           sae.target_layer=${LAYER} \
                           sae.hyperparams.l1_penalty=${k} \
                           seed=${seed} \
                           experiment.number=${exp_num}"
            
            ((job_count++))
            
            # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
            sleep 2m
        done
    done
done

echo ""
echo "=========================================="
echo "Submitted ${job_count} training jobs"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
