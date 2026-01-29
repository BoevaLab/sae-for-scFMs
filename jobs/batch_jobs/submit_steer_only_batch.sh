#!/bin/bash
# Batch submission script for steering only (no benchmarking)
# Usage: bash batch_jobs/submit_steer_only_batch.sh
# Use this when analysis completed but steering failed or needs to be re-run

# Configuration: List of experiments to process
# Format: "experiment_number:timestamp"
EXPERIMENTS=(
    "exp02:Jan12-15-49"
    "exp02:Jan12-15-51"
    "exp02:Jan12-15-53"
    "exp02:Jan12-15-55"
    "exp02:Jan12-15-57"
    "exp02:Jan12-15-59"
    "exp02:Jan12-16-01"
    "exp02:Jan12-16-03"
    "exp02:Jan12-16-05"
    "exp02:Jan12-16-07"
    "exp02:Jan12-16-09"
    "exp02:Jan12-16-11"
)

# Parameter sweeps
SEEDS=(42)  # Random seeds for reproducibility
CLAMP_VALUES=(-2.0)  # Clamp values to test
N_FEATURES=(20)  # Number of features to steer
SELECTORS=("AMI")  # Feature selection strategies (AMI, random, etc.)

# SLURM configuration
TIME_STEER="4:00:00"
MEM_STEER="20000"
GPUS="1"
CPUS="1"

# Counter
steer_count=0

echo "========================================"
echo "Batch Steering (Steering Only)"
echo "========================================"
echo ""

for exp_entry in "${EXPERIMENTS[@]}"; do
    # Parse experiment and timestamp
    IFS=':' read -r exp_num timestamp <<< "$exp_entry"
    
    echo "Experiment: ${exp_num} (${timestamp})"
    
    # Check if analysis results exist (needed for feature selection)
    analysis_file="experiments/analysis/${exp_num}/results-${timestamp}.csv"
    if [ ! -f "$analysis_file" ]; then
        echo "  ⚠ WARNING: Analysis file not found: ${analysis_file}"
        echo "  Please run analyze_features first!"
        echo "  Skipping..."
        echo ""
        continue
    fi
    
    # Loop through all parameter combinations for steering
    for seed in "${SEEDS[@]}"; do
        for clamp in "${CLAMP_VALUES[@]}"; do
            for n_feat in "${N_FEATURES[@]}"; do
                for selector in "${SELECTORS[@]}"; do
                    # Create job name
                    job_name="steer_${exp_num}_s${seed}_c${clamp}_f${n_feat}_${selector}"
                    
                    # Determine selector target based on type
                    if [ "$selector" = "random" ]; then
                        selector_target="sae4scfm.core.steering.RandomFeatureSelector"
                    else
                        selector_target="sae4scfm.core.steering.FileFeatureSelector"
                    fi
                    
                    echo "  Submitting: seed=${seed}, clamp=${clamp}, n_features=${n_feat}, selector=${selector}"
                    
                    # Submit steering job
                    sbatch --job-name="${job_name}" \
                           --time="${TIME_STEER}" \
                           --mem-per-cpu="${MEM_STEER}" \
                           --gpus="${GPUS}" \
                           --cpus-per-task="${CPUS}" \
                           --output="logs/steer_${exp_num}_s${seed}_c${clamp}_f${n_feat}_${selector}_%j.out" \
                           --error="logs/steer_${exp_num}_s${seed}_c${clamp}_f${n_feat}_${selector}_%j.err" \
                           --wrap="python -m scripts.steer_features \
                                   sae_checkpoint.experiment=${exp_num} \
                                   sae_checkpoint.timestamp=${timestamp} \
                                   steering.seeds=[${seed}] \
                                   steering.clamp_values=[${clamp}] \
                                   steering.n_features_list=[${n_feat}] \
                                   steering.feature_selection._target_=${selector_target} \
                                   steering.feature_selection.feature_file=experiments/analysis/${exp_num}/results-${timestamp}.csv"
                    
                    ((steer_count++))
                    
                    # Small delay
                    sleep 0.2
                done
            done
        done
    done
    
    echo "  ✓ Submitted steering jobs for ${exp_num}"
    echo ""
done

echo ""
echo "=========================================="
echo "Submitted ${steer_count} steering jobs"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo ""
echo "Steering results will be in: experiments/steer/{experiment}/{timestamp}/"
echo ""
echo "To run benchmarking afterwards, use submit_benchmark_only_batch.sh"
