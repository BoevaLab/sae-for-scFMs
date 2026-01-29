#!/bin/bash
# Batch submission script for steering and benchmarking with parameter sweeps
# Usage: bash batch_jobs/submit_steer_benchmark_batch.sh

# Configuration: List of experiments to process
# Format: "experiment_number:timestamp"
EXPERIMENTS=(
    "exp03:Jan13-16-43"
    "exp03:Jan13-16-45"
    "exp03:Jan13-16-47"
    "exp03:Jan13-16-49"
    "exp03:Jan13-16-51"
    "exp03:Jan13-16-53"
    "exp03:Jan13-16-55"
    "exp03:Jan13-16-57"
    "exp03:Jan13-16-59"
    "exp03:Jan13-17-01"
)

# Parameter sweeps
SEEDS=(42)  # Random seeds for reproducibility
CLAMP_VALUES=(-2.0)  # Clamp values to test
N_FEATURES=(20)  # Number of features to steer
SELECTORS=("AMI")  # Feature selection strategies (AMI, random, etc.)

# SLURM configuration
TIME_STEER="4:00:00"
TIME_BENCHMARK="4:00:00"
MEM_STEER="20000"
MEM_BENCHMARK="20000"
GPUS="1"
CPUS="1"

# Benchmark comparison ID (increment for different runs)
COMPARISON_ID="sweep_k"

# Counters
steer_count=0
benchmark_count=0

echo "========================================"
echo "Batch Steering & Benchmarking"
echo "========================================"
echo ""

# First, submit all steering jobs
echo "PHASE 1: Submitting steering jobs..."
echo ""

for exp_entry in "${EXPERIMENTS[@]}"; do
    # Parse experiment and timestamp
    IFS=':' read -r exp_num timestamp <<< "$exp_entry"
    
    echo "Experiment: ${exp_num} (${timestamp})"
    
    # Check if analysis results exist (needed for feature selection)
    analysis_file="experiments/analysis/${exp_num}/results-${timestamp}.csv"
    if [ ! -f "$analysis_file" ]; then
        echo "  ⚠ Warning: Analysis file not found: ${analysis_file}"
        echo "  Please run analyze_features first!"
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
echo "----------------------------------------"
echo "Submitted ${steer_count} steering jobs"
echo "----------------------------------------"
echo ""

# Wait a moment before submitting benchmark
sleep 2

# Now submit benchmark job (depends on all steering jobs)
echo "PHASE 2: Submitting benchmark job..."
echo ""

# Create a custom benchmark config file for this sweep
benchmark_config_file="config/benchmark_${COMPARISON_ID}.yaml"

cat > "${benchmark_config_file}" << EOF
defaults:
  - benchmark

comparison_id: ${COMPARISON_ID}

experiments:
EOF

# Build the benchmark config by creating entries for each experiment
for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_num timestamp <<< "$exp_entry"
    
    # Add experiment entry to config file
    cat >> "${benchmark_config_file}" << EOF
  - number: "${exp_num}"
    date: "${timestamp}"
    seeds: [${SEEDS[*]}]
    clamps: [${CLAMP_VALUES[*]}]
    n_features: [${N_FEATURES[*]}]
    selector: [${SELECTORS[*]}]
    original: true
EOF
done

echo "Created benchmark config: ${benchmark_config_file}"

# Submit benchmark job (will wait for all steering jobs to complete)
benchmark_job_name="benchmark_${COMPARISON_ID}"

echo "Submitting benchmark comparison (ID: ${COMPARISON_ID})..."

sbatch --job-name="${benchmark_job_name}" \
       --time="${TIME_BENCHMARK}" \
       --mem-per-cpu="${MEM_BENCHMARK}" \
       --gpus="${GPUS}" \
       --cpus-per-task="${CPUS}" \
       --output="logs/benchmark_${COMPARISON_ID}_%j.out" \
       --error="logs/benchmark_${COMPARISON_ID}_%j.err" \
       --wrap="python -m scripts.benchmark_integration \
               --config-name=benchmark_${COMPARISON_ID}"

((benchmark_count++))

echo "  ✓ Submitted benchmark job"
echo ""

echo ""
echo "=========================================="
echo "Summary:"
echo "  - ${steer_count} steering jobs submitted"
echo "  - ${benchmark_count} benchmark job submitted"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo ""
echo "Steering results will be in: experiments/steer/{experiment}/{timestamp}/"
echo "Benchmark results will be in: experiments/benchmark/${COMPARISON_ID}/"
echo ""
echo "Note: Benchmark job will wait for steering jobs to complete."
echo "      Adjust --dependency in the script for explicit control."
