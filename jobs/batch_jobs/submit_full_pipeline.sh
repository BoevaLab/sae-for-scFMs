#!/bin/bash
# Advanced: Full pipeline automation with job dependencies
# This script chains train -> generate -> analyze -> steer -> benchmark
# Usage: bash batch_jobs/submit_full_pipeline.sh

# Configuration
LAYERS=(10)  # Start with one layer for testing
DATASETS=("pancreas")  # Start with one dataset
EXPERIMENT_BASE="exp_pipeline"
MODEL="scgpt"

# Parameters for steering
SEEDS=(42)
CLAMP_VALUES=(-2.0 0.0 2.0)
N_FEATURES=(20 50)
SELECTORS=("AMI")
COMPARISON_ID="pipeline_$(date +%Y%m%d_%H%M%S)"

# SLURM configuration
TIME_TRAIN="4:00:00"
TIME_GENERATE="4:00:00"
TIME_ANALYZE="4:00:00"
TIME_STEER="4:00:00"
TIME_BENCHMARK="4:00:00"
MEM="20000"
GPUS="1"
CPUS="1"

echo "========================================"
echo "Full Pipeline Submission"
echo "Comparison ID: ${COMPARISON_ID}"
echo "========================================"
echo ""

# Create temporary file to store experiment:timestamp pairs
TEMP_FILE=$(mktemp)
trap "rm -f ${TEMP_FILE}" EXIT

# Store all job IDs for final benchmark dependency
ALL_STEER_JOBS=()

for dataset in "${DATASETS[@]}"; do
    for layer in "${LAYERS[@]}"; do
        exp_num="${EXPERIMENT_BASE}_${dataset}_L${layer}"
        job_base="${dataset}_L${layer}"
        
        echo "========================================="
        echo "Pipeline for: ${exp_num}"
        echo "========================================="
        
        # PHASE 1: TRAINING
        echo "[1/5] Submitting training job..."
        
        train_job_id=$(sbatch --job-name="train_${job_base}" \
               --time="${TIME_TRAIN}" \
               --mem-per-cpu="${MEM}" \
               --gpus="${GPUS}" \
               --cpus-per-task="${CPUS}" \
               --output="logs/train_${job_base}_%j.out" \
               --error="logs/train_${job_base}_%j.err" \
               --parsable \
               --wrap="python -m scripts.train_sae \
                       scfm=${MODEL} \
                       data=${dataset} \
                       sae.target_layer=${layer} \
                       experiment.number=${exp_num} \
                       experiment.group=pipeline")
        
        echo "  Train job ID: ${train_job_id}"
        
        # PHASE 2: FEATURE GENERATION
        # This needs to extract the timestamp from the training output directory
        # We use a wrapper script that waits for training, finds the timestamp, then generates
        echo "[2/5] Submitting feature generation (depends on train)..."
        
        generate_wrapper=$(cat <<'EOF'
#!/bin/bash
# Wait for training to complete and find the timestamp
exp_num=$1
dataset=$2
layer=$3

# Find the most recent timestamp directory
timestamp=$(ls -t experiments/${exp_num}/ | head -1)

if [ -z "$timestamp" ]; then
    echo "Error: No timestamp found for ${exp_num}"
    exit 1
fi

echo "Found timestamp: ${timestamp}"
echo "Generating features..."

python -m scripts.generate_features \
    sae_checkpoint.experiment=${exp_num} \
    sae_checkpoint.timestamp=${timestamp}

# Save experiment:timestamp to a file for downstream jobs
echo "${exp_num}:${timestamp}" >> /tmp/pipeline_${COMPARISON_ID}_exps.txt
EOF
)
        
        generate_job_id=$(sbatch --job-name="gen_${job_base}" \
               --time="${TIME_GENERATE}" \
               --mem-per-cpu="${MEM}" \
               --gpus="${GPUS}" \
               --cpus-per-task="${CPUS}" \
               --dependency=afterok:${train_job_id} \
               --output="logs/gen_${job_base}_%j.out" \
               --error="logs/gen_${job_base}_%j.err" \
               --parsable \
               --wrap="bash -c '${generate_wrapper}' -- ${exp_num} ${dataset} ${layer}")
        
        echo "  Generate job ID: ${generate_job_id}"
        
        # PHASE 3: ANALYSIS
        echo "[3/5] Submitting analysis (depends on generate)..."
        
        analyze_wrapper=$(cat <<'EOF'
#!/bin/bash
exp_num=$1
timestamp=$(ls -t experiments/${exp_num}/ | head -1)

python -m scripts.analyze_features \
    sae_checkpoint.experiment=${exp_num} \
    sae_checkpoint.timestamp=${timestamp}
EOF
)
        
        analyze_job_id=$(sbatch --job-name="analyze_${job_base}" \
               --time="${TIME_ANALYZE}" \
               --mem-per-cpu="40000" \
               --gpus="${GPUS}" \
               --cpus-per-task="${CPUS}" \
               --dependency=afterok:${generate_job_id} \
               --output="logs/analyze_${job_base}_%j.out" \
               --error="logs/analyze_${job_base}_%j.err" \
               --parsable \
               --wrap="bash -c '${analyze_wrapper}' -- ${exp_num}")
        
        echo "  Analyze job ID: ${analyze_job_id}"
        
        # PHASE 4: STEERING (multiple jobs for parameter sweep)
        echo "[4/5] Submitting steering jobs (depends on analyze)..."
        
        steer_count=0
        for seed in "${SEEDS[@]}"; do
            for clamp in "${CLAMP_VALUES[@]}"; do
                for n_feat in "${N_FEATURES[@]}"; do
                    for selector in "${SELECTORS[@]}"; do
                        if [ "$selector" = "random" ]; then
                            selector_target="sae4scfm.core.steering.RandomFeatureSelector"
                        else
                            selector_target="sae4scfm.core.steering.FileFeatureSelector"
                        fi
                        
                        steer_wrapper=$(cat <<EOF
#!/bin/bash
exp_num=$1
timestamp=\$(ls -t experiments/\${exp_num}/ | head -1)

python -m scripts.steer_features \
    sae_checkpoint.experiment=\${exp_num} \
    sae_checkpoint.timestamp=\${timestamp} \
    steering.seeds=[${seed}] \
    steering.clamp_values=[${clamp}] \
    steering.n_features_list=[${n_feat}] \
    steering.feature_selection._target_=${selector_target} \
    steering.feature_selection.feature_file=experiments/analysis/\${exp_num}/results-\${timestamp}.csv
EOF
)
                        
                        steer_job_id=$(sbatch --job-name="steer_${job_base}_s${seed}_c${clamp}_f${n_feat}" \
                               --time="${TIME_STEER}" \
                               --mem-per-cpu="${MEM}" \
                               --gpus="${GPUS}" \
                               --cpus-per-task="${CPUS}" \
                               --dependency=afterok:${analyze_job_id} \
                               --output="logs/steer_${job_base}_s${seed}_c${clamp}_f${n_feat}_%j.out" \
                               --error="logs/steer_${job_base}_s${seed}_c${clamp}_f${n_feat}_%j.err" \
                               --parsable \
                               --wrap="bash -c '${steer_wrapper}' -- ${exp_num}")
                        
                        ALL_STEER_JOBS+=($steer_job_id)
                        ((steer_count++))
                    done
                done
            done
        done
        
        echo "  Submitted ${steer_count} steering jobs"
        echo ""
        
        sleep 0.5
    done
done

# PHASE 5: BENCHMARK (depends on all steering jobs)
echo ""
echo "========================================="
echo "[5/5] Submitting benchmark job..."
echo "========================================="

# Create dependency string for all steering jobs
if [ ${#ALL_STEER_JOBS[@]} -gt 0 ]; then
    dependency_string="afterok:$(IFS=:; echo "${ALL_STEER_JOBS[*]}")"
    
    benchmark_job_id=$(sbatch --job-name="benchmark_${COMPARISON_ID}" \
           --time="${TIME_BENCHMARK}" \
           --mem-per-cpu="${MEM}" \
           --gpus="${GPUS}" \
           --cpus-per-task="${CPUS}" \
           --dependency="${dependency_string}" \
           --output="logs/benchmark_${COMPARISON_ID}_%j.out" \
           --error="logs/benchmark_${COMPARISON_ID}_%j.err" \
           --parsable \
           --wrap="python -m scripts.benchmark_integration comparison_id=${COMPARISON_ID}")
    
    echo "Benchmark job ID: ${benchmark_job_id}"
else
    echo "No steering jobs submitted, skipping benchmark"
fi

echo ""
echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "Comparison ID: ${COMPARISON_ID}"
echo "Total experiments: $(( ${#DATASETS[@]} * ${#LAYERS[@]} ))"
echo "Total steering jobs per experiment: $(( ${#SEEDS[@]} * ${#CLAMP_VALUES[@]} * ${#N_FEATURES[@]} * ${#SELECTORS[@]} ))"
echo "Total jobs submitted: $(squeue -u $USER | wc -l)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View pipeline: squeue -u \$USER --format='%.18i %.20j %.8T %.10M %.20E'"
echo ""
echo "Results will be in:"
echo "  - Training: experiments/${EXPERIMENT_BASE}_*/"
echo "  - Features: cached_features/${EXPERIMENT_BASE}_*/"
echo "  - Analysis: experiments/analysis/${EXPERIMENT_BASE}_*/"
echo "  - Steering: experiments/steer/${EXPERIMENT_BASE}_*/"
echo "  - Benchmark: experiments/benchmark/${COMPARISON_ID}/"
