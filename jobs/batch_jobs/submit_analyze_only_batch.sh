#!/bin/bash
# Batch submission script for analyzing already-generated features
# Usage: bash batch_jobs/submit_analyze_only_batch.sh
# Use this when generation completed but analysis failed

# Configuration: List of experiments and timestamps to process
# Format: "experiment_number:timestamp"
EXPERIMENTS=(
    "covid_sweeps:Jan17-13-28"
    "covid_sweeps:Jan17-13-29"
    "covid_sweeps:Jan17-13-33"
    "covid_sweeps:Jan17-13-37"
    "covid_sweeps:Jan17-13-39"
    "covid_sweeps:Jan17-13-41"
    "covid_sweeps:Jan17-13-45"
    "covid_sweeps:Jan17-18-11"
    "covid_sweeps:Jan21-17-33"
    "covid_sweeps:Jan21-17-35"
    "covid_sweeps:Jan21-17-38"
    "covid_sweeps:Jan18-00-37"
)

# SLURM configuration
TIME_ANALYZE="4:00:00"
MEM_ANALYZE="120000"
CPUS="1"

# Counter
analyze_count=0

echo "========================================"
echo "Batch Feature Analysis (Analysis Only)"
echo "========================================"
echo ""

for exp_entry in "${EXPERIMENTS[@]}"; do
    # Parse experiment and timestamp
    IFS=':' read -r exp_num timestamp <<< "$exp_entry"
    
    echo "Processing experiment: ${exp_num} (${timestamp})"

    
    # ===== SUBMIT ANALYZE JOB =====
    analyze_job_name="analyze_${exp_num}"
    
    echo "  Submitting analysis job..."
    
    sbatch --job-name="${analyze_job_name}" \
           --time="${TIME_ANALYZE}" \
           --mem-per-cpu="${MEM_ANALYZE}" \
           --cpus-per-task="${CPUS}" \
           --output="logs/analyze_${exp_num}_%j.out" \
           --error="logs/analyze_${exp_num}_%j.err" \
           --wrap="python -m scripts.analyze_features \
                   sae_checkpoint.experiment=${exp_num} \
                   sae_checkpoint.timestamp=${timestamp}\
                   +data=covid_evaluation"
    
    ((analyze_count++))
    
    echo "  ✓ Submitted analysis for ${exp_num}"
    echo ""
    
    # Small delay between submissions
    sleep 0.5
done

echo ""
echo "=========================================="
echo "Submitted ${analyze_count} analyze jobs"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo ""
echo "Analysis results will be in: experiments/analysis/{experiment}/"
