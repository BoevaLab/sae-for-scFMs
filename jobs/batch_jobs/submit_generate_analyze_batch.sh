#!/bin/bash
# Batch submission script for generating features and analyzing them
# Usage: bash batch_jobs/submit_generate_analyze_batch.sh

# Configuration: List of experiments and timestamps to process
# Format: "experiment_number:timestamp"
EXPERIMENTS=(
    "covid_sweeps:Jan18-20-40"
    "covid_sweeps:Jan18-20-43"
    "covid_sweeps:Jan18-20-46"
    "covid_sweeps:Jan18-20-49"
    "covid_sweeps:Jan18-20-52"
    "covid_sweeps:Jan18-21-21"
    "covid_sweeps:Jan18-21-23"
    "covid_sweeps:Jan18-21-29"
    "covid_sweeps:Jan18-21-36"
    "covid_sweeps:Jan21-19-04"
    "covid_sweeps:Jan21-19-07"
    "covid_sweeps:Jan21-19-10"
)

# SLURM configuration
TIME_GENERATE="4:00:00"
TIME_ANALYZE="4:00:00"
MEM_GENERATE="50000"
MEM_ANALYZE="80000"
GPUS="rtx_3090:1"
CPUS="1"

# Counters
generate_count=0
analyze_count=0

echo "========================================"
echo "Batch Feature Generation & Analysis"
echo "========================================"
echo ""

for exp_entry in "${EXPERIMENTS[@]}"; do
    # Parse experiment and timestamp
    IFS=':' read -r exp_num timestamp <<< "$exp_entry"
    
    echo "Processing experiment: ${exp_num} (${timestamp})"
    
    # ===== SUBMIT GENERATE JOB =====
    generate_job_name="gen_${exp_num}"
    
    echo "  [1/2] Submitting feature generation..."
    
    generate_job_id=$(sbatch --job-name="${generate_job_name}" \
           --time="${TIME_GENERATE}" \
           --mem-per-cpu="${MEM_GENERATE}" \
           --gpus="${GPUS}" \
           --cpus-per-task="${CPUS}" \
           --output="logs/generate_${exp_num}_%j.out" \
           --error="logs/generate_${exp_num}_%j.err" \
           --parsable \
           --wrap="python -m scripts.generate_features \
                   sae_checkpoint.experiment=${exp_num} \
                   sae_checkpoint.timestamp=${timestamp} \
                   +data=covid_evaluation")
                   
    
    ((generate_count++))
    
    # ===== SUBMIT ANALYZE JOB (DEPENDS ON GENERATE) =====
    analyze_job_name="analyze_${exp_num}"
    
    echo "  [2/2] Submitting analysis (depends on job ${generate_job_id})..."
    
    sbatch --job-name="${analyze_job_name}" \
           --time="${TIME_ANALYZE}" \
           --mem-per-cpu="${MEM_ANALYZE}" \
           --cpus-per-task="${CPUS}" \
           --dependency=afterok:${generate_job_id} \
           --output="logs/analyze_${exp_num}_%j.out" \
           --error="logs/analyze_${exp_num}_%j.err" \
           --wrap="python -m scripts.analyze_features \
                   sae_checkpoint.experiment=${exp_num} \
                   sae_checkpoint.timestamp=${timestamp} \
                   +data=covid_evaluation"
    
    ((analyze_count++))
    
    echo "  ✓ Submitted generate + analyze pipeline for ${exp_num}"
    echo ""
    
    # Small delay between submissions
    sleep 0.5
done

echo ""
echo "=========================================="
echo "Submitted ${generate_count} generate jobs"
echo "Submitted ${analyze_count} analyze jobs (with dependencies)"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo ""
echo "Generated features will be in: cached_features/{experiment}/"
echo "Analysis results will be in: experiments/analysis/{experiment}/"
