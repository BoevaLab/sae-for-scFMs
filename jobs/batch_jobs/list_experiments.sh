#!/bin/bash
# Helper script to quickly find experiment timestamps after training
# Usage: bash batch_jobs/list_experiments.sh [experiment_pattern]

PATTERN="${1:-exp*}"

echo "========================================"
echo "Available Experiments"
echo "========================================"
echo ""

for exp_dir in experiments/${PATTERN}/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        echo "Experiment: ${exp_name}"
        
        # List all timestamps for this experiment
        for timestamp_dir in "${exp_dir}"*/; do
            if [ -d "$timestamp_dir" ]; then
                timestamp=$(basename "$timestamp_dir")
                model_path="${timestamp_dir}model.pth"
                
                if [ -f "$model_path" ]; then
                    echo "  ✓ ${timestamp} (model exists)"
                else
                    echo "  ⚠ ${timestamp} (model missing)"
                fi
            fi
        done
        echo ""
    fi
done

echo "========================================"
echo "Usage for batch scripts:"
echo "========================================"
echo ""
echo "Copy-paste format for submit_generate_analyze_batch.sh:"
echo ""

for exp_dir in experiments/${PATTERN}/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        
        for timestamp_dir in "${exp_dir}"*/; do
            if [ -d "$timestamp_dir" ]; then
                timestamp=$(basename "$timestamp_dir")
                model_path="${timestamp_dir}model.pth"
                
                if [ -f "$model_path" ]; then
                    echo "    \"${exp_name}:${timestamp}\""
                fi
            fi
        done
    fi
done

echo ""
