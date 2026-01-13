#!/bin/bash
# =============================================================================
# Submit all 4 parts of the lux binned pipeline with dependencies
# =============================================================================
#
# This wrapper script submits all 4 parts of the lux ray-tracing pipeline
# with SLURM dependencies so they run sequentially (avoiding scheduler overload).
#
# Usage:
#   ./submit_lux_pipeline.sh                  # Submit all 4 parts
#   ./submit_lux_pipeline.sh --dry-run        # Print commands without submitting
#   ./submit_lux_pipeline.sh --part 2         # Submit only part 2 onward
#
# =============================================================================

set -e

cd /mnt/home/mlee1/hydro_replace2/batch

DRY_RUN=false
START_PART=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --part)
            START_PART=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--part N]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Lux Binned Pipeline Submission"
echo "=============================================="
echo "Starting from part: $START_PART"
echo "Dry run: $DRY_RUN"
echo ""

# Function to submit a job
submit_job() {
    local script=$1
    local dep=$2
    
    if [ "$DRY_RUN" = true ]; then
        if [ -z "$dep" ]; then
            echo "[DRY RUN] sbatch $script"
        else
            echo "[DRY RUN] sbatch --dependency=afterany:$dep $script"
        fi
        echo "0"  # Return fake job ID
    else
        if [ -z "$dep" ]; then
            JOB_ID=$(sbatch "$script" | awk '{print $NF}')
        else
            JOB_ID=$(sbatch --dependency=afterany:$dep "$script" | awk '{print $NF}')
        fi
        echo "$JOB_ID"
    fi
}

# Submit jobs
PREV_JOB=""

if [ $START_PART -le 1 ]; then
    echo "Submitting Part 1 (jobs 0-509)..."
    JOB1=$(submit_job "run_lux_binned_part1.sh" "")
    echo "  Job ID: $JOB1"
    PREV_JOB=$JOB1
fi

if [ $START_PART -le 2 ]; then
    echo "Submitting Part 2 (jobs 510-1019)..."
    if [ $START_PART -eq 2 ]; then
        JOB2=$(submit_job "run_lux_binned_part2.sh" "")
    else
        JOB2=$(submit_job "run_lux_binned_part2.sh" "$PREV_JOB")
    fi
    echo "  Job ID: $JOB2"
    PREV_JOB=$JOB2
fi

if [ $START_PART -le 3 ]; then
    echo "Submitting Part 3 (jobs 1020-1529)..."
    if [ $START_PART -eq 3 ]; then
        JOB3=$(submit_job "run_lux_binned_part3.sh" "")
    else
        JOB3=$(submit_job "run_lux_binned_part3.sh" "$PREV_JOB")
    fi
    echo "  Job ID: $JOB3"
    PREV_JOB=$JOB3
fi

if [ $START_PART -le 4 ]; then
    echo "Submitting Part 4 (jobs 1530-2039)..."
    if [ $START_PART -eq 4 ]; then
        JOB4=$(submit_job "run_lux_binned_part4.sh" "")
    else
        JOB4=$(submit_job "run_lux_binned_part4.sh" "$PREV_JOB")
    fi
    echo "  Job ID: $JOB4"
fi

echo ""
echo "=============================================="
echo "Submission complete!"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check completion status with:"
echo "  python scripts/check_progress.py"
echo "=============================================="
