#!/bin/bash
# =============================================================================
# Chained SLURM Array Job Submission for Lux Pipeline
# =============================================================================
# Submits array jobs in batches of 200, each batch depending on the previous.
# This works around the ~200 concurrent job limit.
#
# Usage: ./submit_lux_chained.sh
# =============================================================================

BATCH_SIZE=200
TOTAL_JOBS=2040
SCRIPT="run_lux_binned_unified.sh"

# Calculate number of batches
N_BATCHES=$(( (TOTAL_JOBS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Total jobs: $TOTAL_JOBS"
echo "Batch size: $BATCH_SIZE"
echo "Number of batches: $N_BATCHES"
echo ""

PREV_JOB_ID=""

for batch in $(seq 0 $((N_BATCHES - 1))); do
    START=$((batch * BATCH_SIZE))
    END=$((START + BATCH_SIZE - 1))
    
    # Cap at total jobs
    if [ $END -ge $TOTAL_JOBS ]; then
        END=$((TOTAL_JOBS - 1))
    fi
    
    echo "Batch $batch: array indices $START-$END"
    
    if [ -z "$PREV_JOB_ID" ]; then
        # First batch - no dependency
        JOB_OUTPUT=$(sbatch --array=${START}-${END} $SCRIPT)
    else
        # Subsequent batches - depend on previous batch completing
        JOB_OUTPUT=$(sbatch --dependency=afterany:${PREV_JOB_ID} --array=${START}-${END} $SCRIPT)
    fi
    
    # Extract job ID from sbatch output ("Submitted batch job 12345")
    PREV_JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')
    echo "  Submitted job $PREV_JOB_ID"
done

echo ""
echo "All batches submitted. Jobs will run in sequence as previous batches complete."
echo "Monitor with: squeue -u \$USER"
