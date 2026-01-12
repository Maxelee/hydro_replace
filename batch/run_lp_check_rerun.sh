#!/bin/bash
#SBATCH --job-name=lp_check_rerun
#SBATCH --output=logs/lp_check_rerun_%j.o
#SBATCH --error=logs/lp_check_rerun_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=cca
#SBATCH --dependency=afterany:2368352

# ============================================================================
# Check and Resubmit Failed LP Recovery Jobs
# ============================================================================
# This job runs after the LP recovery array completes (success or failure).
# It checks which snapshots still have missing lensplanes and resubmits
# a new recovery job with extended time (12 hours) for any failures.
# ============================================================================

module purge
module load python
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "========================================"
echo "Checking LP Recovery Status"
echo "Time: $(date)"
echo "========================================"

# Run the diagnostic to find any remaining missing planes
python3 scripts/find_missing_lensplanes.py > /tmp/lp_check_$$.txt 2>&1

# Check if there are still missing planes
MISSING=$(grep "Total missing:" /tmp/lp_check_$$.txt | awk '{print $3}')

echo "Missing lensplanes: $MISSING"

if [ "$MISSING" -eq 0 ] || [ -z "$MISSING" ]; then
    echo "All lensplanes complete! No resubmission needed."
    echo "Proceeding to LUX conversion..."
    
    # The dependent jobs should already be queued, but let's verify
    echo "Dependent jobs status:"
    squeue -u $USER | grep -E "lux_conv|rt_recov"
    
else
    echo "Found $MISSING missing lensplanes - resubmitting recovery job with 12 hours..."
    
    # Cancel any dependent jobs that would fail
    echo "Cancelling dependent jobs to requeue after recovery..."
    scancel 2368353 2368354 2>/dev/null
    
    # Resubmit with extended time
    NEW_LP_JOB=$(sbatch --parsable batch/run_lensplane_recovery.sh)
    echo "New LP recovery job: $NEW_LP_JOB"
    
    # Resubmit dependent jobs
    NEW_LUX_JOB=$(sbatch --parsable --dependency=afterok:$NEW_LP_JOB batch/run_lux_convert_recovery.sh)
    echo "New LUX conversion job: $NEW_LUX_JOB (depends on $NEW_LP_JOB)"
    
    NEW_RT_JOB=$(sbatch --parsable --dependency=afterok:$NEW_LUX_JOB batch/run_rt_recovery.sh)
    echo "New RT job: $NEW_RT_JOB (depends on $NEW_LUX_JOB)"
fi

cat /tmp/lp_check_$$.txt
rm -f /tmp/lp_check_$$.txt

echo "========================================"
echo "Check complete: $(date)"
echo "========================================"
