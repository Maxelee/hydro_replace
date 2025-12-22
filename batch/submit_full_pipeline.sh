#!/bin/bash
# ==============================================================================
# Master Pipeline Script for L205n2500TNG
#
# This script submits all jobs in the correct order with dependencies:
# 1. Generate matches for snap096 (missing)
# 2. Generate all maps for snap096
# 3. Generate BCM maps for snapshots missing them
# 4. Generate lens planes for all models
# 5. Run lux ray-tracing
#
# Usage:
#   cd /mnt/home/mlee1/hydro_replace2
#   bash batch/submit_full_pipeline.sh
#
# Check status with:
#   squeue -u $USER
#   sacct --starttime $(date +%Y-%m-%d) -u $USER
# ==============================================================================

set -e  # Exit on error

cd /mnt/home/mlee1/hydro_replace2

echo "=============================================="
echo "Submitting L205n2500TNG Pipeline"
echo "Date: $(date)"
echo "=============================================="

# --------------------------------------------------------------------------
# Step 1: Generate matches for snap096
# --------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Submitting matches for snap096..."
MATCHES_JOB=$(sbatch --parsable batch/run_matches_snap096.sh)
echo "    Job ID: $MATCHES_JOB"

# --------------------------------------------------------------------------
# Step 2: Generate all maps for snap096 (depends on matches)
# --------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Submitting maps for snap096..."
SNAP96_JOB=$(sbatch --parsable --dependency=afterok:$MATCHES_JOB batch/run_maps_snap096.sh)
echo "    Job ID: $SNAP96_JOB (depends on $MATCHES_JOB)"

# --------------------------------------------------------------------------
# Step 3: Generate BCM maps for other snapshots (can run immediately)
# --------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Submitting BCM maps for 17 snapshots..."
BCM_JOB=$(sbatch --parsable batch/run_maps_bcm_missing.sh)
echo "    Job ID: $BCM_JOB (array job 0-16)"

# --------------------------------------------------------------------------
# Step 4: Generate lens planes (depends on maps)
# --------------------------------------------------------------------------
echo ""
echo ">>> Step 4: Submitting lens planes for L205n2500TNG..."
LP_JOB=$(sbatch --parsable --dependency=afterany:$SNAP96_JOB:$BCM_JOB batch/run_lensplanes_2500.sh)
echo "    Job ID: $LP_JOB (array job 0-9, depends on $SNAP96_JOB and $BCM_JOB)"

# --------------------------------------------------------------------------
# Step 5: Run lux ray-tracing (depends on lens planes)
# --------------------------------------------------------------------------
echo ""
echo ">>> Step 5: Submitting lux ray-tracing..."
LUX_JOB=$(sbatch --parsable --dependency=aftercorr:$LP_JOB batch/run_lux_2500.sh)
echo "    Job ID: $LUX_JOB (array job 0-9, depends on $LP_JOB)"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Pipeline Submitted!"
echo "=============================================="
echo ""
echo "Job Chain:"
echo "  1. Matches snap096:  $MATCHES_JOB"
echo "  2. Maps snap096:     $SNAP96_JOB -> depends on $MATCHES_JOB"
echo "  3. BCM maps:         $BCM_JOB (runs immediately)"
echo "  4. Lens planes:      $LP_JOB -> depends on $SNAP96_JOB, $BCM_JOB"
echo "  5. Lux ray-tracing:  $LUX_JOB -> depends on $LP_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  watch -n 30 'squeue -u $USER'"
echo ""
echo "Check logs in: /mnt/home/mlee1/hydro_replace2/logs/"
echo "=============================================="

# Save job IDs for reference
cat > pipeline_jobs.txt << EOF
# Pipeline submitted: $(date)
MATCHES_JOB=$MATCHES_JOB
SNAP96_JOB=$SNAP96_JOB
BCM_JOB=$BCM_JOB
LP_JOB=$LP_JOB
LUX_JOB=$LUX_JOB
EOF
echo "Job IDs saved to pipeline_jobs.txt"
