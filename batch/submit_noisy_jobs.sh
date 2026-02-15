#!/bin/bash
# Submit all noisy stats jobs for all survey × smoothing combinations.
#
# Usage:
#   bash batch/submit_noisy_jobs.sh
#
# This submits:
#   1. DMO RMS for all 6 configs (array job with 6 tasks)
#   2. Replace model stats for each config (6 × array jobs with 102 tasks)
#   3. BCM model stats for each config (6 × array jobs)
#
# The DMO RMS jobs must complete before stats jobs start (--dependency).

set -e

echo "============================================="
echo "Submitting noisy statistics pipeline"
echo "============================================="
echo ""
echo "Configurations: 2 surveys (LSST, DES) × 3 smoothing scales (1, 2, 3 arcmin)"
echo ""

# Step 1: Submit DMO RMS (all 6 configs as array)
echo "--- Step 1: DMO RMS ---"
DMO_JOB=$(sbatch --parsable batch/run_compute_noisy_dmo_rms.sh)
echo "Submitted DMO RMS job: $DMO_JOB (array 0-5)"
echo ""

# Step 2: Submit Replace model stats (after DMO RMS completes)
echo "--- Step 2: Replace model stats ---"
for SURVEY in LSST DES; do
    for SMOOTHING in 1.0 2.0 3.0; do
        JOB=$(sbatch --parsable \
            --dependency=afterok:${DMO_JOB} \
            --export=ALL,SURVEY=${SURVEY},SMOOTHING=${SMOOTHING} \
            batch/run_compute_noisy_stats.sh)
        echo "  ${SURVEY} θ_G=${SMOOTHING}' → job $JOB (array 0-101)"
    done
done
echo ""

# Step 3: Submit BCM model stats (after DMO RMS completes)
echo "--- Step 3: BCM model stats ---"
for SURVEY in LSST DES; do
    for SMOOTHING in 1.0 2.0 3.0; do
        JOB=$(sbatch --parsable \
            --dependency=afterok:${DMO_JOB} \
            --export=ALL,SURVEY=${SURVEY},SMOOTHING=${SMOOTHING} \
            batch/run_compute_noisy_bcm_stats.sh)
        echo "  ${SURVEY} θ_G=${SMOOTHING}' → job $JOB"
    done
done
echo ""

echo "============================================="
echo "All jobs submitted!"
echo "DMO RMS must finish first (job $DMO_JOB)"
echo "Then 6 Replace + 6 BCM stat jobs will run."
echo ""
echo "Output files will be saved as:"
echo "  {STATS_BASE}/{model}/stats_{SURVEY}_{theta_G}arcmin.h5"
echo "============================================="
