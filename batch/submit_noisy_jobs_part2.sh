#!/bin/bash
# Submit remaining noisy stats jobs (after initial batch hit QOS limit).
#
# Run this after jobs from submit_noisy_jobs.sh start completing.
# Submits: DES 2.0/3.0 Replace stats + all 6 BCM stats.
#
# Usage:
#   bash batch/submit_noisy_jobs_part2.sh <DMO_RMS_JOB_ID>
#
# Example:
#   bash batch/submit_noisy_jobs_part2.sh 2383863

set -e

# DMO_JOB=${1:?Usage: $0 <DMO_RMS_JOB_ID>}

echo "============================================="
echo "Submitting remaining noisy stats jobs"
# echo "DMO RMS dependency: $DMO_JOB"
echo "============================================="
echo ""

# Remaining Replace model stats (DES 2.0' and 3.0' not yet submitted)
echo "--- Replace model stats (remaining) ---"
for SURVEY in DES; do
    for SMOOTHING in 2.0 3.0; do
        JOB=$(sbatch --parsable \
            --export=ALL,SURVEY=${SURVEY},SMOOTHING=${SMOOTHING} \
            batch/run_compute_noisy_stats.sh)
        echo "  ${SURVEY} θ_G=${SMOOTHING}' → job $JOB (array 0-101)"
    done
done
echo ""

# All BCM model stats
echo "--- BCM model stats ---"
for SURVEY in LSST DES; do
    for SMOOTHING in 1.0 2.0 3.0; do
        JOB=$(sbatch --parsable \
            --export=ALL,SURVEY=${SURVEY},SMOOTHING=${SMOOTHING} \
            batch/run_compute_noisy_bcm_stats.sh)
        echo "  ${SURVEY} θ_G=${SMOOTHING}' → job $JOB"
    done
done
echo ""

echo "============================================="
echo "All remaining jobs submitted!"
echo "============================================="
