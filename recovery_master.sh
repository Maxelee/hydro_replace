#!/bin/bash
# ============================================================================
# Master Recovery Script for Missing Data Products
# ============================================================================
#
# This script orchestrates the recovery of missing lensplanes, LUX conversions,
# and ray-tracing runs.
#
# WORKFLOW:
# 1. First, diagnose what's missing (run find_missing_lensplanes.py)
# 2. Submit lensplane recovery job (run_lensplane_recovery.sh) - waits for completion
# 3. Submit LUX conversion job (run_lux_convert_recovery.sh) - waits for completion  
# 4. Submit ray-tracing job (run_rt_recovery.sh)
# 5. Verify completeness
#
# USAGE:
#   ./recovery_master.sh diagnose    # Just find what's missing
#   ./recovery_master.sh submit      # Submit all recovery jobs
#   ./recovery_master.sh verify      # Check completeness
#
# ============================================================================

set -e

cd /mnt/home/mlee1/hydro_replace2

case "$1" in
    diagnose)
        echo "=========================================="
        echo "STEP 1: Diagnosing missing data..."
        echo "=========================================="
        source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
        python3 scripts/find_missing_lensplanes.py
        ;;
        
    submit)
        echo "=========================================="
        echo "STEP 2: Submitting recovery jobs..."
        echo "=========================================="
        
        # Check current state first
        echo "Current queue status:"
        squeue -u $USER | head -20
        
        echo ""
        echo "Submitting lensplane recovery job..."
        LP_JOB=$(sbatch --parsable batch/run_lensplane_recovery.sh)
        echo "Lensplane recovery job: $LP_JOB"
        
        echo ""
        echo "Submitting LUX conversion job (depends on $LP_JOB)..."
        LUX_JOB=$(sbatch --parsable --dependency=afterok:$LP_JOB batch/run_lux_convert_recovery.sh)
        echo "LUX conversion job: $LUX_JOB"
        
        echo ""
        echo "Submitting ray-tracing job (depends on $LUX_JOB)..."
        RT_JOB=$(sbatch --parsable --dependency=afterok:$LUX_JOB batch/run_rt_recovery.sh)
        echo "Ray-tracing job: $RT_JOB"
        
        echo ""
        echo "=========================================="
        echo "Jobs submitted with dependencies:"
        echo "  1. Lensplane recovery: $LP_JOB"
        echo "  2. LUX conversion: $LUX_JOB (after $LP_JOB)"
        echo "  3. Ray-tracing: $RT_JOB (after $LUX_JOB)"
        echo "=========================================="
        echo ""
        echo "Monitor with: squeue -u \$USER"
        ;;
        
    verify)
        echo "=========================================="
        echo "STEP 3: Verifying completeness..."
        echo "=========================================="
        source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
        python3 scripts/verify_pipeline_completeness.py
        ;;
        
    *)
        echo "Usage: $0 {diagnose|submit|verify}"
        echo ""
        echo "  diagnose  - Find all missing data products"
        echo "  submit    - Submit recovery jobs with dependencies"
        echo "  verify    - Verify all data products are complete"
        exit 1
        ;;
esac
