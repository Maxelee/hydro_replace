#!/bin/bash
# =============================================================================
# Auto-submit remaining Lux jobs when queue has room
# =============================================================================
# Run this in a screen/tmux session. It monitors your queue and submits
# the next batch when there's room.
#
# Usage: nohup ./auto_submit_lux.sh &> auto_submit.log &
# =============================================================================

BATCH_SIZE=200
TOTAL_JOBS=2040
MAX_SUBMITTED=450  # CCA QOS limit is 500, keep buffer
SCRIPT="/mnt/home/mlee1/hydro_replace2/batch/run_lux_binned_unified.sh"
CHECK_INTERVAL=600  # Check every 10 minutes

# Track what we've submitted
SUBMITTED_FILE="/mnt/home/mlee1/hydro_replace2/batch/.lux_submitted_max"

# Initialize if not exists
if [ ! -f "$SUBMITTED_FILE" ]; then
    echo "200" > "$SUBMITTED_FILE"  # You already submitted 0-200
fi

while true; do
    SUBMITTED_MAX=$(cat "$SUBMITTED_FILE")
    
    if [ $SUBMITTED_MAX -ge $((TOTAL_JOBS - 1)) ]; then
        echo "$(date): All jobs submitted (up to index $SUBMITTED_MAX). Done!"
        break
    fi
    
    # Count current SUBMITTED jobs (pending + running) for this user
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    
    echo "$(date): Currently $CURRENT_JOBS jobs in queue, submitted up to index $SUBMITTED_MAX"
    
    # Only submit if we have room in the 500 job submission limit
    # Current submitted = CURRENT_JOBS (what's still in queue)
    ROOM=$((MAX_SUBMITTED - CURRENT_JOBS))
    
    if [ $ROOM -ge $BATCH_SIZE ]; then
        # Submit next batch
        START=$((SUBMITTED_MAX + 1))
        END=$((START + BATCH_SIZE - 1))
        if [ $END -ge $TOTAL_JOBS ]; then
            END=$((TOTAL_JOBS - 1))
        fi
        
        echo "$(date): Submitting array $START-$END"
        sbatch --array=${START}-${END} $SCRIPT
        
        # Update tracker
        echo "$END" > "$SUBMITTED_FILE"
        
        # Short pause to let SLURM register
        sleep 10
    else
        echo "$(date): Queue too full ($CURRENT_JOBS jobs). Waiting..."
        sleep $CHECK_INTERVAL
    fi
done
