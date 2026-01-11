#!/bin/bash
# Monitor Phase 2 test progress

LOG_FILE="phase2_full_test.log"

echo "========================================="
echo "Phase 2 Test Progress Monitor"
echo "========================================="
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Get scenario progress
echo "Scenario Progress:"
echo "-----------------"
grep -c "scenario_execution_started" "$LOG_FILE" && echo " scenarios started" || echo "0 scenarios started"
grep -c "scenario_execution_completed" "$LOG_FILE" && echo " scenarios completed" || echo "0 scenarios completed"
echo ""

# Get current scenario
echo "Current Activity:"
echo "-----------------"
tail -5 "$LOG_FILE" | grep -E "INFO|WARNING|ERROR" | sed 's/\x1b\[[0-9;]*m//g'
echo ""

# Count errors and warnings
echo "Issues:"
echo "-------"
echo "Errors: $(grep -c 'ERROR' "$LOG_FILE" || echo 0)"
echo "Warnings: $(grep -c 'WARNING' "$LOG_FILE" || echo 0)"
echo ""

# Check if test is still running
if ps aux | grep -q "[p]ython -m src.cli yaml"; then
    echo "✅ Test is still running..."
else
    echo "⚠️  Test has completed or stopped"
    echo ""
    echo "Final status:"
    tail -20 "$LOG_FILE" | grep -E "scenario_execution_completed|Results saved|YAML batch execution" | sed 's/\x1b\[[0-9;]*m//g'
fi
