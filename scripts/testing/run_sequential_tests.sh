#!/bin/bash
# Sequential Scenario Testing Script
# Purpose: Test scenarios one at a time with delays to prevent MCP server overload
# Created: 2025-10-27

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_DIR="$PROJECT_DIR/logs"
WAIT_TIME=30  # Seconds between tests

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OmniTarget Sequential Testing${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Test counter
TOTAL_TESTS=6
PASSED=0
FAILED=0

# Function to run a single scenario test
run_scenario_test() {
    local scenario_id=$1
    local scenario_name=$2
    local yaml_file=$3
    
    echo -e "${YELLOW}Testing Scenario ${scenario_id}: ${scenario_name}${NC}"
    echo "  YAML: $yaml_file"
    echo "  Log: ${LOG_DIR}/s${scenario_id}_test.log"
    echo ""
    
    # Run test
    cd "$PROJECT_DIR"
    if python -m src.cli yaml "$yaml_file" > "${LOG_DIR}/s${scenario_id}_test.log" 2>&1; then
        echo -e "  ${GREEN}✅ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "  ${RED}❌ FAILED${NC}"
        echo "  Check log: ${LOG_DIR}/s${scenario_id}_test.log"
        ((FAILED++))
    fi
    
    echo ""
}

# Test Scenario 1 (with our fixes)
echo -e "${BLUE}[1/6] Testing S1 Disease Network Construction${NC}"
run_scenario_test 1 "Disease Network" "examples/yaml_configs/scenario_1_only.yaml"

echo -e "${YELLOW}Waiting ${WAIT_TIME} seconds before next test...${NC}"
sleep $WAIT_TIME
echo ""

# For scenarios 2-6, we'll use the comprehensive YAML but this gives S1 time to settle
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}S1 Isolated Test Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Results:"
echo -e "  ${GREEN}Passed: ${PASSED}${NC}"
echo -e "  ${RED}Failed: ${FAILED}${NC}"
echo ""

# Validate S1 results
echo -e "${YELLOW}Validating S1 results...${NC}"
echo ""

python3 << 'PYTHON_SCRIPT'
import json
import sys

print('='*80)
print('S1 VALIDATION REPORT')
print('='*80)

try:
    with open('results/s1_isolated_test.json') as f:
        data = json.load(f)
        s1 = [r for r in data['results'] if r['scenario_id'] == 1][0]
        s1_data = s1['data']
        
        # Validation checks
        checks = []
        
        # Check 1: Reactome genes
        pathways = s1_data.get('pathways', [])
        reactome_pws = [pw for pw in pathways if pw.get('source_db') == 'reactome']
        reactome_genes = sum(len(pw.get('genes', [])) for pw in reactome_pws)
        
        check1 = reactome_genes >= 100
        checks.append(('Reactome genes', reactome_genes, '≥100', check1))
        print(f'\n✅ Reactome genes: {reactome_genes} (target: ≥100) - {"PASS" if check1 else "FAIL"}')
        
        # Check 2: Node pathways
        nodes = s1_data.get('network_nodes', [])
        nodes_with_pathways = sum(1 for n in nodes if len(n.get('pathways', [])) > 0)
        pct = (nodes_with_pathways / len(nodes) * 100) if nodes else 0
        
        check2 = pct >= 80
        checks.append(('Nodes with pathways', f'{pct:.0f}%', '≥80%', check2))
        print(f'✅ Nodes with pathways: {nodes_with_pathways}/{len(nodes)} ({pct:.0f}%) - {"PASS" if check2 else "FAIL"}')
        
        # Check 3: UniProt IDs
        nodes_with_uniprot = sum(1 for n in nodes if n.get('uniprot_id'))
        pct = (nodes_with_uniprot / len(nodes) * 100) if nodes else 0
        
        check3 = pct >= 80
        checks.append(('Nodes with UniProt', f'{pct:.0f}%', '≥80%', check3))
        print(f'✅ Nodes with UniProt: {nodes_with_uniprot}/{len(nodes)} ({pct:.0f}%) - {"PASS" if check3 else "FAIL"}')
        
        # Check 4: Interaction types
        edges = s1_data.get('network_edges', [])
        edges_with_types = sum(1 for e in edges if e.get('interaction_type'))
        pct = (edges_with_types / len(edges) * 100) if edges else 0
        
        check4 = pct >= 90
        checks.append(('Edges with types', f'{pct:.0f}%', '≥90%', check4))
        print(f'✅ Edges with types: {edges_with_types}/{len(edges)} ({pct:.0f}%) - {"PASS" if check4 else "FAIL"}')
        
        # Check 5: Pathway context
        edges_with_context = sum(1 for e in edges if e.get('pathway_context'))
        pct = (edges_with_context / len(edges) * 100) if edges else 0
        
        check5 = pct >= 30
        checks.append(('Edges with context', f'{pct:.0f}%', '≥30%', check5))
        print(f'✅ Edges with context: {edges_with_context}/{len(edges)} ({pct:.0f}%) - {"PASS" if check5 else "FAIL"}')
        
        # Summary
        passed = sum(1 for c in checks if c[3])
        total = len(checks)
        
        print(f'\n{"="*80}')
        print(f'FINAL RESULT: {passed}/{total} checks passed')
        print(f'{"="*80}')
        
        if passed >= 4:
            print('✅ S1 VALIDATION SUCCESSFUL!')
            sys.exit(0)
        else:
            print('❌ S1 VALIDATION FAILED')
            sys.exit(1)
            
except FileNotFoundError:
    print('❌ Results file not found: results/s1_isolated_test.json')
    sys.exit(1)
except Exception as e:
    print(f'❌ Validation error: {e}')
    sys.exit(1)
PYTHON_SCRIPT

VALIDATION_EXIT=$?

echo ""
echo -e "${BLUE}========================================${NC}"
if [ $VALIDATION_EXIT -eq 0 ]; then
    echo -e "${GREEN}S1 Testing Complete - SUCCESSFUL${NC}"
else
    echo -e "${RED}S1 Testing Complete - FAILED${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Logs available in: $LOG_DIR"
echo "Results available in: $RESULTS_DIR"
echo ""

exit $VALIDATION_EXIT

