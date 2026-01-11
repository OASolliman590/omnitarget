# Scripts

Utility and automation scripts for OmniTarget development and testing.

## Directories

- **testing/** - Test runners and monitoring scripts
- **validation/** - Code validation and verification scripts

## Main Scripts

| Script | Description |
|--------|-------------|
| `analyze_results.py` | Analyze pipeline output results |
| `extract_scenario_data.py` | Extract data from scenario runs |
| `generate_validation_plots.py` | Generate validation visualizations |
| `start_mcp_servers.py` | Start all MCP server processes |

## Testing Scripts

| Script | Description |
|--------|-------------|
| `run_efficient_tests.py` | Main test runner with mode selection |
| `run_benchmark_tests.py` | Performance benchmarks |
| `run_production_tests.py` | Production validation tests |
| `run_mra_validation_tests.py` | MRA-specific validation |

## Usage

```bash
# Run comprehensive test suite
python scripts/testing/run_efficient_tests.py --mode comprehensive

# Start MCP servers
python scripts/start_mcp_servers.py
```
