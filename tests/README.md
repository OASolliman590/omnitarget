# Tests

This project uses pytest with markers to group suites. The curated, comprehensive suite is the recommended default.

## Recommended
```bash
python run_efficient_tests.py --mode comprehensive
```

## Other modes
```bash
python run_efficient_tests.py --mode dev
python run_efficient_tests.py --mode integration
python run_efficient_tests.py --mode performance
```

## Notes
- Legacy and exploratory tests were archived in `archive/legacy_tests/`.
- Benchmark and production suites require real MCP servers and longer runtimes.
