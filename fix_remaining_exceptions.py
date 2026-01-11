#!/usr/bin/env python3
"""
Helper script to systematically fix remaining generic exceptions in OmniTarget.

This script provides:
1. Analysis of remaining exceptions by file
2. Automated patterns for common fixes
3. Prioritization guidance

Usage:
    python fix_remaining_exceptions.py --analyze
    python fix_remaining_exceptions.py --show-file <filepath>
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


def analyze_exceptions():
    """Analyze remaining generic exceptions across the codebase."""
    src_dir = Path("src")
    exception_files = defaultdict(list)
    
    # Find all Python files
    for py_file in src_dir.rglob("*.py"):
        with open(py_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if re.search(r'except\s+Exception', line):
                    # Get context
                    context = lines[max(0, i-3):i+2]
                    exception_files[str(py_file)].append({
                        'line': i,
                        'content': line.strip(),
                        'context': ''.join(context)
                    })
    
    return dict(exception_files)


def prioritize_files(files_dict: Dict) -> List[Tuple[str, List]]:
    """Prioritize files by importance and exception count."""
    priority_patterns = [
        (r'scenarios/scenario_\d+', 3),  # Scenarios - highest priority
        (r'core/(data_standardizer|validation|benchmark)', 2),  # Core modules
        (r'(visualization|utils|core/other)', 1),  # Lower priority
    ]
    
    prioritized = []
    for filepath, exceptions in files_dict.items():
        priority = 0
        for pattern, score in priority_patterns:
            if re.search(pattern, filepath):
                priority = max(priority, score)
        
        prioritized.append((priority, filepath, exceptions))
    
    # Sort by priority (desc) then by count (desc)
    prioritized.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    return prioritized


def show_file_exceptions(filepath: str, files_dict: Dict):
    """Show exceptions in a specific file with suggested fixes."""
    if filepath not in files_dict:
        print(f"File not found or no exceptions: {filepath}")
        return
    
    exceptions = files_dict[filepath]
    print(f"\n{'='*80}")
    print(f"File: {filepath}")
    print(f"Total generic exceptions: {len(exceptions)}")
    print(f"{'='*80}\n")
    
    for i, exc in enumerate(exceptions, 1):
        print(f"\n[{i}] Line {exc['line']}: {exc['content']}")
        print("-" * 80)
        print(exc['context'])
        print("\nSuggested fix pattern:")
        print("```python")
        print("except (DatabaseConnectionError, DatabaseTimeoutError) as e:")
        print("    logger.error('Database error', extra=format_error_for_logging(e))")
        print("    raise")
        print("except Exception as e:")
        print("    logger.error('Unexpected error', extra=format_error_for_logging(e))")
        print("    raise")
        print("```\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze and fix remaining generic exceptions')
    parser.add_argument('--analyze', action='store_true', help='Show prioritized list of files')
    parser.add_argument('--show-file', type=str, help='Show exceptions in specific file')
    
    args = parser.parse_args()
    
    print("Analyzing generic exceptions in OmniTarget codebase...")
    files_dict = analyze_exceptions()
    
    total = sum(len(exc) for exc in files_dict.values())
    print(f"\nTotal generic exceptions found: {total}")
    print(f"Files with exceptions: {len(files_dict)}")
    
    if args.analyze:
        print("\n" + "="*80)
        print("PRIORITIZED FILE LIST")
        print("="*80)
        
        prioritized = prioritize_files(files_dict)
        for priority, filepath, exceptions in prioritized:
            print(f"\n[Priority {priority}] {filepath}")
            print(f"  Exceptions: {len(exceptions)}")
            
            # Show first few lines as preview
            for exc in exceptions[:3]:
                print(f"    Line {exc['line']}: {exc['content'][:60]}...")
    
    if args.show_file:
        show_file_exceptions(args.show_file, files_dict)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Start with high-priority files (scenarios, core modules)
2. Focus on main workflow exceptions, not optional enrichment
3. Use specific exception types from src/core/exceptions.py
4. Add structured logging with format_error_for_logging()
5. Test changes to ensure no regressions

For detailed guidance, see: P0_2_ERROR_HANDLING_PROGRESS.md
""")


if __name__ == '__main__':
    main()
