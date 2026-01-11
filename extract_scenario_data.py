"""
Extract scenario data from JSON results and create markdown files for analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pprint

def format_value(value: Any, indent: int = 0) -> str:
    """Format a value for markdown output."""
    indent_str = "  " * indent

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        for k, v in value.items():
            formatted_v = format_value(v, indent + 1)
            lines.append(f"{indent_str}  {k}: {formatted_v}")
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)
    elif isinstance(value, list):
        if not value:
            return "[]"
        if len(value) <= 3 and all(isinstance(x, (str, int, float, bool)) for x in value):
            return str(value)
        return f"[{len(value)} items]"
    elif isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    else:
        return str(type(value))

def get_data_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary statistics of the data."""
    summary = {}

    for key, value in data.items():
        if isinstance(value, list):
            summary[key] = {
                'type': 'list',
                'count': len(value),
                'sample': value[:3] if value else []
            }
        elif isinstance(value, dict):
            summary[key] = {
                'type': 'dict',
                'keys': list(value.keys())[:10]
            }
        else:
            summary[key] = {
                'type': type(value).__name__,
                'value': value
            }

    return summary

def create_scenario_markdown(scenario_data: Dict[str, Any], output_dir: Path):
    """Create markdown file for a scenario."""
    scenario_id = scenario_data['scenario_id']
    scenario_name = scenario_data['scenario_name']
    status = scenario_data['status']
    data = scenario_data['data']

    # Create markdown content
    md_content = f"""# Scenario {scenario_id}: {scenario_name}

**Status:** {status}
**Execution Time:** {scenario_data.get('execution_time', 'N/A')}

---

## Overview

This document contains the detailed analysis of **{scenario_name}** results from the AXL breast cancer analysis pipeline.

---

## Data Structure Summary

"""

    # Add data structure
    data_keys = list(data.keys())
    md_content += f"The scenario data contains **{len(data_keys)}** main sections:\n\n"

    for i, key in enumerate(data_keys, 1):
        value = data[key]
        if isinstance(value, list):
            md_content += f"{i}. **{key}**: {len(value)} items\n"
        elif isinstance(value, dict):
            md_content += f"{i}. **{key}**: Dictionary with {len(value)} keys\n"
        else:
            md_content += f"{i}. **{key}**: {type(value).__name__}\n"

    md_content += "\n---\n\n"

    # Add detailed sections for each key
    for key in data_keys:
        value = data[key]
        md_content += f"## {key.replace('_', ' ').title()}\n\n"

        if isinstance(value, list):
            md_content += f"**Type:** List with {len(value)} items\n\n"
            if value:
                # Show structure of first item
                first_item = value[0]
                if isinstance(first_item, dict):
                    md_content += "**Structure of items:**\n```\n"
                    for k in first_item.keys():
                        md_content += f"  - {k}\n"
                    md_content += "```\n\n"

                    # Show first few items
                    md_content += "**Sample items (first 3):**\n\n"
                    for i, item in enumerate(value[:3], 1):
                        md_content += f"### Item {i}\n\n"
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, (str, int, float, bool)):
                                    md_content += f"- **{k}**: {v}\n"
                                elif isinstance(v, list):
                                    md_content += f"- **{k}**: {len(v)} items\n"
                                elif isinstance(v, dict):
                                    md_content += f"- **{k}**: {len(v)} keys\n"
                        md_content += "\n"
                else:
                    md_content += f"**Sample items:** {value[:5]}\n\n"

        elif isinstance(value, dict):
            md_content += f"**Type:** Dictionary with {len(value)} keys\n\n"
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool)):
                    md_content += f"- **{k}**: {v}\n"
                elif isinstance(v, list):
                    md_content += f"- **{k}**: List with {len(v)} items\n"
                elif isinstance(v, dict):
                    md_content += f"- **{k}**: Dictionary with {len(v)} keys\n"
                else:
                    md_content += f"- **{k}**: {type(v).__name__}\n"
            md_content += "\n"

        else:
            md_content += f"**Value:** {value}\n\n"

        md_content += "---\n\n"

    # Add placeholder sections for analysis
    md_content += """## Scientific Interpretation

### Key Findings

[To be filled: What are the main biological/scientific findings from this scenario?]

### Biological Significance

[To be filled: What is the biological significance of these results?]

### Clinical Relevance

[To be filled: How might these results be relevant for clinical applications?]

---

## Suggested Visualizations

### Primary Visualizations

1. **[Visualization 1 Name]**
   - Type: [e.g., Network graph, Bar chart, Heatmap]
   - Purpose: [What biological insight does it provide?]
   - Data sources: [Which data fields to use]

2. **[Visualization 2 Name]**
   - Type: [e.g., Network graph, Bar chart, Heatmap]
   - Purpose: [What biological insight does it provide?]
   - Data sources: [Which data fields to use]

### Secondary Visualizations

[Add more visualization ideas as needed]

---

## Notes and Observations

[Add any additional notes, observations, or questions about this scenario's results]

---

*Generated from: axl_breast_cancer_all_6_scenarios_20251112_171142.json*
"""

    # Write markdown file
    output_file = output_dir / f"scenario_{scenario_id}_{scenario_name.lower().replace(' ', '_')}.md"
    with open(output_file, 'w') as f:
        f.write(md_content)

    print(f"Created: {output_file}")
    return output_file

def main():
    # Load results
    results_file = Path("/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_/results/axl_breast_cancer_all_6_scenarios_20251112_171142.json")

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create output directory
    output_dir = Path("/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_/scenario_analysis")
    output_dir.mkdir(exist_ok=True)

    print(f"\nExtracting scenario data from: {results_file.name}")
    print(f"Output directory: {output_dir}\n")

    # Process each scenario
    for scenario_data in results['results']:
        create_scenario_markdown(scenario_data, output_dir)

    print(f"\n✅ Completed! Created {len(results['results'])} scenario markdown files in {output_dir}")

if __name__ == '__main__':
    main()
