#!/usr/bin/env python
"""
Test visualization generation with recent pipeline results.
"""

import logging
from pathlib import Path
from src.visualization.orchestrator import VisualizationOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run visualization test with recent results."""

    # Use the most recent comprehensive result file
    json_path = "results/axl_breast_cancer_all_6_scenarios_20251123_172322.json"
    output_dir = "results/figures_test"

    logger.info(f"Testing visualization with: {json_path}")
    logger.info(f"Output directory: {output_dir}")

    # Create orchestrator
    orchestrator = VisualizationOrchestrator(style='publication')

    try:
        # Generate visualizations for all scenarios
        logger.info("Generating visualizations for all scenarios...")
        generated_files = orchestrator.visualize_all_scenarios(
            json_path=json_path,
            output_dir=output_dir,
            interactive=False,  # Start with static only
            formats=['png']
        )

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VISUALIZATION GENERATION SUMMARY")
        logger.info("="*60)

        total_files = 0
        for scenario_id, files in sorted(generated_files.items()):
            count = len(files)
            total_files += count
            logger.info(f"Scenario {scenario_id}: {count} visualizations")
            for file_path in files:
                logger.info(f"  - {file_path.name}")

        logger.info("="*60)
        logger.info(f"Total visualizations generated: {total_files}")
        logger.info(f"Output directory: {Path(output_dir).absolute()}")
        logger.info("="*60)

        # Check if index.html was created
        index_file = Path(output_dir) / "index.html"
        if index_file.exists():
            logger.info(f"\nSummary report: {index_file}")
            logger.info("Open in browser to view all visualizations")

        return True

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
