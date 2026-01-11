"""
OmniTarget Pipeline Setup

Bioinformatics workflow integration platform with MCP server orchestration.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="omnitarget",
    version="0.1.0",
    author="Omara Soliman",
    author_email="omara.soliman@example.com",
    description="Bioinformatics workflow integration platform with MCP server orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OASolliman590/omnitarget",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omnitarget=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.json", "tests/fixtures/*.json"],
    },
    zip_safe=False,
    keywords="bioinformatics, workflow, mcp, pathway, protein, network, simulation",
    project_urls={
        "Bug Reports": "https://github.com/OASolliman590/omnitarget/issues",
        "Source": "https://github.com/OASolliman590/omnitarget",
        "Documentation": "https://omnitarget.readthedocs.io/",
    },
)
