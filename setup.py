#!/usr/bin/env python
"""Setup script for finsearch package."""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='finsearch',
    version='0.1.0',
    description='Lightweight RAG system for financial document analysis',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'experiment': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            # Entry points can be added here if needed
        ],
    },
)