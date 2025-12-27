"""Setup configuration for ag-viz package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='ag-viz',
    version='0.1.0',
    description='Visualization tools for ARIMA-GARCH models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='rtrimble13',
    author_email='',
    url='https://github.com/rtrimble13/arima-garch',
    packages=find_packages(),
    install_requires=[
        'click>=8.0',
        'matplotlib>=3.5',
        'seaborn>=0.12',
        'pandas>=1.5',
        'numpy>=1.23',
        'scipy>=1.9',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'jupyter>=1.0',
            'notebook>=6.4',
        ],
    },
    entry_points={
        'console_scripts': [
            'ag-viz=ag_viz.cli:cli',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='arima garch time-series forecasting visualization finance',
)
