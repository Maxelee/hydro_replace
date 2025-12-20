"""
Hydro Replace Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="hydro_replace",
    version="0.1.0",
    description="Pipeline for analyzing baryonic correction models using halo replacement",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/hydro_replace",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "h5py>=3.0",
        "astropy>=5.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "mpi": ["mpi4py>=3.0"],
        "power": ["Pylians>=3.0"],
        "bcm": ["BaryonForge>=1.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.0",
            "numpydoc>=1.5",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords="cosmology, simulations, baryon correction, weak lensing",
)
