"""
PuXle: Parallelized Puzzles with JAX

This setup.py is maintained for backward compatibility.
The main configuration is now in pyproject.toml.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="puxle",
    version="0.0.1",
    author="tinker495",
    author_email="wjdrbtjr495@gmail.com",
    description="Parallelized Puzzles implementation based on Jax!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tinker495/puxle",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "jax[cuda]>=0.4.0",
        "chex>=0.1.0",
        "tabulate>=0.9.0",
        "termcolor>=1.1.0",
        "opencv-python>=4.10.0",
        "tqdm>=4.67.1",
        "numpy>=2.2.0",
        "xtructure @ git+https://github.com/tinker495/xtructure.git",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
