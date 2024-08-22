from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="heatflow",
    version="0.0.1",
    author="Satyam Mishra",
    author_email="tubex998@gmail.com",
    description="HeatFlow is a python framework to work with neural networks, tensor, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.19.5,<2.0.0',
        'numba>=0.51.1',
        'tqdm>=4.55.1',
        'requests>=2.26.0',
    ],
    python_requires=">=3.10",
)
