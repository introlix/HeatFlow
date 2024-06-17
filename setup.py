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
    python_requires=">=3.10",
)