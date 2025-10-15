from setuptools import setup, find_packages

setup(
    name="mlops-project",
    version="1.0.0",
    description="End-to-end MLOps project for stock return prediction",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires='>=3.10',
)
