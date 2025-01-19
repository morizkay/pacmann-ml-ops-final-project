from setuptools import setup, find_packages

setup(
    name="gk-prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "mlflow>=2.8.0",
        "flask>=2.0.0",
        "pytest>=7.0.0"
    ],
) 