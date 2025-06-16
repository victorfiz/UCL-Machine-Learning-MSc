from setuptools import setup, find_packages

setup(
    name="custom-tqdm",
    version="0.1.0",
    description="Custom tqdm implementation",
    author="User",
    packages=find_packages(),
    # Create a package that will override the real tqdm
    py_modules=["tqdm"]
)