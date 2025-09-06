from setuptools import setup, find_packages

setup(
    name="laser",
    version="0.1.0",
    description="LASER: Video understanding and segmentation tools",
    author="Jiani Huang, Matthew Kuo",
    author_email="jianih@seas.upenn.edu",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "Pillow",
        "tqdm",
        "opencv-python",
        # Add other dependencies as needed
    ],
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
