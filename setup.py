from setuptools import find_packages, setup


def get_requirements(file_path: str) -> list[str]:
    requirements = []
    with open(file_path, "r") as f:
        requirements = [line.strip() for line in f.readlines() if line.strip()]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Juan Camilo Ossa Giraldo",
    author_email="ossagiraldojuancamilo@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
