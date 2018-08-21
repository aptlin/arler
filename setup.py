from setuptools import setup

__version_info__ = ('0', '0', '1')
__version__ = '.'.join(__version_info__)

setup(
    name="arler",
    version=__version__,
    description=
    "Toy Explorer of Hierarchical Reinforcement Learning through MAXQ & HI-MAT",
    author="Sasha Illarionov",
    author_email="sasha@sdll.space",
    url="https://github.com/sdll/arler",
    keywords=["ML", "HRL", "HI-MAT", "MAXQ"],
    packages=["arler"],
    entry_points={},
    install_requires=["gym", "matplotlib", "pathlib"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 2  - Pre-Alpha",
    ])
