from setuptools import setup

setup(
    name="promap",
    version="0.1",
    description="Models for in-situ fluorosequencing of proteins.",
    url="https://github.com/funkelab/promap",
    author="Jan Funke",
    author_email="funkej@janelia.hhmi.org",
    license="MIT",
    packages=["promap"],
    install_requires=["numpy"],
)
