from setuptools import setup

setup(
        name='procode',
        version='0.1',
        description='Models for in-situ fluorosequencing of proteins.',
        url='https://github.com/funkelab/procode',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'procode'
        ],
        install_requires=[
            "numpy"
        ]
)
