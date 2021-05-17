from setuptools import find_packages, setup


setup(
    name='MLWELL-MCI',
    version='0.1.0',
    packages=find_packages(),
    description='implementation of the marginal contribution feature importance score',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'sklearn',
        'tqdm',
        'pandas'
    ]
)
