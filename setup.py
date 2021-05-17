from setuptools import find_packages, setup


setup(
    name='MCI',
    version='0.1.1',
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
