from setuptools import find_packages, setup


setup(
    name='Marginal-Contribution-Feature-Importance',
    version='0.1.0',
    packages=find_packages(),
    description='implementation of the marginal contribution feature importance score',
    download_url='https://github.com/TAU-MLwell/Marginal-Contribution-Feature-Importance/archive/refs/tags/0.1.0.tar.gz',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'sklearn',
        'tqdm',
        'pandas'
    ]
)
