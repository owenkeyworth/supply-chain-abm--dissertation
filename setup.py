from setuptools import setup, find_packages

setup(
    name="supply_chain_abm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'mesa',
        'pandas',
        'numpy',
        'matplotlib',
        'networkx',
        'sqlalchemy',
        'python-dotenv'
    ],
) 