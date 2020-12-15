import os
from setuptools import setup, find_packages


opts = dict(
        name='BioME',
        version='0.1',
        url='https://github.com/kmherman/BioME',
        license='MIT',
        author='',
        author_email='kmherman@uw.edu',
        description='Supervised Machine Learning for Microbiome Data',
        scripts=['biome/scripts/biome_run.py'],
        install_requires=['torch','scikit-learn','numpy','pandas'],
        packages=find_packages()
        )

if __name__ == '__main__':
    setup(**opts)
