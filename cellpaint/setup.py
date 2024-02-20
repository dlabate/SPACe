#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='cellpaint',
    version='10.0.0',
    python_requires='<11',
    description='Implementation of the cellpainting project',
    author='Kazem Safari',
    author_email='mkazem.safari@gmail.com',
    url='https://github.com/kazemSafari/cellpaint',
    packages=find_packages(exclude=[]),
    install_requires=[
        'cellpose==2.2',
        'torch>=1.6', 'torchvision', 'pyclesperanto-prototype', 'sympy',
        'tifffile',
        'numpy>=1.20.0', 'scipy', 'scikit-image>=0.20.0', 'scikit-learn',
        'SimpleITK',
        'pandas>=2.0.0', 'xlsxwriter', 'openpyxl', 'xlrd', 'jupyter',
        'matplotlib', 'seaborn', 'plotly',
        'pathlib', 'tqdm', 'typing', 'cmcrameri', 'pyefd',]
)
