from setuptools import setup, find_packages
import sys

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='smc',
    version='0.1.1',
    description='Smallest Maximizer Criterion',
    author='Arthur Tofani',
    author_email='arthur.tofani@usp.com',
    url='https://github.com/arthurtofani/smallest_maximizer_criterion',
#    download_url='https://github.com/arthurtofani/smallest_maximizer_criterion',
    packages=find_packages(),
#    package_data={'': ['example_data/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=required,
    python_requires='>=3.6',
    #extras_require={
    #    'docs': ['numpydoc', 'sphinx!=1.3.1', 'sphinx_rtd_theme',
    #             'matplotlib >= 3.2.1',
    #             'sphinx-multiversion >= 0.2.3',
    #             'sphinx-gallery',
    #             'sphinxcontrib-svg2pdfconverter',
    #             'presets'],
    #    'tests': ['matplotlib >= 3.2.1',
    #              'pytest-mpl',
    #              'pytest-cov',
    #              'pytest',
    #              'contextlib2'],
    #    'display': ['matplotlib >= 3.2.1'],
    #}
)
