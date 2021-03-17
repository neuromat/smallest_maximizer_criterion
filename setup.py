from setuptools import setup, find_packages

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='g4l-smc',
    version='0.0.3',
    description='Context tree estimation using the Smallest Maximizer Criterion (SMC)',
    author='Arthur Tofani',
    author_email='arthur.tofani@usp.br',
    url='https://github.com/arthurtofani/smallest_maximizer_criterion',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=required,
    scripts=['bin/ctm', 'bin/smc', 'bin/samplegen', 'bin/ct_view'],
    python_requires='>=3.7',
)
