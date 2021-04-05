python setup.py sdist
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ g4l-smc
