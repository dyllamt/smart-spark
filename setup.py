from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = fh.read()


setup(
    name='smart-spark',
    version='0.0.1',
    description='Processing for S.M.A.R.T. hard-drive statistics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url='',
    author='Maxwell Dylla',
    author_email='maxwell.dylla@gmail.com',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['pyspark'],
)
