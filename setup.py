from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tahini',
    version='2021.01.1',
    author='tahini-dev',
    author_email='tahini.dev@gmail.com',
    description='Python package for graph theory',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tahini-dev/tahini',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'pandas',
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
