from setuptools import setup, find_packages

setup(
    name='tahini',
    version='2020.10.1',
    author='tahini-dev',
    author_email='tahini.dev@gmail.com',
    description='Python package for graph theory',
    long_description='https://github.com/tahini-dev/tahini',
    url='https://github.com/tahini-dev/tahini',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
