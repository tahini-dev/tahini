from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tahini',
    version='2020.10',
    author='tahini-dev',
    author_email='tahini.dev@gmail.com',
    description='Python package for graph theory',
    long_description=long_description,
    url='https://github.com/tahini-dev/tahini',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language:: Python :: 3',
        'Programming Language:: Python :: 3.7',
        'Programming Language:: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    include_package_data=True
)
