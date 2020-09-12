from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tahini',
    version='2020.09',
    author='tahini-dev',
    author_email='',
    description='Python package for ...',
    long_description=long_description,
    url='https://github.com/tahini-dev/tahini',
    packages=find_packages(),
    install_requires=[
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language:: Python :: 3',
        'Programming Language:: Python :: 3.7',
        'Programming Language:: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
)
