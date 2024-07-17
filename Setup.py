# setup.py

# setup.py

from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='0.1',
    packages=find_packages(),  # Automatically finds the package(s)
    description='A simple ETL package',
    author='Your Name',
    author_email='your.email@example.com',
    install_requires=[
        'oracledb',
        'pyarrow',
        'pandas',
        'colorama',
        'yaml',
        'sqlalchemy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
