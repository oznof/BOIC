import re
import io
import os
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = 0.1

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='boic',
    version=version,
    packages=find_packages(),
    url='https://github.com/oznof/BOIC',
    license='MIT',
    author='Afonso Eduardo',
    author_email='aflmeduardo@gmail.com',
    description='Bayesian optimization with Informative Covariance',
    python_requires="==3.8.10",
    install_requires=requirements
)
