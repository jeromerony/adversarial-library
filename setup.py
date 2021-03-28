import os
from distutils.core import setup

from setuptools import find_packages

setup_path = os.path.abspath(os.path.dirname(__file__))

setup(name='adv_lib',
      version='0.1',
      url='https://github.com/jeromerony/adversarial-library',
      maintainer='Jerome Rony',
      maintainer_email='jerome.rony@gmail.com',
      description='Library of various adversarial resources in PyTorch',
      author='Jerome Rony',
      author_email='jerome.rony@gmail.com',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      python_requires='>=3.8',
      install_requires=[
          'torch>=1.7.0',
          'torchvision>=0.8.0',
          'tqdm>=4.48.0',
          'visdom>=0.1.8',
      ],
      packages=find_packages())
