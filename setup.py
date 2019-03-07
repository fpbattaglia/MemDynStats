from setuptools import setup, find_packages

from codecs import open
from os import path
import os
import glob
import subprocess

config = {

}

current_path = os.getcwd()
try:
    GIT_VERSION = subprocess.check_output(["git", "describe", "--tags"]).strip().decode('utf-8')
    GIT_VERSION = GIT_VERSION.split('-')[0]
except subprocess.CalledProcessError as e:
    GIT_VERSION = "0.1"
os.chdir(current_path)

print(GIT_VERSION)

here = path.abspath(path.dirname(__file__))

# find test_data
test_data = glob.glob('test_data/*.mat')

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(description='statistics addons for neural data analysis in the MemoryDynamics Lab',
      long_description=long_description,
      url='https://github.com/MemDynLab/MemDynStats',
      author='Francesco P. Battaglia',
      author_email='fpbattaglia@gmail.com',
      version=GIT_VERSION,
      license='GPL3',
      install_requires=['numpy', 'pandas', 'numba', 'scipy'],
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      name='MemDynStats',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ]
      )
