import io
import os

from setuptools import setup, find_packages

__author__ = 'Giacomo Berardi <giacbrd.com>'


def readme():
    with open('README.rst') as f:
        return f.read()


def readfile(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()

setup(
    name='ShallowLearn',
    version='0.0.1',
    description='A collection of supervised learning models based on shallow neural network approaches (e.g., word2vec and fastText)',
    long_description=readfile('README.rst'),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    url='https://github.com/giacbrd/ShallowLearn',
    download_url='',
    author='Giacomo Berardi',
    author_email='barnets@gmail.com',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'gensim==0.13.2',
        'scikit-learn>=0.18'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False
)
