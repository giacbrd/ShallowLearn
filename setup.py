import io
import os

import numpy
from setuptools import Extension
from setuptools import setup, find_packages

__author__ = 'Giacomo Berardi <giacbrd.com>'


def readme():
    with open('README.rst') as f:
        return f.read()


def readfile(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()


package_dir = os.path.join(os.path.dirname(__file__), 'shallowlearn')

setup(
    name='ShallowLearn',
    version='0.0.5',
    description='A collection of supervised learning models based on shallow neural network approaches '
                '(e.g., word2vec and fastText) with some additional exclusive features',
    long_description=readfile('README.rst'),
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    url='https://github.com/giacbrd/ShallowLearn',
    author='Giacomo Berardi',
    author_email='barnets@gmail.com',
    packages=['shallowlearn'] + find_packages('shallowlearn'),
    install_requires=[
        'cython>=0.24.1',
        'scikit-learn>=0.18',
        'gensim==0.13.4',
        'fasttext==0.8.2'
    ],
    setup_requires=['pytest-runner==2.9'],
    tests_require=['pytest==3.0.3'],
    include_package_data=True,
    ext_modules=[
        Extension(
            'shallowlearn.word2vec_inner',
            sources=['./shallowlearn/word2vec_inner.pyx'],
            include_dirs=[package_dir, numpy.get_include()]
        )
    ]
)
