#!/usr/bin/env bash

echo "Testing fast Cython version"
python setup.py test
mkdir temp
mv shallowlearn/word2vec_inner* temp/
touch shallowlearn/word2vec_inner.pyx
echo "Testing slow Numpy version"
python setup.py test
mv temp/word2vec_inner* shallowlearn/
rm -rf temp
