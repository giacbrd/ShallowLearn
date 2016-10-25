#!/usr/bin/env bash

echo "Testing fast Cython version"
python setup.py test
mkdir temp
mv ../shallowlearn/word2vec_innner* ./temp/
echo "Testing slow Numpy version"
python setup.py test
mv ./temp/* ../shallowlearn/
rm -rf temp
