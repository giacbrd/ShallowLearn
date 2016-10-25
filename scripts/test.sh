#!/usr/bin/env bash

# test fast cython version
python setup.py test
mkdir temp
mv ../shallowlearn/word2vec_innner* ./temp/
# test numpy version
python setup.py test
mv ./temp/* ../shallowlearn/
rm -rf temp
