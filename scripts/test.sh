#!/usr/bin/env bash

echo "Testing fast Cython version"
E1=$(python setup.py test)
mkdir temp
mv shallowlearn/word2vec_inner* temp/
touch shallowlearn/word2vec_inner.pyx
echo "Testing slow Numpy version"
E2=$(python setup.py test)
mv temp/word2vec_inner* shallowlearn/
rm -rf temp
exit $(($E2 + $E1))