#!/usr/bin/env bash

# Arguments: <version>

git flow release start $1
git flow release finish $1
git push --tags
rm -fr build/*
rm -fr dist/*
rm -fr ShallowLearn.egg-info/*
python setup.py sdist bdist_wheel
twine upload dist/ShallowLearn-$1.tar.gz