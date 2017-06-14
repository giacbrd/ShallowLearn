#!/usr/bin/env bash

# Arguments: <version>

read -r -p "Have you updated the CHANGELOG and renamed every version occurrence to $1? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8
    git flow release start $1
    git flow release finish $1
    git push --follow-tags
    rm -fr build/*
    rm -fr dist/*
    rm -fr ShallowLearn.egg-info/*
    python setup.py sdist bdist_wheel
    twine upload dist/ShallowLearn-$1.tar.gz
    echo "Remember to update the release description on Github!"
fi

