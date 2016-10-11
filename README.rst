ShallowLearn
============
A collection of supervised learning models based on shallow neural network approaches (e.g., word2vec and fastText)
with some additional exclusive features.
They are written in Python and fully compatible with `Scikit-learn <http://scikit-learn.org>`_.

.. image:: https://travis-ci.org/giacbrd/ShallowLearn.svg?branch=master
    :target: https://travis-ci.org/giacbrd/ShallowLearn
.. image:: https://badge.fury.io/py/shallowlearn.svg
    :target: https://badge.fury.io/py/shallowlearn

Installation
------------
``pip install shallowlearn``

Models
------
``shallowlearn.models.GensimFastText``
    A supervised learning model based on the fastText algorithm [1]_.
    The code is mostly taken and rewritten from `Gensim <https://radimrehurek.com/gensim>`_,
    it takes advantage of its optimizations and support.
    **TODO**: Cython code

``shallowlearn.models.FastText``
    **TODO**: The supervised algorithm of fastText implemented in https://github.com/salestock/fastText.py

``shallowlearn.models.DeepInverseRegression``
    **TODO**: Based on https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score

Exclusive Features
------------------
**TODO**

Performances
------------
**TODO**:  Comparison with other classifiers in effectiveness and computation cost

References
----------
    .. [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification
