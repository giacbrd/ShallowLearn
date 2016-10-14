ShallowLearn
============
A collection of supervised learning models based on shallow neural network approaches (e.g., word2vec and fastText)
with some additional exclusive features.
Written in Python and fully compatible with `Scikit-learn <http://scikit-learn.org>`_.

.. image:: https://travis-ci.org/giacbrd/ShallowLearn.svg?branch=master
    :target: https://travis-ci.org/giacbrd/ShallowLearn
.. image:: https://badge.fury.io/py/shallowlearn.svg
    :target: https://badge.fury.io/py/shallowlearn

Getting Started
---------------
Install the latest version:

.. code:: shell

    pip install cython
    pip install shallowlearn

Import models from ``shallowlearn.models``, they implement the standard methods for supervised learning in Scikit-learn,
e.g., ``fit(X, y)``, ``predict(X)``, etc.

Data is raw text, each sample is a list of tokens (words of a document), while each target value in ``y`` can be a
single label (or a list in case of multi-label training set) associated with the relative sample.

Models
------
``shallowlearn.models.GensimFastText``
    A supervised learning model based on the fastText algorithm [1]_.
    The code is mostly taken and rewritten from `Gensim <https://radimrehurek.com/gensim>`_,
    it takes advantage of its optimizations (e.g. Cython) and support.

``shallowlearn.models.FastText``
    **TODO**: The supervised algorithm of fastText implemented in https://github.com/salestock/fastText.py

``shallowlearn.models.DeepInverseRegression``
    **TODO**: Based on https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score

Exclusive Features
------------------
**TODO**

Benchmarks
----------
The script ``scripts/document_classification_20newsgroups.py`` refers to this
`Scikit-learn example <http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html>`_
in which text classifiers are compared on a reference dataset;
we added our models to the comparison.
**The current results, even if still preliminary, are comparable with other
approaches, achieving the best performance in speed**.

Results as of release `0.0.2 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.2>`_,
with *chi2_select* option set to 80%.
The times take into account of *tf-idf* vectorization in the “classic” classifiers;
the evaluation measure is *macro F1*.

.. image:: https://cdn.rawgit.com/giacbrd/ShallowLearn/develop/benchmark.svg
    :alt: Text classifiers comparison
    :align: center
    :width: 888 px

References
----------
    .. [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification
