ShallowLearn
============
A collection of supervised learning models based on shallow neural network approaches (e.g., word2vec and fastText)
with some additional exclusive features.
Written in Python and fully compatible with `scikit-learn <http://scikit-learn.org>`_.

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

Import models from ``shallowlearn.models``, they implement the standard methods for supervised learning in scikit-learn,
e.g., ``fit(X, y)``, ``predict(X)``, etc.

Data is raw text, each sample is a list of tokens (words of a document), while each target value in ``y`` can be a
single label (or a list in case of multi-label training set) associated with the relative sample.

Models
------
GensimFastText
~~~~~~~~~~~~~~
A supervised learning model based on the fastText algorithm [1]_.
The code is mostly taken and rewritten from `Gensim <https://radimrehurek.com/gensim>`_,
it takes advantage of its optimizations (e.g. Cython) and support.

It is possible to choose the Softmax loss function (default) or one of its two "approximations":
Hierarchical Softmax and Negative Sampling. It is also possible to load pre-trained word vectors at initialization,
passing a Gensim ``Word2Vec`` or a ShallowLearn ``LabeledWord2Vec`` instance (the latter is retrievable from a
``GensimFastText`` model by the attribute ``classifier``).

Constructor argument names are a mix between the ones of Gensim and the ones of fastText (see this class docstring).

.. code:: python

    >>> from shallowlearn.models import GensimFastText
    >>> clf = GensimFastText(size=100, min_count=0, loss='hs', max_iter=3, random_state=66)
    >>> clf.fit([('i', 'am', 'tall'), ('you', 'are', 'fat')], ['yes', 'no'])
    >>> clf.predict([('tall', 'am', 'i')])
    ['yes']

FastText
~~~~~~~~
The supervised algorithm of fastText implemented in `fastText.py <https://github.com/salestock/fastText.py>`_ ,
which exposes an interface on the original C++ code.
The current advantages of this class over ``GensimFastText`` are the *subwords* ant the *n-gram features* implemented
via the *hashing trick*.
The constructor arguments are equivalent to the original `supervised model
<https://github.com/salestock/fastText.py#supervised-model>`_, except for ``input_file``, ``output`` and
``label_prefix``.

**WARNING**: The only way of loading datasets in fastText.py is through the filesystem (as of version 0.8.0),
so data passed to ``fit(X, y)`` will be written in temporary files on disk.

.. code:: python

    >>> from shallowlearn.models import FastText
    >>> clf = FastText(dim=100, min_count=0, loss='hs', epoch=3, bucket=5, word_ngrams=2)
    >>> clf.fit([('i', 'am', 'tall'), ('you', 'are', 'fat')], ['yes', 'no'])
    >>> clf.predict([('tall', 'am', 'i')])
    ['yes']

DeepInverseRegression
~~~~~~~~~~~~~~~~~~~~~
*TODO*: Based on https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score

Exclusive Features
------------------
*TODO: future features are going to be listed as Issues*

Benchmarks
----------
The script ``scripts/document_classification_20newsgroups.py`` refers to this
`scikit-learn example <http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html>`_
in which text classifiers are compared on a reference dataset;
we added our models to the comparison.
**The current results, even if still preliminary, are comparable with other
approaches, achieving the best performance in speed**.

Results as of release `0.0.3 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.3>`_,
with *chi2_select* option set to 80%.
The times take into account of *tf-idf* vectorization in the “classic” classifiers, and the I/O operations for the
training of fastText.py. The evaluation measure is *macro F1*.

.. image:: https://rawgit.com/giacbrd/ShallowLearn/develop/benchmark.svg
    :alt: Text classifiers comparison
    :align: center
    :width: 888 px

References
----------
.. [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification
