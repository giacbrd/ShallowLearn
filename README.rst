ShallowLearn
============
A collection of supervised learning models based on shallow neural network approaches (e.g., word2vec and fastText)
with some additional exclusive features.
Written in Python and fully compatible with `scikit-learn <http://scikit-learn.org>`_.

**Discussion group** for users and developers: https://groups.google.com/d/forum/shallowlearn

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

Data is raw text, each sample in the iterable ``X`` is a list of tokens (words of a document), 
while each element in the iterable ``y`` (corresponding to an element in ``X``) can be a single label or a list in case of a multi-label training set. Obviously, ``y`` must be of the same size of ``X``.

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

Using the hashing trick together with ``partial_fit(X, y)`` yield a powerful *online* text classifier.

Constructor argument names are a mix between the ones of Gensim and the ones of fastText (see this class docstring).

.. code:: python

    >>> from shallowlearn.models import GensimFastText
    >>> clf = GensimFastText(size=100, min_count=0, loss='hs', iter=3, seed=66)
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

DeepAveragingNetworks
~~~~~~~~~~~~~~~~~~~~~
*TODO*: Based on https://github.com/miyyer/dan

Exclusive Features
------------------
Next cool features will be listed as Issues in Github

Persistence
~~~~~~~~~~~
Any model can be serialized and de-serialized with the two methods ``save`` and ``load``.
They overload the `SaveLoad <https://radimrehurek.com/gensim/utils.html#gensim.utils.SaveLoad>`_ interface of Gensim,
so it is possible to control the cost on disk usage of the models, instead of simply *pickling* the objects.
``save`` can create multiple files with names prefixed by the name given to the serialized model.

.. code:: python

    >>> from shallowlearn.models import GensimFastText
    >>> clf = GensimFastText(size=100, min_count=0, loss='hs', iter=3, seed=66)
    >>> clf.save('./model')
    >>> loaded = GensimFastText.load('./model') # it also creates ./model.CLF

Benchmarks
----------
The script ``scripts/document_classification_20newsgroups.py`` refers to this
`scikit-learn example <http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html>`_
in which text classifiers are compared on a reference dataset;
we added our models to the comparison.
**The current results, even if still preliminary, are comparable with other
approaches, achieving the best performance in speed**.

Results as of release `0.0.4 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.4>`_,
with *chi2_select* option set to 80%.
The times take into account of *tf-idf* vectorization in the “classic” classifiers, and the I/O operations for the
training of fastText.py.
The evaluation measure is *macro F1*.

.. image:: https://rawgit.com/giacbrd/ShallowLearn/master/benchmark.svg
    :alt: Text classifiers comparison
    :align: center
    :width: 888 px

References
----------
.. [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification
