ShallowLearn
============
A collection of supervised learning models based on shallow neural network approaches (e.g., word2vec and fastText)
with some additional exclusive features.
They are written in Python and fully compatible with `Scikit-learn <http://scikit-learn.org>`_

Getting Started
---------------
Install the latest version:
``pip install shallowlearn``
Import models from ``shallowlearn.models``, they implement the standard methods for supervised learning in Scikit-learn,
e.g., ``fit(X, y)``, ``predict(X)``, etc.

Data is raw text, each sample is a list of tokens (words of a document), while each target value in ``y`` can be a
single label (or a list in case of multi-label training set) associated with the relative sample.

Models
------
``shallowlearn.models.GensimFTClassifier``
    A supervised learning model based on the fastText algorithm [1]_.
    The code is mostly taken and rewritten from `Gensim <https://radimrehurek.com/gensim>`_,
    it takes advantage of its optimizations and support.
    **TODO**: Cython code

``shallowlearn.models.FastTextClassifier``
    **TODO**: The supervised algorithm of fastText implemented in https://github.com/salestock/fastText.py

``shallowlearn.models.DeepIRClassifier``
    **TODO**: Based on https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score

Exclusive Features
------------------
**TODO**

Benchmarks
----------
``scripts/document_classification_20newsgroups.py`` refers to this `Scikit-learn example <http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html>`_ in which text classifiers are compared on a reference dataset.
We added ``GensimFTClassifier`` and the original *fastText* implementation for a solid benchmark of our models.
Results as for release **0.0.1**:
**TODO**

TODO
----

- Documents can be structured, made of different sections, learned independently
- Taking into account https://github.com/RaRe-Technologies/gensim/pull/847, implementing the hashing trick
- Given the previous point, implementing n-grams of words

References
----------
    .. [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification