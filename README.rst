ShallowLearn
============
A collection of supervised learning models based on shallow neural network approaches (e.g., word2vec and fastText).
They are written in Python and fully compatible with `Scikit-learn <http://scikit-learn.org>`_

Models
------
shallowlearn.models.GensimFTClassifier
    A supervised learning model based on the fastText algorithm [1]_.
    The code is mostly taken and rewritten from `Gensim <https://radimrehurek.com/gensim>`_,
    It takes advantage of its optimizations and support.

shallowlearn.models.fastTextClassifier
    **TODO** - The supervised algorithm of fastText implemented in https://github.com/salestock/fastText.py

shallowlearn.models.DeepIRClassifier
    **TODO** - Based on https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score
Performances
------------
**TODO** - Comparison with other classifiers in effectiveness and computation cost
TODO
----
- Tests!
- Documents can be structured, made of different sections, learned independently.
- Taking into account https://github.com/RaRe-Technologies/gensim/pull/847, implementing the hashing trick
References
----------
    .. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
