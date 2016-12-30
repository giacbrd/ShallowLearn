Changelog
=========

`0.0.5 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.5>`_ (2016-12-30)
----------------------------------------------------------------------------------

* Online learning and better pre-training in GensimFastTex:
    - Hashing trick for building the vocabulary, similar to the original fastText approach
    - It is possible to pre-fit word embeddings from a dataset with word2vec
    - True online earning with ``partial_fit``, the vocabulary is incrementally updated
* New version of fastText.py: 0.8.2
* New version of Gensim: 0.13.4
* Fixed ``predict_proba`` output format

`0.0.4 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.4>`_ (2016-11-05)
----------------------------------------------------------------------------------

* Faster prediction for multiple labels with one ``predict`` call
* Better persistence with ``save`` and ``load`` methods
* Fixed parameter names convention

`0.0.3 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.3>`_ (2016-10-28)
----------------------------------------------------------------------------------

* FastText classifier based on version 0.8.0 of https://github.com/salestock/fastText.py
* GensimFastText has now:
    - negative sampling
    - softmax as alternative output function
    - almost complete LabeledWord2Vec as subclass of Gensim's Word2Vec
* More tests

`0.0.2 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.2>`_ (2016-10-14)
----------------------------------------------------------------------------------

* Cython code for fastText in Gensim
* Script for benchmarks

`0.0.1 <https://github.com/giacbrd/ShallowLearn/releases/tag/0.0.1>`_ (2016-10-11)
----------------------------------------------------------------------------------

* First working model: GensimFastText
