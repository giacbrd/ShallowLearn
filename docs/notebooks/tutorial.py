__author__ = 'Giacomo Berardi <giacbrd.com>'

import io
import sys
import unicodedata

import logging

import itertools
from collections import Counter

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

from shallowlearn.models import GensimFastText, FastText

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


_remove_punct = dict.fromkeys(i for i in xrange(sys.maxunicode)
                              if unicodedata.category(unichr(i)).startswith('P'))


def iter_file(file_name, labels):
    with io.open(file_name) as input:
        for line in input:
            line = line.split()
            result = []
            for word in line:
                if word.startswith('__label__'):
                    if labels:
                        result.append(word[9:])
                else:
                    if not labels:
                        result.append(word.lower().translate(_remove_punct))
            yield tuple(result)


#FIXME don't use lists but rigenerate iterators! (inside shallowleanr these are consumed more times) see Gensim for solutions
X_train = list(iter_file('/Users/giacomo/Desktop/temp/shallowlearn_tutorial/cooking.train', labels=False))
y_train = list(iter_file('/Users/giacomo/Desktop/temp/shallowlearn_tutorial/cooking.train', labels=True))
X_test = list(iter_file('/Users/giacomo/Desktop/temp/shallowlearn_tutorial/cooking.valid', labels=False))
y_test = list(iter_file('/Users/giacomo/Desktop/temp/shallowlearn_tutorial/cooking.valid', labels=True))

classifier1 = GensimFastText(size=200, iter=5, min_count=2, sample=0.)
classifier2 = FastText(dim=200, epoch=5, min_count=2, t=0., loss='hs')

# print X_train
# print y_train
# print len(X_train)
# print len(y_train)

mm = MultiLabelBinarizer()
mm.fit(y_train + y_test)

for classifier in (classifier1, classifier2):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    yy = mm.transform(t for i, t in enumerate(y_test) if predictions[i] is not None)
    pp = mm.transform([(p,) for p in predictions if p is not None])
    print predictions
    print metrics.classification_report(yy, pp)
#     for i, sample in enumerate(X_test):
#         print sample, predictions[i], y_test[i]
#
    print Counter(predictions).most_common(20)
    print Counter(itertools.chain(*y_test)).most_common(20)

