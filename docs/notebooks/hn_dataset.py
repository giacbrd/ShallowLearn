import json
import sys
import io
import requests
from gensim.utils import to_unicode
from datetime import datetime
from six.moves import range

__author__ = 'Giacomo Berardi <giacbrd.com>'


SIZE = 10000

session = requests.Session()

date_end = datetime.strptime(sys.argv[2], '%Y-%m-%d')
max_id = int(session.get('https://hacker-news.firebaseio.com/v0/maxitem.json').text.strip())
count = 0

with io.open(sys.argv[1], 'w', encoding='utf-8') as dataset:
    for item_id in reversed(range(max_id)):
        resp = session.get('https://hacker-news.firebaseio.com/v0/item/%s.json' % item_id)
        item = resp.json()
        if not resp.status_code == 200 or item is None:
            print(item_id, resp.status_code, resp.text)
            continue
        if not item.get('deleted') and 'type' in item and item['type'] == 'story' and datetime.fromtimestamp(item['time']) <= date_end:
            dataset.write(to_unicode(json.dumps(item) + '\n'))
            count += 1
        if count >= SIZE:
            break
