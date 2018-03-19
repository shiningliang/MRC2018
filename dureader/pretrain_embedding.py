import simplejson as json
import sys
from gensim.models import word2vec

sys.path.append('..')


def load_data(data_file):
    fin = open(data_file, encoding='utf8')
    data_set = []
    for lidx, line in enumerate(fin):
        # 每次加载一个json文件-sample
        sample = json.loads(line.strip())
        data_set.append(sample['segmented_question'])
        for d_idx, doc in enumerate(sample['documents']):
            data_set += doc['segmented_paragraphs']
        del sample
    fin.close()
    return data_set


corpus = []
file_path = ['../data/demo/trainset/search.train.json', '../data/demo/devset/search.dev.json',
             '../data/demo/testset/search.test.json']
for file in file_path:
    corpus += load_data(file)

model = word2vec.Word2Vec(corpus, size=300, min_count=2, workers=8)
with open('w2v_dic.data', 'w', encoding='utf-8') as f:
    for word in model.wv.vocab:
        f.write(word + ' ')
        for num in list(map(str, model[word])):
            f.write(num + ' ')
        f.write('\n')
f.close()
