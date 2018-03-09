import simplejson as json

path = r"G:\OpenSourceDatasetCode\Dataset\dureader_preprocessed"
train_search = path + r'\trainset\search.train.json'
train_zhidao = path + r'\trainset\zhidao.train.json'
test_search = path + r'\testset\search.test.json'
test_zhidao = path + r'\testset\zhidao.test.json'
dev_search = path + '\devset\search.dev.json'
dev_zhidao = path + '\devset\zhidao.dev.json'

# files = [train_search, train_zhidao, dev_search, dev_zhidao, test_search, test_zhidao]
files = [dev_search, dev_zhidao]
sets = []
for i in range(len(files)):
    file = open(files[i], encoding='utf8')
    line = file.readline()
    tmp = []
    while line:
        tmp.append(json.loads(line))
        line = file.readline()

    file.close()
    sets.append(tmp)

dev_set = sets[0] + sets[1]
print(len(dev_set))
# test_set = sets[2].extend(sets[3])
# print(len(test_set))