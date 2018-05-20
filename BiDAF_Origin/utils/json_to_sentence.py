import os


def load_data(brc_data, tar_dir):
    # print('Converting ' + file)
    # fin = open(file, encoding='utf8')
    out_file = os.path.join(tar_dir, 'train_set.seg')
    with open(out_file, 'w', encoding='utf8') as ftrain:
        for sample in brc_data.train_set:
            ftrain.write(' '.join(sample['segmented_question']) + '\n')
            for passage in sample['passages']:
                ftrain.write(' '.join(passage['passage_tokens']) + '\n')
            del sample
    ftrain.close()

    out_file = os.path.join(tar_dir, 'dev_set.seg')
    with open(out_file, 'w', encoding='utf8') as fdev:
        for sample in brc_data.dev_set:
            fdev.write(' '.join(sample['segmented_question']) + '\n')
            for passage in sample['passages']:
                fdev.write(' '.join(passage['passage_tokens']) + '\n')
            del sample
    fdev.close()

    out_file = os.path.join(tar_dir, 'test_set.seg')
    with open(out_file, 'w', encoding='utf8') as ftest:
        for sample in brc_data.test_set:
            ftest.write(' '.join(sample['segmented_question']) + '\n')
            for passage in sample['passages']:
                ftest.write(' '.join(passage['passage_tokens']) + '\n')
            del sample
    ftest.close()
