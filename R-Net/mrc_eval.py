# coding:utf8
"""
This module computes evaluation metrics for DuReader dataset.
"""

import argparse
import itertools
import ujson as json
import zipfile
from bleu import BLEUWithBonus
from rouge import RougeLWithBonus

EMPTY = ''
YESNO_LABELS = set(['Yes', 'No', 'Depends'])


def normalize(s):
    """
    Normalize strings to space joined chars.
    Args:
        s: a list of strings.
    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(''.join(tokens))
    return normalized


def data_check(obj):
    """
    Check data.

    Raises:
        Raises AssertionError when data is not legal.
    """
    # 判断是否有answer_id
    assert 'question_id' in obj, "Missing 'question_id' field."
    # assert 'yesno_answers' in obj, \
    #        "Missing 'yesno_answers' field. question_id: {}".format(obj['question_id'])
    # 如果包含yesno_answers，那么格式必须为list
    if "yesno_answers" in obj:
        assert isinstance(obj['yesno_answers'], list), \
            r"""'yesno_answers' field must be a list, if the 'question_type' is not
            'YES_NO', then this field should be an empty list.
            question_id: {}""".format(obj['question_id'])
    else:
        obj["yesno_answers"] = []
    if "entity_answers" not in obj:
        obj["entity_answers"] = []


def read_file(file_name, is_ref=False):
    """
    Read predict answers or reference answers from file.

    Args:
        file_name: the name of the file containing predict result or reference
                   result.

    Returns:
        A dictionary mapping question_id to the result information. The result
        information itself is also a dictionary with has four keys:
        - question_type: type of the query.
        - yesno_answers: A list of yesno answers corresponding to 'answers'.
        - answers: A list of predicted answers.
        - entity_answers: A list, each element is also a list containing the entities
                    tagged out from the corresponding answer string.
    """

    def _open(file_name, mode, zip_obj=None):
        if zip_obj is not None:
            return zip_obj.open(file_name, mode)
        return open(file_name, mode)

    results = {}
    # 是否是参考答案
    if is_ref:
        keys = ['source', 'answers', 'yesno_answers', 'entity_answers', 'question_type']
    else:
        keys = ['answers', 'yesno_answers']
    # 如果是zip文件则以zip方式读取
    zf = zipfile.ZipFile(file_name, 'r') if file_name.endswith('.zip') else None
    # zip包中文件列表
    file_list = [file_name] if zf is None else zf.namelist()

    for fn in file_list:
        for line in _open(fn, 'r', zip_obj=zf):
            try:
                obj = json.loads(line.strip())
            except ValueError:
                raise ValueError("Every line of data should be legal json")
            data_check(obj)
            qid = obj['question_id']
            # 必须有question id
            assert qid not in results, "Duplicate question_id: {}".format(qid)
            results[qid] = {}
            for k in keys:
                if k == 'answers':
                    results[qid][k] = normalize(obj[k])
                else:
                    results[qid][k] = obj[k]
            if is_ref:
                for i, e in enumerate(results[qid]['entity_answers']):
                    results[qid]['entity_answers'][i] = normalize(e)
    return results


def calc_metrics(pred_result, ref_result, bleu_eval, rouge_eval):
    """Computes bleu-4 and rouge-l.

    Args:
        - pred_result: Refer to the returned dict of `read_file` with
                       'is_ref=False'.
        - ref_result: Refer to the returned dict of `ref_file` with
                      'is_ref=True'.
        - bleu_result: A BleuWithBonus object.
        - rouge_result: A RougeLWithBonus object.
    Returns:
        bleu-4 and rouge-l values as a tuple of float values.
    """
    for qid, results in ref_result.items():
        # 根据question id从预测结果中选择答案
        cand_result = pred_result.get(qid, {})
        pred_answers = cand_result.get('answers', [])
        if not pred_answers:
            pred_answers = EMPTY
        else:
            pred_answers = pred_answers[0]
        pred_yn_label = None
        ref_entities = None
        ref_answers = results.get('answers', [])
        if not ref_answers:
            continue
        if results['question_type'] == 'ENTITY':
            ref_entities = set(
                itertools.chain(*results.get('entity_answers', [[]])))
            if not ref_entities:
                ref_entities = None
        if results['question_type'] == 'YES_NO':
            cand_yesno = cand_result.get('yesno_answers', [])
            pred_yn_label = None if len(cand_yesno) == 0 \
                else cand_yesno[0]
        bleu_eval.add_inst(
            pred_answers,
            ref_answers,
            yn_label=pred_yn_label,
            yn_ref=results['yesno_answers'],
            entity_ref=ref_entities)
        rouge_eval.add_inst(
            pred_answers,
            ref_answers,
            yn_label=pred_yn_label,
            yn_ref=results['yesno_answers'],
            entity_ref=ref_entities)
    bleu4 = bleu_eval.score()[-1]
    rouge_l = rouge_eval.score()
    return bleu4, rouge_l


def main(args):
    err = None
    metrics = {}
    bleu4, rouge_l = 0.0, 0.0
    alpha = args.alpha  # default 1.0
    beta = args.beta  # default 1.0
    bleu_eval = BLEUWithBonus(4, alpha=alpha, beta=beta)
    rouge_eval = RougeLWithBonus(alpha=alpha, beta=beta, gamma=1.2)
    # 载入answer文件 格式dict question_id: {answers:[], yesno_answers:[]}
    pred_result = read_file(args.pred_file)
    ref_result = read_file(args.ref_file, is_ref=True)
    bleu4, rouge_l = calc_metrics(pred_result,
                                  ref_result,
                                  bleu_eval,
                                  rouge_eval)
    metrics = {
        'ROUGE-L': round(rouge_l * 100, 2),
        'BLEU-4': round(bleu4 * 100, 2),
    }
    print(json.dumps(metrics, ensure_ascii=False).encode('utf8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', help='predict file')
    parser.add_argument('--ref_file', help='reference file')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='common value of alpha')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='common value of beta')
    args = parser.parse_args()
    main(args)
