# Data Preprocessing Strategy

Here is an example of preprocessed data:
```
{
    "question_id": 186358,
    "question_type": "YES_NO",
    "question": "上海迪士尼可以带吃的进去吗",
    "segmented_question": ["上海", "迪士尼", "可以", "带", "吃的", "进去", "吗"],
    "documents": [
        {
            "paragraphs": ["text paragraph 1", "text paragraph 2"],
            "segmented_paragraphs": [["tokens of paragraph1"], ["tokens of paragraph2"]],
            "title": "上海迪士尼可以带吃的进去吗",
            "segmented_title": ["上海", "迪士尼", "可以", "带", "吃的", "进去", "吗"],
            "bs_rank_pos": 1,
            "is_selected": True
            "most_related_para": 0,
        },
        # ...
    ],
    "answers": [
        "完全密封的可以，其它不可以。",                                        # answer1
        "可以的，不限制的。只要不是易燃易爆的危险物品，一般都可以带进去的。",  # answer2
        "罐装婴儿食品、包装完好的果汁、水等饮料及包装完好的食物都可以带进乐园，但游客自己在家制作的食品是不能入园，因为自制食品有一定的安全隐患。"        # answer3
    ]
    "answer_docs": [0],
    "answer_spans": [[0, 15]]
    "fake_answers": ["完全密封的可以，其他不可以。"],
    "match_scores": [1.00],
    "segmented_answers": [
        ["完全", "密封", "的", "可以", "，", "其它", "不可以", "。"],
        ["tokens for answer2"],
        ["tokens for answer3"],

    "yesno_answers": [
        "Depends",                      # corresponding to answer 1
        "Yes",                          # corresponding to answer 2
        "Depends"                       # corresponding to asnwer 3
    ]
}
```

To make it easier for researchers to use DuReader Dataset, we also release the preprocessed data. The preprocessing mainly does the following things:
1. Word segmentation. We segment all questions, answers, document titles and paragraphs into Chinese words, and the results are stored with a new field which prefix the corresponding field name with "segmented_". For example, the segmented question is stored in "segmented_question".
2. Answer paragraph targeting. In DuReader dataset, each question has up to 5 related documents, and the average document length is 394, since it is too heavy to feed all 5 documents into popular RC models, so we previously find the most answer related paragraph that might contain an answer for each document. And we replace original documents with the most related paragraphs  in our baseline models. The most related paragraphs are selected according to highest recall of the answer tokens of each document, and the index of the selected paragraph of each document is stored in "most_related_para".
3. Locating answer span. For many popular RC models, an answer span is required in training. Since the original DuReader dataset doesn't provide the answer span, we provide a simple answer span locating strategy  for convenience in our preprocess code as an optional preprocess strategy. In the strategy, we match real answer with each documents, then search the substring with maximum F1-score of the real answers, and use the span of substring as the candidate answer span. For each question we find single span as candidate, and store it in the "answer_spans" field, the corresponding substring spanned by answer span is stored in "fake_answers", the recall of the answer span of the real answer is stored in "match_scores", and the document index of the answer span is stored in "answer_docs".

Except for word segmentation, the rest of the preprocessing strategy is implemented in `utils/preprocess.py`.

数据预处理主要包含以下过程：
1. 分词。我们对所有的问题，答案，文档的标题和段落进行分词，将结果存储在以"segmented_"为前缀的新域中。例如，分词的问题被存储在"segmented_question"
2. 答案目标段落。在DuReader数据集中，每个问题至多有5篇相关文档，文档的平均长度为394。由于将5篇文档全部输入RC模型计算量过大，我们预先在每篇文档中找出可能包含答案的与答案最相关的段落。在baseline模型中用最相关段落代替原文档。根据每篇文档的答案tokens的最高召回率选择最相关段落，被选出的段落的下标存储在"most_related_para"
3. 定位answer span。对于大多数流行的RC模型，训练时需要answer span。因为原始数据集中没有提供answer span，为了方便，在预处理代码中，我们提供了一个简单的answer span定位策略。在该策略中，我们将真实答案与每篇文档匹配，然后搜索真实答案获得最大F1-score的子串，并将子串的span作为候选answer span。对每个问题，我们找出一个span作为候选，存储在"answer_span"域，与answer span对应的子串存储在"fake_answers"，真实答案的answer span的召回率存储在"match_scores"，answer span的文档下标存储在"answer_docs"