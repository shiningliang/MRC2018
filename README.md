# MRC2018
- 2018机器阅读理解技术竞赛 [竞赛网站](http://mrc2018.cipsc.org.cn/)
- 参赛模型：BiDAF+Self Attention+Pre(single)
- 最终排名：28/105（菜鸡第一次参赛）

## 最近更新
- 2018/08/06更新，po主参加了在[语言与智能高峰论坛](http://www.cipsc.org.cn/lis2018/index.html)上举办的比赛颁奖典礼，发现都是前期特征工程提升巨大，模型上未有亮眼工作，如果拿到了前几名的技术报告，会推上来
- 2018/08/06更新，百度现已开放全部数据，下边的数据集统计表中已更新链接，比赛成绩也会放上来，大家可以日常打榜。颁奖典礼上负责人表示，比赛明年还会继续举办，大家加油！

## 参考模型
- [R-Net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
- [BiDAF](https://allenai.github.io/bi-att-flow/)

## 参考代码
- [HKUST](https://github.com/HKUST-KnowComp/R-Net)
- [DuReader](https://github.com/baidu/DuReader)

## Requirements
### General
- Python >= 3.4
- numpy

### Python Packages
- tensorflow-gpu >= 1.5.0
- ujson
- pickle
- tqdm

### Data

类型 | train | dev | test
---|---|---|---|
[比赛](http://ai.baidu.com/broad/download?dataset=dureader) | 27W| 1W | 2W |
[开放](http://ai.baidu.com/broad/download) | 20W | 1W | 1W

## Performance
### Score(Public Board)

Model | Rouge-L | Bleu-4
---|---|---
BiDAF(cuDNN based) | 46.56 | 40.95
R-Net | 42.09 | 41.1
BiDAF+Self Attention | 47.28 | 41.3
BiDAF+Self Attention+Gated RNN | 47.71 | 41.75

### Memory and Time
i7-7700k + 32G RAM + GTX1080Ti  
batch size=32 dropout=0.7

Model | GPU Memory | Time(50 batch) | word embedding trainable
---|---|---|---
BiDAF(origin)| 8431M | 47s | false
MLSTM | 10655M | 1min27s | false
R-Net | 4295M | 23s | false
BiDAF+Self Attention(cuDNN based) | 8431M | 22s | false
BiDAF+Self Attention+Gated RNN(Pre) | N/A | N/A | false

## BUG：
1. BiDAF+Self Attention无法保存后再加载模型，tensorflow的cuDNN_LSTM虽然极快，但太难用了
2. R-Net本地的两个指标极差，提交的结果倒是正常

## Other
- 实际还有基于HKUST的BiDAF版本，显存和时间占用略小于R-Net，但效果比BiDAF(origin)大约低2个点，可能是使用了GRU的原因
- 最终在训练时无法保存最优模型的情况下，只能针对当前最优epoch进行一次predict，极为耗时
- 这个repo的Self Attention加在了match layer，后来发现cs224n的做法是基于match layer的输出做Self Attention，估计效果更好
