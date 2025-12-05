# modules/metrics.py

# 文本生成指标，不依赖 sklearn
from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor import Meteor   # 没 Java，先关掉
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider.cider import Cider   # 新增 CIDEr


def compute_scores(gts, res):
    """
    使用 pycocoevalcap 计算生成报告的文本指标：
    - BLEU_1 ~ BLEU_4
    - ROUGE_L
    - CIDEr

    gts: {id: [ref_sentence1, ref_sentence2, ...]}
    res: {id: [gen_sentence]}
    """

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        # (Meteor(), "METEOR"),  # 如以后配好了 Java 再打开
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    eval_res = {}

    for scorer, method in scorers:
        # 有的实现带 verbose，有的没有，所以做一个兼容
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)

        if isinstance(method, list):
            # BLEU 返回 4 个分数
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

    return eval_res


def compute_mlc(gt, pred, label_set):
    """
    多标签分类指标（AUC、F1、Recall、Precision）
    注意：只有调用这个函数时才会 import sklearn，
    这样就不会影响你单独测试文本指标。
    """
    # ⬇⬇⬇ 把 sklearn 的导入挪到函数内部
    from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

    res_mlc = {}
    avg_aucroc = 0
    for i, label in enumerate(label_set):
        res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
        avg_aucroc += res_mlc['AUCROC_' + label]
    res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)

    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc
