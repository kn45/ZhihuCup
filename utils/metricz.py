# -*- coding=utf8 -*-
# implement by https://biendata.com/competition/zhihu/evaluation/
import math


def eval_score(p_vals, t_vals):
    def _precision(p_vals, t_vals):
        sample_num = float(len(t_vals))
        hit_label_at_pos_num = [0., 0., 0., 0., 0.]  # hit @ each pos
        for p_val, t_val in zip(p_vals, t_vals):
            for pos, pred in enumerate(p_val):
                if pred in t_val:  # hit
                    hit_label_at_pos_num[pos] += 1
        prec = 0.
        for pos, hit_num in enumerate(hit_label_at_pos_num):
            prec += ((hit_num / sample_num)) / math.log(2.0 + pos)
        return prec

    def _recall(p_vals, t_vals):
        hit_label_num = 0.  # hit count
        true_label_num = 0.
        for p_val, t_val in zip(p_vals, t_vals):
            true_label_num += len(t_val)
            for pos, pred in enumerate(p_val):
                if pred in t_val:  # hit
                    hit_label_num += 1
        return hit_label_num / true_label_num

    prec = _precision(p_vals, t_vals)
    rec = _recall(p_vals, t_vals)
    if (prec + rec) == 0.:
        score = 0.
    else:
        score = (prec * rec) / (prec + rec)
    return prec, rec, score
