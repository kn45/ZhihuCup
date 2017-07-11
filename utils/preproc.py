#!/usr/bin/env python
import copy
import numpy as np
import sys


class DictTable(object):
    def __init__(self, dict_file):
        self.table = {}
        self.rev_table = {}
        if isinstance(dict_file, basestring):
            with open(dict_file) as f:
                for line in f:
                    k, v = line.rstrip('\n').split('\t')
                    self.table[k] = int(v)
                    self.table[int(v)] = k
        if isinstance(dict_file, dict):
            self.table = copy.deepcopy(dict_file)
            for k in dict_file:
                self.rev_table[dict_file[k]] = k

    def lookup(self, words):
        ids = []
        for word in words:
            if word in self.table:
                ids.append(self.table[word])
            else:
                ids.append(None)
        return ids

    def lookup_rev(self, ids):
        words = []
        for idx in ids:
            if idx in self.rev_table:
                words.append(self.rev_table[idx])
            else:
                words.append(None)
        return words

def sparse2dense(ids, ndim):
    out = np.zeros((ndim), dtype=np.int32)
    for idx in ids:
        out[idx] = 1
    return out

def zero_padding(inp, seq_len):
    out = np.zeros((seq_len), dtype=np.int32)
    for i, v in enumerate(inp):
        if i >= seq_len:
            break
        out[i] = v
    return out

