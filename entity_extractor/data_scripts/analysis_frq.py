# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 20:44
# @Author  : heyee (jialeyang)

# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 15:50
# @Author  : heyee (jialeyang)

import os.path as osp
import os
from collections import defaultdict
import json
from eval_performance import *
from tqdm import tqdm
import pandas as pd
from collections import Counter

tqdm.pandas()


class AnalysisFrq:

    def __init__(self):
        self.frq_dict = defaultdict()

    def count_frq(self, data):
        res = Counter(data)
        res_dict = dict(res)
        res_dict2 = dict(sorted(res_dict.items(), key=lambda item: item[1], reverse=True))
        # out_file = ''
        # fout = open(out_file, 'w', encoding='utf-8')
        # json.dump(res_dict2, fout, ensure_ascii=False, indent=4)
        with open('/Users/apple/XHSworkspace/data/structure/food/dataset/frq_ana/笔记query_frq_ana.json', 'w', encoding='utf-8') as fp:
            json.dump(res_dict2, fp, ensure_ascii=False, indent=4)
        print(data)

    def read_file(self, file):
        """ file format:   note_id \t\t property \t\t note
        """
        with open(file, 'r', encoding='utf-8') as reader:
            tmp = reader.read()
            lines = tmp.split("\n")
        return [w for w in lines if len(w) > 0]

    def write_file(self, file, data):
        with open(file, 'w', encoding='utf-8') as writer:
            for parsed_text in data:
                if isinstance(parsed_text, str) and len(parsed_text) > 0:
                    writer.write(f'{parsed_text}\n')
                else:
                    print(parsed_text)


if __name__ == '__main__':
    ana = AnalysisFrq()
    dir_path = "/Users/apple/XHSworkspace/data/structure/food/dataset/frq_ana"
    data = ana.read_file(
        osp.join(dir_path, "000000_0_res2"))
    # data = pd.read_csv(osp.join(dir_path, "美食query.csv_10000_res"),
    #                    error_bad_lines=False,
    #                    header=None)
    data2 = []
    for i in data:
        data2.extend(i.split("\t"))
    ana.count_frq(data=data2)
    print("hi")
