# -*- coding: utf-8 -*-
# @Time    : 2021/8/27 19:22
# @Author  : heyee (jialeyang)
import itertools
import json
from tqdm import tqdm
from collections import defaultdict

tqdm.pandas()


def delete(data, p, v):
    for key in data:
        if key == p and type(data[key]) == [].__class__:
            if v in data[key]:
                data[key].remove(v)
        if type(data[key]) == {}.__class__:
            delete(data[key], p, v)


def add(data, p, v):
    for key in data:
        if key == p and type(data[key]) == [].__class__:
            if v not in data[key]:
                data[key].append(v)
        if type(data[key]) == {}.__class__:
            add(data[key], p, v)


SEARCH_COLLECTION = []


def search_all_type(data, p):
    for key in data:
        if key == p and type(data[key]) == [].__class__:
            SEARCH_COLLECTION.extend(data[key])
        if type(data[key]) == {}.__class__:
            search_all_type(data[key], p)


def empty_type(data, p):
    for key in data:
        if key == p and type(data[key]) == [].__class__:
            data[key] = []
        if type(data[key]) == {}.__class__:
            empty_type(data[key], p)


def generate_data_pair(data_list, cate):
    pv = []
    for i in data_list:
        pv.append((cate, i))
    return pv


def add_data_from_file(file, cate, merge_data=[], cate_0=None):
    with open(file, 'r', encoding='utf-8') as reader:
        tmp = reader.read()
        lines = tmp.split("\n")[:-1]
    if len(merge_data) > 0:
        lines.extend(merge_data)
        lines = list(set(lines))
    pv = generate_data_pair(data_list=lines, cate=cate)
    for (p, v) in tqdm(pv):
        if cate_0 is not None:
            add(org[cate_0], p, v)
        else:
            add(org, p, v)


def add_data_from_data(cate, data, merge_data, cate_0=None):
    if len(merge_data) > 0:
        data.extend(merge_data)
    pv = generate_data_pair(data_list=list(set(data)), cate=cate)
    for (p, v) in tqdm(pv):
        if cate_0 is not None:
            add(org[cate_0], p, v)
        else:
            add(org, p, v)


def list_intersection(listA, listB):
    """求交集
    """
    return list(set(listA).intersection(set(listB)))


def list_difference(listA, listB):
    """求差集，在A中但不在B中
    """
    return list(set(listA).difference(set(listB)))


def list_union(listA, listB):
    """求并集
    """
    return list(set(listA).union(set(listB)))


def write_file(file, data):
    with open(file, 'w', encoding='utf-8') as writer:
        for parsed_text in data:
            if isinstance(parsed_text, str) and len(parsed_text) > 0:
                writer.write(f'{parsed_text}\n')
            else:
                print(parsed_text)


# with open("/Users/apple/XHSworkspace/data/structure/food/tagging/food_20210901_pass", 'r', encoding='utf-8') as reader:
#     tmp = reader.read()
#     lines = tmp.split("\n")
#     lines = list(set(lines))
#     # lines.sort(key=lambda s: len(s))
#     write_file(data=lines,
#                file='/Users/apple/XHSworkspace/data/structure/food/tagging/food_20210901_pass')


# collect_parsed_res = json.load(
#     open('/Users/apple/XHSworkspace/data/structure/food/config/collect_parsed_res.json', 'r', encoding='utf-8'))
org = json.load(
    open('/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/red_tree_food_concept_v28.json', 'r', encoding='utf-8'))

""" 美食 """
# for key, value in collect_parsed_res[0].items():
#     search_all_type(data=org["美食"], p=key)
#     empty_type(data=org["美食"], p=key)
#     add_data_from_data(cate=key, cate_0="美食", data=value, merge_data=list(set(SEARCH_COLLECTION)))
#     print(key)
#     SEARCH_COLLECTION = []

res = defaultdict(list)
select_list = ['工艺', '功效', '食品', '食材', '口味', '工具', '适宜人群', '品牌', '美食概念词', '否定修饰']
# cate_0 = "美食"
for i in select_list:
    search_all_type(data=org, p=i)
    # SEARCH_COLLECTION.sort(key=lambda s: len(s))
    SEARCH_COLLECTION.sort(key=lambda item: (-len(item), item[-1]), reverse=True)
    res[i] = SEARCH_COLLECTION
    print(i + ":" + str(len(SEARCH_COLLECTION)))
    write_file(data=SEARCH_COLLECTION,
               file='/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v8.json' + "_" + i)
    SEARCH_COLLECTION = []
comb = list(itertools.combinations(select_list, 2))
for i in comb:
    print(list_intersection(res[i[0]], res[i[1]]))
    print(str(i) + ":" + str(
        0 if len(list_difference(res[i[0]], res[i[1]])) == 0 else len(list_intersection(res[i[0]], res[i[1]])) / len(list_difference(res[i[0]], res[i[1]]))))
        # len(list_intersection(res[i[0]], res[i[1]])) / len(res[i[0]])))
    print(str(i) + ":" + str(
        0 if len(list_difference(res[i[1]], res[i[0]])) == 0 else len(list_intersection(res[i[0]], res[i[1]])) / len(list_difference(res[i[1]], res[i[0]]))))
        # len(list_intersection(res[i[0]], res[i[1]])) / len(res[i[1]])))
    # print(list_intersection(res[i[0]], res[i[1]])[:20])
    # print(list_union(res[i[0]], res[i[1]])[:20])
res_len = []
for i in res["食品"]:
    res_len.append(len(i))

from collections import Counter

result = Counter(res_len)

# import matplotlib.pyplot as plt
# import numpy as np
#
# data_range = [0, max(res_len)]  # x轴范围
# bins = np.arange(data_range[0], data_range[1], 1)
#
# res_plot = defaultdict(list)
# for i in bins:
#
# # 其中的1，是每个bin的宽度。这个值取小，可以提升画图的精度
# plt.hist(bins, color='blue', alpha=0.5)  # alpha设置透明度，0为完全透明
#
# plt.xlabel('scores')
# plt.ylabel('count')
# plt.xlim(data_range[0], data_range[1])  # 设置x轴分布范围
#
# plt.show()

cate_0 = "美食"
cate = "菜品口味"
search_all_type(data=org[cate_0], p=cate)
print(len(org[cate_0][cate]))
empty_type(data=org[cate_0], p=cate)
add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_口味",
                   cate_0=cate_0,
                   cate=cate,
                   merge_data=list(set(SEARCH_COLLECTION)))
print(len(org[cate_0][cate]))
SEARCH_COLLECTION = []

cate = "品牌"
search_all_type(data=org[cate_0], p=cate)
print(len(org[cate_0][cate]))
empty_type(data=org[cate_0], p=cate)
add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_品牌",
                   cate_0=cate_0,
                   cate=cate,
                   merge_data=list(set(SEARCH_COLLECTION)))
print(len(org[cate_0][cate]))
SEARCH_COLLECTION = []

cate = "水果"
search_all_type(data=org[cate_0], p=cate)
print(len(org[cate_0][cate]))
empty_type(data=org[cate_0], p=cate)
add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_水果",
                   cate_0=cate_0,
                   cate=cate,
                   merge_data=list(set(SEARCH_COLLECTION)))
print(len(org[cate_0][cate]))
SEARCH_COLLECTION = []

cate = "食品"
search_all_type(data=org[cate_0], p=cate)
print(len(org[cate_0][cate]))
empty_type(data=org[cate_0], p=cate)
add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_菜名",
                   cate_0=cate_0,
                   cate=cate,
                   merge_data=list(set(SEARCH_COLLECTION)))
print(len(org[cate_0][cate]))
SEARCH_COLLECTION = []

cate = "食材"
search_all_type(data=org[cate_0], p=cate)
print(len(org[cate_0][cate]))
empty_type(data=org[cate_0], p=cate)
add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_食材",
                   cate_0=cate_0,
                   cate=cate,
                   merge_data=list(set(SEARCH_COLLECTION)))
print(len(org[cate_0][cate]))
SEARCH_COLLECTION = []

cate = "主要工艺"
search_all_type(data=org[cate_0], p=cate)
print(len(org[cate_0][cate]))
empty_type(data=org[cate_0], p=cate)
add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/food/tagging/food_mt_烹饪方式",
                   cate_0=cate_0,
                   cate=cate,
                   merge_data=list(set(SEARCH_COLLECTION)))
print(len(org[cate_0][cate]))
SEARCH_COLLECTION = []

# print(org)

""" 单例 """
# pv = [('功能', '搭配')]
# for (p, v) in pv:
#     delete(org, p, v)
# pv = [('流行元素', '垫肩')]
# for (p, v) in pv:
#     add(org, p, v)

""" 时尚/美妆 """
# """ 清空指定一级类目下的三级类目：case 清空 时尚 下的所有品牌"""
# empty_type(data=org["时尚"], p="品牌")
# add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/entity_total_note_0721.xlsx_brand_fashion_list",
#                    cate_0="时尚",
#                    cate="品牌")
# empty_type(data=org["美妆"], p="品牌")
# add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/entity_total_note_0721.xlsx_brand_beauty_list",
#                    cate_0="美妆",
#                    cate="品牌")
#
# """ 品类去重合并指定一级类目下的三级类目：case 去重合并 时尚 下的所有品牌 """
# search_all_type(data=org["时尚"], p="品类")
# empty_type(data=org["时尚"], p="品类")
# add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/entity_total_note_0721.xlsx_pinlei_fashion_list",
#                    cate_0="时尚",
#                    cate="品类",
#                    merge_data=list(set(SEARCH_COLLECTION)))
# SEARCH_COLLECTION = []
#
# search_all_type(data=org["美妆"], p="品类")
# empty_type(data=org["美妆"], p="品类")
# add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/entity_total_note_0721.xlsx_pinlei_fashion_list",
#                    cate_0="美妆",
#                    cate="品类",
#                    merge_data=list(set(SEARCH_COLLECTION)))
# SEARCH_COLLECTION = []
#
# search_all_type(data=org, p="时间")
# empty_type(data=org, p="时间")
# add_data_from_file(file="/Users/apple/XHSworkspace/data/structure/entity_total_note_0721.xlsx_time_list",
#                    cate="时间",
#                    merge_data=list(set(SEARCH_COLLECTION)))
# SEARCH_COLLECTION = []

out_file = '/Users/apple/XHSworkspace/data/structure/food/config/red_tree_food_v6.json'
fout = open(out_file, 'w', encoding='utf-8')
json.dump(org, fout, ensure_ascii=False, indent=4)
