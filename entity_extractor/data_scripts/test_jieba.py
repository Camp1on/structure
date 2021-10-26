# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 23:36
# @Author  : heyee (jialeyang)

import jieba
import jieba.posseg as pseg

jieba.load_userdict("/Users/apple/XHSworkspace/data/structure/food/config/type_9_1/jieba_food_test")

line = '5f8bae940000000001006577/t /t 三里屯附近大骨头南小蟹/t /t 位置:三里屯soho地下/t 人均:7080/t 大颗粒煮面蟹黄超级多,吃多了会有些腻,最后点一些蔬菜/t 招牌蟹黄面68元/t 小吃都在15到20左右/t 经济实惠,幸福感满满'
line_txt_list = ""
words = pseg.cut(line)
for word, flag in words:
    if flag in ['工艺', '功效', '食品', '食材', '口味', '工具', '适宜人群', '品牌', '美食概念词', '否定修饰']:
        line_txt_list += word + "_[" + flag + "]_"
print(''.join(line_txt_list))
print("done")