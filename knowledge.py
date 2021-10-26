import json
import random
from collections import defaultdict
from structure.util import Singleton


@Singleton
class Knowledge:

    def __init__(self, file):
        with open(file, 'r', encoding='utf-8') as fin:
            self.knowledge = json.load(fin)
        self.COMMON_CATE = '通用'
        self._copy_common(self.knowledge)

    def _copy_common(self, data):
        values = list(data.values())
        if len(values) > 0 and type(values[0]) == [].__class__:
            if self.COMMON_CATE not in self.knowledge:
                return
            for p in self.knowledge[self.COMMON_CATE]:
                if p not in data:
                    data[p] = []
                data[p].extend(self.knowledge[self.COMMON_CATE][p])
            return
        for key in data:
            if type(data[key]) == {}.__class__:
                self._copy_common(data[key])

    def _get_cate_knowledge(self, know, cate):
        for key in know:
            if key == cate and type(know[key]) == {}.__class__:
                return know[key]
            elif type(know[key]) == {}.__class__:
                rst = self._get_cate_knowledge(know[key], cate)
                if rst is not None:
                    return rst
        return None

    '''
    获取特定类目的知识
    '''
    def get_cate_knowledge(self, cate):
        return self._get_cate_knowledge(self.knowledge, cate)

    def _get_property(self, know, rst):
        if know is None:
            return
        for key in know:
            if type(know[key]) == [].__class__:
                rst[key] = list(set(rst[key]).union(set(know[key])))
            if type(know[key]) == {}.__class__:
                self._get_property(know[key], rst)

    '''
    获取特定类目的属性和属性值对
    '''
    def get_cate_property(self, cate):
        cate_know = self.get_cate_knowledge(cate)
        rst = defaultdict(list)
        self._get_property(cate_know, rst)
        return rst

    '''
    获取所有的属性和属性值对
    '''
    def get_all_property(self):
        rst = defaultdict(list)
        self._get_property(self.knowledge, rst)
        return rst

    def _get_value(self, know):
        if know is None:
            return None
        properties = defaultdict(list)
        rst = set()
        self._get_property(know, properties)
        for key in properties:
            rst = rst.union(properties[key])
        return list(rst)

    '''
    获取所有的属性值
    '''
    def get_all_value(self):
        return self._get_value(self.knowledge)

    '''
    获取特定属性的所有属性值
    '''
    def get_property_value(self, prop):
        all_cate_property = self.get_all_property()
        if prop in all_cate_property:
            return all_cate_property[prop]
        return None

    '''
    获取特定类目下的所有属性值
    '''
    def get_cate_value(self, cate):
        cate_knowledge = self.get_cate_knowledge(cate)
        return self._get_value(cate_knowledge)


if __name__ == '__main__':
    file = '/data/model/note_structure/knowledge/20210811.json'
    knowledge = Knowledge(file)
    p = knowledge.get_all_property()
    keys = p.keys()
    for i in range(10):
        pp = random.sample(keys, 2)
