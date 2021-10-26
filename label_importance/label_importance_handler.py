import copy

class LabelImportanceHandler:

    def __init__(self):
        pass

    def predict(self, sents, entity_rst, relation_rst):
        entity_rst = copy.deepcopy(entity_rst)
        relation_rst = copy.deepcopy(relation_rst)
        entity_importance_rst = []
        for i in entity_rst:
            i.update({'imp': 0.9})
            entity_importance_rst.append(i)

        relation_importance_rst = []
        for i in relation_rst:
            i.update({'imp': 0.9})
            relation_importance_rst.append(i)

        return entity_importance_rst, relation_importance_rst

