# -*- coding: utf-8 -*-


import pickle
import numpy as np
from mention_extractor import MentionExtractor
from entity_extractor import SubjectExtractor
from tuple_extractor import TupleExtractor
from gstore import SearchAnsChain
import re
import thulac


class AnswerByPkubase(object):
    def __init__(self,):
        self.me = MentionExtractor()
        self.se = SubjectExtractor()
        self.te = TupleExtractor()
        self.topn_e = 5
        self.topn_t = 3

        self.subject_classifier_model = pickle.load(open('../model/entity_classifier_model.pkl', 'rb'))
        self.segger = thulac.thulac()

    def subject_filter(self, subjects):
        """
        输入候选主语和对应的特征，使用训练好的模型进行打分，排序后返回前topn个候选主语
        """
        entitys = []
        features = []
        for s in subjects:
            entitys.append(s)
            features.append(subjects[s][1:])
        prepro = self.subject_classifier_model.predict_proba(np.array(features))[:, 1].tolist()
        sample_prop = [each for each in zip(prepro, entitys)]  # (prop,(tuple))

        sample_prop = sorted(sample_prop, key=lambda x: x[0], reverse=True)
        entitys = [each[1] for each in sample_prop if each[0] > 0.05]


        if len(entitys) > self.topn_e:
            predict_entitys = entitys[:self.topn_e]
        else:
            predict_entitys = entitys
        new_subjects = {}
        for e in predict_entitys:
            new_subjects[e] = subjects[e]
        return new_subjects
    
    def tuple_filter(self, tuples):
        """
        使用训练好的模型得分，排序后返回前topn个候选答案
        """
        sample_prop = sorted(tuples.items(), key=lambda x: x[1], reverse=True)
        tuples_sorted = [each for each in sample_prop]
        if len(tuples_sorted) > self.topn_t:
            tuples_sorted = tuples_sorted[:self.topn_t]
        return tuples_sorted

    def get_most_overlap_tuple(self, question, tuples):
        # 从排名前几的tuples里选择与问题overlap最多的
        max_ = 0
        ans = tuples[0]
        for t in tuples:
            text = ''
            for element in t[0]:
                # element = element[1:-1].split('_')[0]
                element = element[1:-1]
                text = text + element
            f2 = len(set(text).intersection(set(question)))/len(set(text))
            f = f2
            if f > max_:
                ans = t[0]
                max_ = f
        return ans
    
    def answer_main(self, question):
        """
        输入问题，依次执行：
        抽取实体mention、抽取属性值、生成候选实体并得到特征、候选实体过滤、生成候选查询路径（单实体双跳）、候选查询路径过滤
        使用top1的候选查询路径检索答案并返回
        input:
            question : python-str
        output:
            answer : python-list, [str]
        """
        dic = {}
        question = re.sub('原名', '中文名', question)
        question = re.sub('英文', '外文', question)
        question = re.sub('英语', '外文', question)
        dic['question'] = question
        print('====question====')
        print(question)

        mentions = self.me.extract_mentions(question)
        dic['mentions'] = mentions

        subject_props = {}
        subjects = self.se.extract_subject(mentions, subject_props, question)
        dic['subjects'] = subjects
        if len(subjects) == 0:
            return (), []

        subjects = self.subject_filter(subjects)
        dic['subjects_filter'] = subjects
        if len(subjects) == 0:
            return (), []

        tuples = self.te.extract_tuples(subjects, question)
        dic['tuples'] = tuples
        if len(tuples) == 0:
            return (), []

        tuples = self.tuple_filter(tuples)  # 得到top1的单实体问题tuple
        dic['tuples_filter'] = tuples

        tuples = [tuple for tuple in tuples if tuple[1] > 0.5]
        if not tuples:
            return (), []
        top_tuple = self.get_most_overlap_tuple(question, tuples)

        print('====最终候选查询路径为====')
        print(top_tuple)
        
        e1 = top_tuple[0]
        r1 = top_tuple[1]
        r2 = None
        if len(top_tuple) > 2:
            r2 = top_tuple[2]
        predict_ans = SearchAnsChain(e1, r1, r2)
        print('====最终答案为====')
        print(predict_ans)


if __name__ == '__main__':
    ansbot = AnswerByPkubase()
    question_list = [
        '巴金的中文名是什么',
        '刻舟求剑这个故事发生在哪个年代',
        '中国民权保障同盟是谁发起的'
    ]
    for question in question_list:
        ansbot.answer_main(question)

