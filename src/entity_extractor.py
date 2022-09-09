# -*- coding: utf-8 -*-

import codecs as cs
import pickle
from gstore import GetRelations_2hop, GetRelationNum
import thulac


def features_from_two_sequences(s1, s2):
    # overlap
    overlap = len(set(s1)&(set(s2)))
    # 集合距离
    jaccard = len(set(s1)&(set(s2))) / len(set(s1)|(set(s2)))
    # 词向量相似度
    # wordvecsim = model.similarity(''.join(s1),''.join(s2))
    return [overlap, jaccard]


class SubjectExtractor(object):
    def __init__(self):
        self.mention2entity_dic = pickle.load(open('../data/mention2entity_dic.pkl', 'rb'))
        try:                
            self.entity2hop_dic = pickle.load(open('../data/entity2hop_dic.pkl', 'rb'))
        except:
            self.entity2hop_dic = {}
        self.word_2_frequency = self.load_word_to_index('../data/SogouLabDic.dic')
        self.entity2pop_dict = pickle.load(open('../data/entity2pop.pkl', 'rb'))

        self.not_pos = {'f','d','h','k','r','c','p','u','y','e','o','g','w','m'}  # 'q','mq','v','a','t',
        self.segger = thulac.thulac()
        self.pass_mention_dic = {'是什么','在哪里','哪里','什么','提出的','有什么','国家','哪个','所在',
                                 '培养出','为什么','什么时候','人','你知道','都包括','是谁','告诉我','又叫做','有','是'}
        print('entity extractor loaded')
    
    def load_word_to_index(self, path):
        dic = {}
        with cs.open(path, 'r', 'utf-8') as fp:
            lines = fp.read().split('\n')[:-1]
            for line in lines:
                line = line.strip()
                token = line.split('\t')[0]
                f = int(line.split('\t')[1])//10000
                dic[token] = f
        return dic
    
    def get_mention_feature(self, question, mention):
        f1 = float(len(mention))  # mention的长度
        try:
            f2 = float(self.word_2_frequency[mention])  # mention的tf/10000
        except:
            f2 = 1.0
        if mention[-2:] == '大学':
            f2 = 1.0
        try:
            f3 = float(question.index(mention))
        except:
            f3 = 3.0
            #print ('这个mention无法提取位置')
        return [mention, f1, f2, f3]

    def compute_entity_features(self, question, entity, relations):
        """
        抽取每个实体或属性值2hop内的所有关系，来跟问题计算各种相似度特征
        input:
            question: python-str
            entity: python-str <entityname>
            relations: python-dic key:<rname>
        output：
            [word_overlap,char_overlap,word_embedding_similarity,char_overlap_ratio]
        """
        # 得到主语-谓词的tokens及chars
        p_tokens = []
        for p in relations:
            p_tokens.extend(self.segger.cut(p[1:-1]))
        p_tokens = [token[0] for token in p_tokens]
        p_chars = [char for char in ''.join(p_tokens)]

        q_tokens = self.segger.cut(question)
        q_tokens = [token[0] for token in q_tokens]
        q_chars = [char for char in question]

        e_tokens = self.segger.cut(entity[1:-1])
        e_tokens = [token[0] for token in e_tokens]
        e_chars = [char for char in entity[1:-1]]

        qe_feature = features_from_two_sequences(q_tokens, e_tokens) + features_from_two_sequences(q_chars, e_chars)
        qr_feature = features_from_two_sequences(q_tokens, p_tokens) + features_from_two_sequences(q_chars, p_chars)
        # 实体名和问题的overlap除以实体名长度的比例
        return qe_feature + qr_feature

    def extract_subject(self, entity_mentions, subject_props, question):
        """
        根据前两部抽取出的实体mention和属性值mention，得到候选主语实体
        Input:
            entity_mentions: {str:list} {'贝鲁奇':['贝鲁奇',1,1,1]}
            subject_props: {str:list} {'1997-02-01':['1997年2月1日',1,1,1]}
        output:
            candidate_subject: {str:list}
        """
        candidate_subject = {}
        
        for mention in entity_mentions:  # 遍历每一个mention
            # 过滤词性明显不对的mention
            poses = self.segger.cut(mention)
            if len(poses) == 1 and poses[0][1] in self.not_pos:
                continue
            # 过滤停用词
            if mention in self.pass_mention_dic:
                continue
            if mention in self.mention2entity_dic:  # 如果它有对应的实体
                # mention的特征
                mention_features = self.get_mention_feature(question, mention)
                for entity in self.mention2entity_dic[mention]:
                    # 得到实体两跳内的所有关系
                    entity = '<'+entity+'>'
                    relations = self.entity2hop_dic.get(entity, {})

                    # 计算问题和主语实体及其两跳内关系间的相似度
                    similar_features = self.compute_entity_features(question, entity, relations)

                    # 实体的流行度特征
                    if entity in self.entity2pop_dict:
                        popular_feature = self.entity2pop_dict[entity]
                    else:
                        popular_feature = GetRelationNum(entity)
                    candidate_subject[entity] = mention_features + similar_features + [popular_feature ** 0.5]
        return candidate_subject
