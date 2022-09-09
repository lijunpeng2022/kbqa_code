# -*- coding: utf-8 -*-


from gstore import GetRelationPaths
from transformers import BertTokenizer, DataCollatorWithPadding
from transformers import BertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
import torch


class TupleExtractor(object):

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = BertForSequenceClassification.from_pretrained('../model/similarity_model')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device
        self.model.to(self.device)
        print('bert相似度匹配模型加载完成')
        print('tuples extractor loaded')
        
    def extract_tuples(self, candidate_entitys, question):
        candidate_tuples = []
        entity_list = candidate_entitys.keys()  # 得到有序的实体列表
        origin_question_list = []
        generated_question_list = []
        for entity in entity_list:
            # 得到该实体的所有关系路径
            relations = GetRelationPaths(entity)
            if not relations:
                continue
            mention = candidate_entitys[entity][0]
            for r in relations:
                this_tuple = tuple([entity] + r)  # 生成候选tuple
                predicates = [relation[1:-1] for relation in r]
                human_question = '的'.join([mention] + predicates)
                origin_question_list.append(question)
                generated_question_list.append(human_question)
                # inputs.append((question, human_question))
                candidate_tuples.append(this_tuple)

        def map_function(example):
            return self.tokenizer(example["sentence1"], example["sentence2"], truncation=True)

        tokenized_datasets = Dataset.from_dict({
            'sentence1': origin_question_list,
            'sentence2': generated_question_list
        }).map(map_function, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2"])
        tokenized_datasets.set_format('torch')
        dataloader = DataLoader(tokenized_datasets, batch_size=64, collate_fn=self.data_collator)
        prediction_result = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=-1)
            prediction_result += predictions.tolist()
        final_candidate_tuples = {}
        for single_tuple, score in zip(candidate_tuples, prediction_result):
            final_candidate_tuples[single_tuple] = score[1]
        return final_candidate_tuples
