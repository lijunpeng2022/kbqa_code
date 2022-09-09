# -*- coding: UTF-8 -*-
"""
# Filename: gstore.py
# Author: Junpeng Li
# Description: a simple GET-example of python API
"""
import sys
from gconn import GstoreConnector
import json

# before you run this example, make sure that you have started up ghttp service (using bin/ghttp port)
# "GET" is a default parameter that can be omitted
IP = "10.1.31.20"
Port = 9000
username = "root"
password = "123456"
KG_NAME="pkubase"
FORMAT="json"

gc =  GstoreConnector(IP, Port, username, password)


def GetQueryResult(q_sparql):
    res = gc.query(KG_NAME, FORMAT, q_sparql)
    res = json.loads(res)
    bindings = res.get('results', {}).get('bindings', [])
    return bindings


def GetTuplesSingle(entity):
    '''根据实体名，得到所有1跳关系和对应尾实体列表'''
    q_sparql = 'select distinct ?r ?t where { ' + entity +' ?r ?t. }'
    relation_list = []
    entity_list = []
    bindings = GetQueryResult(q_sparql)
    for binding in bindings:
        relation = '<' + binding.get('r', {}).get('value', '') + '>'

        tail_type = binding.get('t', {}).get('type', '')
        tail = binding.get('t', {}).get('value', '')
        if relation == '<类型>':
            continue
        relation_list.append(relation)
        entity_list.append((tail_type, tail))
    return relation_list, entity_list


def GetRelationPathsSingle(entity):
    '''根据实体名，得到所有1跳关系路径'''
    relation_list, _ = GetTuplesSingle(entity)
    return [[relation] for relation in set(relation_list)]


def GetRelationPaths(entity):
    '''根据实体名，得到所有2跳内的关系路径，用于问题和关系路径的匹配'''
    relation1_list, tail_list = GetTuplesSingle(entity)
    all_relation_list = [[relation1] for relation1 in relation1_list]
    two_hop_relation_set = set()
    for relation1, tail_with_type in zip(relation1_list, tail_list):
        if tail_with_type[0] == 'literal':
            continue
        relation2_list, _ = GetTuplesSingle('<' + tail_with_type[1] + '>')
        for relation2 in relation2_list:
            two_hop_relation_set.add(relation1 + '\t' + relation2)
    for two_hop_relation in two_hop_relation_set:
        relation1, relation2 = two_hop_relation.split('\t')
        all_relation_list.append([relation1, relation2])
    return all_relation_list


def GetRelations_2hop(entity):
    '''根据实体名，得到两跳内的所有关系字典，用于问题和实体子图的匹配'''
    rpaths2 = GetRelationPaths(entity)
    dic = {}
    for rpath in rpaths2:
        for r in rpath:
            dic[r] = 0
    return dic


def GetTwoEntityTuple(e1, r1, e2):
    q_sparql = 'select distinct ?r2 where { ' + e1 + ' ' + r1 + '?x. ?x ?r2' + e2 + '. }'
    tuples = []
    bindings = GetQueryResult(q_sparql)
    for binding in bindings:
        relation = binding.get('r2', {}).get('value', '')
        tuples.append(tuple([e1, r1, '<' + relation + '>', e2]))
    return tuples


def SearchAnsChain(e, r1, r2=None):
    '''对于链式问题，e-r-ans或e-r1-r2-ans，根据最终的实体和关系查询结果'''
    ans = []
    if not r2:
        q_sparql = 'select ?e2 where { ' + e + ' ' + r1 + '?e2. }'
    else:
        q_sparql = 'select distinct ?e2 where { ' + e + ' ' + r1 + ' ?t. ?t ' + r2 + ' ?e2. }'
    bindings = GetQueryResult(q_sparql)
    for binding in bindings:
        tail = binding.get('e2', {}).get('value', '')
        ans.append(tail)
    return ans


def GetRelationNum(entity):
    '''根据实体名，得到与之相连的关系数量，代表实体在知识库中的流行度'''
    q_sparql = 'select (count(?r) as ?entity_pop) where { ' + entity + ' ?r ?t. }'
    ans = 0
    bindings = GetQueryResult(q_sparql)
    if bindings:
        ans = int(bindings[0].get('entity_pop', {}).get('value', '0'))
    return ans


def main():
    # print(GetTuplesSingle('<迪丽热巴迪力木拉>'))
    # print(GetRelationPathsSingle('<迪丽热巴迪力木拉>'))
    # print(GetRelations_2hop('<迪丽热巴迪力木拉>'))
    # print(GetTwoEntityTuple('<迪丽热巴迪力木拉>', '<毕业院校>', '<胡歌_（中国内地男演员）>'))
    # print(SearchAnsChain('<迪丽热巴迪力木拉>', '<毕业院校>'))
    # print(SearchAnsChain('<迪丽热巴迪力木拉>', '<毕业院校>', '<知名校友>'))
    print(GetRelationNum('<迪丽热巴迪力木拉>'))

if __name__ == "__main__":
    main()
