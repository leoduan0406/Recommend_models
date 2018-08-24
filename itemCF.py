#-*- coding:utf-8 -*-
"""
@author:duan
Created on:2018/8/24 21:12
"""
# -*- coding: utf-8 -*-
'''
Created on 2017年9月18日
@author: Jason.F
'''

import math
import random
import os
from itertools import islice


class ItemBasedCF:
    def __init__(self, datafile=None):
        self.datafile = datafile
        self.readData()
        self.splitData()

    def readData(self, datafile=None):
        self.datafile = datafile or self.datafile
        self.data = []
        file = open(self.datafile, 'r')
        for line in islice(file, 1, None):  # file.readlines():
            userid, itemid, record ,timestamp = line.split(',')
            self.data.append((userid, itemid, float(record)))

    def splitData(self, data=None, k=3, M=10, seed=10):
        self.testdata = {}
        self.traindata = {}
        data = data or self.data
        random.seed(seed)  # 生成随机数
        for user, item, record in self.data:
            self.traindata.setdefault(user, {})
            self.traindata[user][item] = record  # 全量训练
            if random.randint(0, M) == k:  # 测试集
                self.testdata.setdefault(user, {})
                self.testdata[user][item] = record

    def ItemSimilarity(self, train=None):
        train = train or self.traindata
        self.itemSim = dict()
        item_user_count = dict()  # item_user_count{item: likeCount} the number of users who like the item
        count = dict()  # count{i:{j:value}} the number of users who both like item i and j
        for user, item in train.items():  # initialize the user_items{user: items}
            for i in item.keys():
                item_user_count.setdefault(i, 0)
                item_user_count[i] += 1
                for j in item.keys():
                    if i == j:
                        continue
                    count.setdefault(i, {})
                    count[i].setdefault(j, 0)
                    count[i][j] += 1
        for i, related_items in count.items():
            self.itemSim.setdefault(i, dict())
            for j, cuv in related_items.items():
                self.itemSim[i].setdefault(j, 0)
                self.itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j] * 1.0)

    def recommend(self, user, train=None, k=10, nitem=5):
        train = train or self.traindata
        rank = dict()
        ru = train.get(user, {})
        for i, pi in ru.items():
            for j, wj in sorted(self.itemSim[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if j in ru:
                    continue
            rank.setdefault(j, 0)
            rank[j] += pi * wj

        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitem])

    def recallAndPrecision(self, train=None, test=None, k=8, nitem=5):
        train = train or self.traindata
        test = test or self.testdata
        hit = 0
        recall = 0
        precision = 0
        for user in test.keys():
            tu = test.get(user, {})
            rank = self.recommend(user, train=train, k=k, nitem=nitem)
            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        return (hit / (recall * 1.0), hit / (precision * 1.0))

    def coverage(self, train=None, test=None, k=8, nitem=5):
        train = train or self.traindata
        test = test or self.testdata
        recommend_items = set()
        all_items = set()
        for user in test.keys():
            for item in test[user].keys():
                all_items.add(item)
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)

    def popularity(self, train=None, test=None, k=8, nitem=5):
        train = train or self.traindata
        test = test or self.testdata
        item_popularity = dict()
        for user, items in train.items():
            for item in items.keys():
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in test.keys():
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / (n * 1.0)

    def testRecommend(self, user):
        rank = self.recommend(user, k=10, nitem=5)
        for i, rvi in rank.items():
            items = self.traindata.get(user, {})
            record = items.get(i, 0)
            print("%5s: %.4f--%.4f" % (i, rvi, record))


if __name__ == "__main__":

    ibc = ItemBasedCF(os.getcwd() + '\\ratings.csv')  # 初始化数据
    ibc.ItemSimilarity()  # 计算物品相似度矩阵
    ibc.testRecommend(user="345")  # 单用户推荐
    print("Hello...")
    print("%3s%20s%20s%20s%20s" % ('K', "recall", 'precision', 'coverage', 'popularity'))
    for k in [5, 10, 15, 20]:
        recall, precision = ibc.recallAndPrecision(k=k)
        coverage = ibc.coverage(k=k)
        popularity = ibc.popularity(k=k)
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % (k, recall * 100, precision * 100, coverage * 100, popularity))
