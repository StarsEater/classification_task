import os
from configparser import ConfigParser
from random import shuffle
import random
import sys
import collections
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import logging
from tools import *
random.seed(2200)
def showstatic_oneLabel(train_set,dev_set,test_set):
    def static_one_set(name,one_set):
        print("count from %s set"%name)
        c = collections.Counter([x["label"][0] for x in one_set])
        for k,v in c.items():
            print(k,"->",v)
    static_one_set("train",train_set)
    static_one_set("valid",dev_set)
    static_one_set("test",test_set)
def dataset_merge_shuffle(file_path,save_path,split_ratio=[0.9,0.1,0.1]):
    total_dataset = readJson(file_path,lines=True)
    shuffle(total_dataset)

    r1,r2,r3 = split_ratio
    assert r1+r2+r3==1

    n = len(total_dataset)
    trainset = total_dataset[:round(r1*n)]
    devset = total_dataset[round(r1*n):round((r1+r2)*n)]
    testset = total_dataset[round((r1+r2)*n):]

    train_path = os.path.join(save_path, "trainset")
    dev_path = os.path.join(save_path, "devset")
    test_path =os.path.join(save_path, "testset")

    saveJson(trainset,train_path,lines=True)
    saveJson(devset,dev_path,lines=True)
    saveJson(testset,test_path,lines=True)
    print("save it ",train_path)
    # 针对多分类但标签的统计，将标签列表的第一个作为标签统计
    showstatic_oneLabel(trainset,devset,testset)

if __name__ == '__main__':
    config = ConfigParser()
    config.read(sys.argv[1],encoding='utf-8')

    type_file_path = config['samples_generate']['type_save_path']
    type_save_path = config['dataset_split']['type_dataset_save_path']
    split_ratio = config['dataset_split']['split_ratio'].split(',')
    split_ratio = [float(item) for item in split_ratio]
    checkFileOMake(type_save_path)
    dataset_merge_shuffle(type_file_path,type_save_path,split_ratio)

    degree_file_path = config['samples_generate']['degree_save_path']
    degree_save_path = config['dataset_split']['degree_dataset_save_path']
    split_ratio = config['dataset_split']['split_ratio'].split(',')
    split_ratio = [float(item) for item in split_ratio]
    checkFileOMake(degree_save_path)
    dataset_merge_shuffle(degree_file_path, degree_save_path, split_ratio)