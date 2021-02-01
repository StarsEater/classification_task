import logging
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import time
import re
import collections
import pandas as pd
from configparser import ConfigParser
from tools import *


def wash_data(text):
    # 去除年月日，去除双引号之前只有字母和数字的和包含有组、侧、区的片段
    text = re.sub("[0-9]{4}年[0-9]+月[0-9]+日|\“[a-zA-Z0-9]+\”|\“.*?[组|侧|区].*?\”", "", text)
    # 连续数字或者字母长度超过5的片段
    text = re.sub("[a-zA-Z-\.\\/:;0-9\u3000]{5,}", "", text)
    # 替换符号为空格
    text = re.sub("[\t|\n|\r|\u3000]+", " ", text)
    # 去除连续空格
    text = re.sub("[ ]{2,}", " ", text)
    if len(text) < 2:
        return ""
    return text

def sample_generate(raw_path,save_path,already_json_path=None):
    """
    只针对分类任务，
    从标注平台导出的csv或者jsonl的标注文件路径  和 标签的json文件 作为为raw_path
    :param raw_path:
    :param save_path:
    :return: 清洗完的数据保存到save_path
    """
    print("sample_generate function")
    if not already_json_path:
        assert isinstance(raw_path,list) and len(raw_path)==2,print("缺少标注文件或者标签文件")
        text_label_path,label_path = raw_path
        text_label_res = readFileFromClassificationTask(text_label_path)
        labels = readJson(text_label_path,lines=False)
        label_id2name_dic = {v['id']:v['text'] for v in labels}
        json_data = []
        for k,v in text_label_res.items():
            json_data.append({
                "text":k,
                "label":[label_id2name_dic[x] for x in v]
            })
    else:
        print(already_json_path)
        json_data = readJson(already_json_path,lines=True)
    print("raw_data num is %d"%len(json_data))
    washed_data = []
    for jd in json_data:
        if ('瘤' not in jd['text']) and ('癌' not in jd['text']):
            continue
        washed_data.append({
            "text":wash_data(jd["text"]),
            "label":list(set(jd["label"]))
        })
    print(len(washed_data))
    saveJson(washed_data,save_path,lines=True)
def split_type_degree(merge_save_path,type_save_path,degree_save_path):
    merge_data = readJson(merge_save_path,lines=True)
    type_json,degree_json = [],[]
    for sd in merge_data:
        if len(sd['text']) < 2:
            continue
        tmp = set(sd['label'])
        degree_v = set([x for x in tmp if '分化' in x])
        type_v = tmp - degree_v

        degree_l = list(degree_v) if len(degree_v) > 0  else ['其他']
        type_l   = list(type_v) if len(type_v) > 0 else ['其他']
        type_json.append({
            'text':sd['text'],
            'label':type_l
        })
        degree_json.append({
            'text':sd['text'],
            'label':degree_l
        })
    saveJson(type_json,type_save_path,lines=True)
    saveJson(degree_json,degree_save_path,lines=True)






def readFileFromClassificationTask(text_label_path):
    """
    :param text_label_path:标注好的文件
    :return: dict, text --> labels(set)
    """
    assert text_label_path.endswith("csv") or text_label_path.endswith("jsonl"),print("错误文件格式")
    res = collections.defaultdict(set)
    if text_label_path.endswith("csv"):
        data = pd.read_csv(text_label_path)
        text,label_idx = list(data['text']),list(data['label'])
        for (a,b) in zip(text,label_idx):
            res[a].add(b)

    elif text_label_path:
        data = readJsonL(text_label_path)
        for d in data:
            for s in d['annotations']:
                res[d['text']].add(s)
    return res


if __name__ == '__main__':
    config = ConfigParser()
    # config_path = "../data/dev/samples_generate_v1.conf"
    # config.read(config_path,encoding='utf-8')
    config.read(sys.argv[1],encoding='utf-8')
    print(sys.argv[1])
    print(config.sections())
    raw_path = config['samples_generate']["raw_data_path"]
    sample_save_path = config["samples_generate"]["sample_save_path"]
    already_json_path = config['samples_generate']['already_json_path']
    type_save_path = config['samples_generate']['type_save_path']
    degree_save_path = config['samples_generate']['degree_save_path']
    log_path = config["samples_generate"]["log"]

    checkFileOMake(log_path)

    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(log_path,'log'),
                        filemode='a')
    logging.info("Pathology samples generating !")
    localtime = time.asctime(time.localtime(time.time()))
    logging.info("### start time : %s"%(localtime))
    time_stamp = int(time.time())
    logging.info("time stamp: %d"%(time_stamp))


    try:
        print("保存数据路径: %s"%(sample_save_path))
        logging.info("原始数据路径: %s"%(raw_path))
        sample_generate(raw_path,sample_save_path,already_json_path=already_json_path)
        split_type_degree(sample_save_path,type_save_path,degree_save_path)
    except:
        logging.info("出现异常")





