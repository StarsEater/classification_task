import os

import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score,f1_score
def readVectors(path,topn):
    """
     读取 topn个词向量
    :param path:
    :param n:
    :return:
    """
    lines_num,dim = 0,0
    vectors,iw,wi = {},{},{}
    with open(path,encoding='utf-8') as f:
        tokens = f.readlines()
        dim =  int(tokens[0].rstrip().split()[1])
        for i in range(1,len(tokens)):
            token = tokens[i].rstrip().split(' ')
            if len(token[0])==1:
                lines_num += 1
                vectors[token[0]] = np.asarray([float(x) for x in token[1:]])
                iw.append(token[0])
            if topn !=0 and lines_num >= topn:
                break
    unk = np.random.rand(dim)
    unk = (unk - 0.5)/100
    vectors['unk'] = unk

    wi = {w: i for i, w in enumerate(iw)}
    print("word vector lens(including unk) is %d"% (len(vectors)))
    return vectors,iw,wi,dim

# 多标签的精确匹配
def multi_label_f1_score(true_label_lst,pre_label_lst):
    """

    :param true_label_lst:[[1,0,1],[0,0,1],[1,0,1],[0,0,1]]
    :param pre_label_lst: [[],[],[],[]]
    :return: 精确匹配的 f1,precision,recall,accurate
    """
    assert len(true_label_lst)==len(pre_label_lst),\
        print("长度不一致",len(true_label_lst),len(pre_label_lst))
    assert  len(true_label_lst[0])==len(pre_label_lst[0]),\
        print("长度不一致",len(true_label_lst),len(pre_label_lst))
    pass

# 单标签多分类
def one_label_f1_score(true_label_lst,pre_label_lst,target_name,way="micro"):
    assert way in ["macro","micro"]
    assert len(true_label_lst) == len(pre_label_lst), \
        print("长度不一致", len(true_label_lst), len(pre_label_lst))
    true_label_lst = reduce(lambda x,y:x+y,true_label_lst)
    pre_label_lst = reduce(lambda x,y:x+y,pre_label_lst)
    print(classification_report(true_label_lst,pre_label_lst,labels=range(len(target_name)),target_names=target_name))
    acc = accuracy_score(true_label_lst,pre_label_lst)
    pre,rec,f1 = precision_score(true_label_lst,pre_label_lst,average=way),\
                 recall_score(true_label_lst,pre_label_lst,average=way),\
                 f1_score(true_label_lst,pre_label_lst,average=way)
    return acc,pre,rec,f1

def plot_loss(history, save_root, model_name,time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_loss'], 'r', history['val_loss'], 'b')
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss during training")
    if save_mode:
        plt.savefig(os.path.join(save_root,'loss_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()

def plot_acc_score(history, save_root,model_name, time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_acc'], 'r', history['val_acc'], 'b')
    plt.legend(["train_acc_score", "val_acc_score"])
    plt.xlabel("epoch")
    plt.ylabel("acc_score")
    plt.title("acc_score during training")
    if save_mode:
        plt.savefig(os.path.join(save_root , 'acc_score_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()

def plot_f1_score(history, save_root, model_name,time_stamp=0,save_mode=True):
    plt.figure()
    plt.plot(history['train_f1'], 'r', history['val_f1'], 'b')
    plt.legend(["train_f1_score", "val_f1_score"])
    plt.xlabel("epoch")
    plt.ylabel("f1_score")
    plt.title("f1_score during training")
    if save_mode:
        plt.savefig(os.path.join(save_root , 'f1_score_'+model_name+'_'+str(time_stamp)+'.jpg'))
    plt.close()





