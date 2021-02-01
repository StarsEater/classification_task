import collections
import logging
import os
import random
import sys
import time
import torch.nn as nn
import torchnet.meter as meter
from configparser import ConfigParser

import torch
import numpy as np
import tqdm
from prettyprinter import cpprint
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.transformer_series import BertDataset, BertCollate
from model.bert_linear import BertLinear
from tools import *
from utils._utils import  one_label_f1_score, plot_loss, plot_acc_score, plot_f1_score
from transformers import BertTokenizer,AdamW, get_linear_schedule_with_warmup

import warnings

from utils.dataset_merge_and_shuffle import dataset_merge_shuffle
from utils.mysql_util import  MysqlExec

warnings.filterwarnings("ignore")
task_name = os.path.abspath(os.path.dirname(os.path.dirname(__file__))).split("/")[-1]
def seed_torch(seed=2200):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()

def PreProcess(train_raw_path,labels_map,config_path = "./data/dev/train_pipeline.conf",project_name=None):
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')
    dataset_save_path = config['dataset_split']['dataset_save_path'].replace("Pathology",project_name)
    checkFileOMake(dataset_save_path)
    split_ratio = [float(x) for x in config['dataset_split']['split_ratio'].split(",")]
    raw_train_data = readJson(train_raw_path,lines=True)


    # 删去其他
    for k,v in labels_map.items():
        if "其他" in v:
            v.remove("其他")
        v = set(v)
        dataset_dir = dataset_save_path.replace("clss",k).replace("Pathology",project_name)
        checkFileOMake(dataset_dir)
        tmp_path = os.path.join(dataset_dir,"raw.json")
        sub_data = []
        for t_l in raw_train_data:
            t,l = t_l["text"],t_l["label"]
            l = list(filter(lambda x:x in v,l))
            if len(l)==0:
                l = ["其他"]
            sub_data.append({
                "text":t,
                "label":l
            })
        saveJson(sub_data,tmp_path,lines=True)
        dataset_merge_shuffle(tmp_path,dataset_dir,split_ratio)

def Train(labels,clss, config_path = "./data/dev/train_pipeline.conf",project_name=None):
    """
    :param labels:  标签列表
    :param clss:  类别
    :param config_path: 配置文件名
    :return:
    """
    config = ConfigParser()
    config.read(config_path,encoding='utf-8')
    dataset_save_path = config['dataset_split']['dataset_save_path'].replace("clss",clss).replace("Pathology",project_name)
    trainset_path    = os.path.join(dataset_save_path,config['model_train']['train_path'])
    valset_path      = os.path.join(dataset_save_path,config['model_train']['dev_path'])
    resume           = int(config['model_train']['resume'])
    checkpoint_path  = config['model_train']['checkpoint'].replace("clss",clss).replace("Pathology",project_name)
    history_path     = config['model_train']['history'].replace("clss",clss).replace("Pathology",project_name)
    log_path         = config['model_train']['log']
    model_name       = config['model_train']['model_save_name'].replace("clss",clss).replace("Pathology",project_name)
    model_resume_name= config['model_train']['model_resume_name'].replace("clss",clss).replace("Pathology",project_name)
    model_choice     = config['model_train']['model_choice']
    batch_size       = int(config['model_train']['batch_size'])
    end_epoch        = int(config['model_train']['end_epoch'])
    lr               = float(config['model_train']['lr'])
    embed_type       = config['model_train']['embed_type']
    warmup           = True if config['model_train']['warmup']=='True' else False
    num_warmup_steps = int(config['model_train']['num_warmup_steps'])
    num_total_steps  = int(config['model_train']['num_total_steps'])
    bert_pretrain    = True if config['model_train']['bert_pretrain']=='True' else False
    device_id        = config['model_train']['train_device_id']
    device_id        = [int(item) for item in device_id.split(",")]

    # 构造标签映射词典
    labels2numdic = {v:i for i,v in enumerate(labels)}
    assert "其他" in labels2numdic
    labels_size = len(labels2numdic)
    print("total label num is %d"%(labels_size))

    if not embed_type:
        print("Warning! No embed type assigned!")
    checkFileOMake(log_path)
    checkFileOMake(checkpoint_path)
    checkFileOMake(history_path)

    log_save_name = 'log_' + model_name + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(log_path,log_save_name),
                        filemode='a')
    checkpoint_name = os.path.join(checkpoint_path,model_name+'_best_ckpt.pth')
    model_ckpt_name = os.path.join(checkpoint_path,model_name+'_best.pkl')
    print("模型保存路径为%s"%(model_ckpt_name))
    print("模型训练评价指标保存路径为%s"%(checkpoint_name))
    checkFileOMake(model_resume_name)

    localtime = time.asctime(time.localtime(time.time()))
    logging.info('#### start time : %s'%(localtime))
    time_stamp = int(time.time())
    logging.info('time stamp: %d'%(time_stamp))
    logging.info('###### Model: %s'%(model_name))
    logging.info('trainset path: %s'%(trainset_path))
    logging.info('valset path: %s'%(valset_path))
    print('trainset path: %s'%(trainset_path))
    print('valset path: %s'%(valset_path))

    #######################################
    if embed_type=='pretrained':
        pass
    elif embed_type=='bert':
        print('Loading bert tokenizer!')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        print("loading <bert> dataset    ... ...")
        train_dataset = BertDataset(tokenizer=tokenizer,
                                    label2num=labels2numdic,
                                    train_path=trainset_path)
        val_dataset   = BertDataset(tokenizer=tokenizer,
                                    label2num=labels2numdic,
                                    val_path=valset_path)
        train_dataloader = DataLoader(dataset = train_dataset,
                                      batch_size = batch_size,
                                      shuffle=True,
                                      num_workers=1,
                                      collate_fn=BertCollate)
        val_dataloader =   DataLoader(dataset=val_dataset,
                                      batch_size=batch_size*2,
                                      shuffle=False,
                                      num_workers=1,
                                      collate_fn=BertCollate)

    #####################################
    print('batch_size:%d'%(batch_size))
    print('learning rate:%f'%(lr))
    print('end epoch:%d'%(end_epoch))
    logging.info('batch_size:%d'%(batch_size))
    logging.info('learning rate:%f'%(lr))
    logging.info('end epoch:%d'%(end_epoch))
    device= torch.device('cuda:'+str(device_id[0]) if torch.cuda.is_available() else 'cpu')

    print("use",device)

    #####################################
    if model_choice=='bert':
        model = BertLinear(
                 labelset_size=labels_size,
                 load_pretrain=bert_pretrain,
                 pretrain_ckpt="",
                 bert_path='bert-base-chinese',
                 loss_weight = None)
    elif model_choice=='':
        pass

    if len(device_id) > 1:
        print("let's use ",len(device_id),'GPU!')
        model = nn.DataParallel(model,device_id=device_id)

    model.to(device)

   ######################################
    if resume !=0:
        logging.info("Resuming from checkpoint ...")
        model.load_state_dict(torch.load(model_resume_name))
        checkpoint = torch.load(checkpoint_name)
        best_f1 = checkpoint['f1']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
    else:
        best_f1 = 0.0
        start_epoch = -1
        history = {
            'train_loss':[],'val_loss':[],
            'train_acc':[],'val_acc':[],
            'train_f1':[],'val_f1':[],
            'train_pre':[],'val_pre':[],
            'train_rec':[],'val_rec':[]
        }

    ###################################
    if not warmup:
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = StepLR(optim, step_size=5, gamma=0.8)
    else:
        optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_total_steps)  # PyTorch scheduler
        logging.info("warm up steps %d " % (num_warmup_steps))
        logging.info("total steps %d " % (num_total_steps))

    print("Start training")
    writer = SummaryWriter(os.path.join('./summary_runs/',model_name))
    global_step = 0
    loss_tr = meter.AverageValueMeter()
    acc_tr  = meter.AverageValueMeter()
    f1_tr   = meter.AverageValueMeter()
    pre_tr  = meter.AverageValueMeter()
    rec_tr  = meter.AverageValueMeter()

    loss_va = meter.AverageValueMeter()
    acc_va  = meter.AverageValueMeter()
    f1_va   = meter.AverageValueMeter()
    pre_va  = meter.AverageValueMeter()
    rec_va  = meter.AverageValueMeter()

    for epoch in range(start_epoch+1,end_epoch):
        print("--------------epoch:%d------------"%(epoch))
        logging.info("-----------epoch:%d-----------"%(epoch))
        model.train()
        loss_tr.reset()
        acc_tr.reset()
        f1_tr.reset()
        pre_tr.reset()
        rec_tr.reset()

        loss_va.reset()
        acc_va.reset()
        f1_va.reset()
        pre_va.reset()
        rec_va.reset()

        print("start training !!! !!!")
        for batch_id,batch in enumerate(train_dataloader):
            in_texts, masks, out_labels = map(lambda x:x.to(device),batch)
            model.zero_grad()
            loss,pre_labels = model(in_texts,masks,out_labels)
            loss.backward()

            loss_item = loss.item()
            writer.add_scalar('train_loss',loss_item,global_step)
            global_step += 1

            optim.step()

            if warmup:
                scheduler.step()
            loss_tr.add(loss_item)
            if batch_id % 20==0:
                print('-----------------batch:%d------------'%(batch_id))
                print('-----------------loss:%f--------------'%(loss_item))
        mean_loss = loss_tr.value()[0]
        print('trainset loss:%f' % (mean_loss))
        logging.info('trainset loss:%f' % (mean_loss))
        history['train_loss'].append(mean_loss)

        ###################  val ########################
        model.eval()
        with torch.no_grad():
            print("start validating !!! !!!")
            true_label_lst, pre_label_lst = [],[]
            for batch_id, batch in enumerate(val_dataloader):
                in_texts, masks, tmp_out_labels = map(lambda x: x.to(device), batch)
                model.zero_grad()
                loss, tmp_pre_labels = model(in_texts, masks, tmp_out_labels)
                loss_item = loss.item()
                loss_va.add(loss_item)
                true_label_lst += tmp_out_labels.detach().tolist()
                pre_label_lst  += tmp_pre_labels.detach().tolist()


            acc, pre, rec, f1 = one_label_f1_score(true_label_lst, pre_label_lst,target_name=list(labels2numdic.keys()))
            acc_va.add(acc)
            pre_va.add(pre)
            rec_va.add(rec)
            f1_va.add(f1)

            mean_loss = loss_va.value()[0]
            mean_acc = acc_va.value()[0]
            mean_pre = pre_va.value()[0]
            mean_rec = rec_va.value()[0]
            mean_f1 = f1_va.value()[0]

            print('valset loss:%f' % (mean_loss))
            print('valset f1:%f' % (mean_f1))
            print('valset acc:%f' % (mean_acc))
            print('valset pre:%f' % (mean_pre))
            print('valset rec:%f' % (mean_rec))

            logging.info('valset loss:%f' % (mean_loss))
            logging.info('valset f1:%f' % (mean_f1))
            logging.info('valset acc:%f' % (mean_acc))
            logging.info('valset pre:%f' % (mean_pre))
            logging.info('valset rec:%f' % (mean_rec))

            history['val_loss'].append(mean_loss)
            history['val_acc'].append(mean_acc)
            history['val_f1'].append(mean_f1)
            history['val_pre'].append(mean_pre)
            history['val_rec'].append(mean_rec)

            if mean_f1 >best_f1:
                logging.info('Checkpoint Saving ...')
                print("best f1 score so far ! Checkpoint Saving ...")
                print("save state path is %s model path is %s"%(checkpoint_name,model_ckpt_name))
                state = {
                    'epoch':epoch,
                    'f1':mean_f1,
                    'history':history
                }
                torch.save(state,checkpoint_name)
                best_f1 = mean_f1
                torch.save(model.state_dict(),model_ckpt_name)
            plot_loss(history,history_path,model_name,time_stamp)
            plot_acc_score(history,history_path,model_name,time_stamp)
            plot_f1_score(history,history_path,model_name,time_stamp)
        if not warmup:
            scheduler.step()
            logging.info('current lr:%f'%(scheduler.get_lr()[0]))

@torch.no_grad()
def Test(host,user,password,database,port,charset,tablename,sql_words,id,value,
         labels,
         clss,
         config_path = "./data/dev/train_pipeline.conf",
         project_name = None):
    print("start Test !!! !!! ")
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')


    train_device_id = config['model_train']['train_device_id']
    train_device_id = [int(item) for item in train_device_id.split(",")]
    test_device_id = config['model_test']['test_device_id']
    test_device_id = [int(item) for item in test_device_id.split(",")]
    device = torch.device('cuda:' + str(test_device_id[0]) if torch.cuda.is_available() else 'cpu')

    train_cuda = 'cuda:' + str(train_device_id[0])
    test_cuda = 'cuda:' + str(test_device_id[0])

    model_name = config['model_train']['model_save_name'].replace("clss", clss).replace("Pathology",project_name)
    model_resume_name = config['model_train']['model_resume_name'].replace("clss", clss).replace("Pathology",project_name)
    model_ckpt_name = os.path.join(model_resume_name, model_name + '_best.pkl')
    model_choice = config['model_train']['model_choice']

    pre_out_dir = config['model_test']['pred_out_path'].replace("Pathology",project_name)
    checkFileOMake(pre_out_dir)


    mysqlexec = MysqlExec(host=host,user=user,password=password,database=database,port=port,charset=charset,table_name=tablename)

    # 构造类型标签映射词典
    labels2numdic = {v: i for i, v in enumerate(labels)}
    assert "其他" in labels2numdic

    # 加载类型模型
    if model_choice == 'bert':
        model = BertLinear(
            labelset_size=len(labels),
            bert_path='bert-base-chinese',
            loss_weight=None)
    elif model_choice == '':
        pass
    model.to(device)
    logging.info("Type Model Resuming from checkpoint ...")
    model.load_state_dict(torch.load(model_ckpt_name, map_location={train_cuda: test_cuda}))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = BertDataset(tokenizer=tokenizer, label2num=collections.defaultdict(lambda:0),exector=mysqlexec,sql_words=sql_words)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=1,collate_fn=BertCollate)
    label_lst = []
    for batch in tqdm.tqdm(dataloader):
        in_texts, masks, out_labels = map(lambda x: x.to(device), batch)
        _, tmp_type_labels =  model(in_texts, masks, out_labels)
        _, tmp_degree_labels = model(in_texts, masks, out_labels)
        label_lst += tmp_type_labels.detach().tolist()

    np.save(os.path.join(pre_out_dir,str(clss)+".npy"), np.array(label_lst))
    label_lst = [labels[x[0]] for x in label_lst]
    saveJson(label_lst,os.path.join(pre_out_dir,str(clss)+"tr.json"),lines=False)

def PostProcess(host,user,password,database,port,charset,tablename,sql_words,id,value,
                labels_map={},
                sql_text="",
                ids="",
                config_path = "./data/dev/train_pipeline_v1.conf",
                project_name = None
                ):
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')
    pred_out_dir = config['model_test']['pred_out_path'].replace("Pathology",project_name)


    mysqlexec = MysqlExec(host=host,user=user,password=password,database=database,port=port,charset=charset,table_name=tablename)


    type_labels = labels_map["type1"]
    degree_labels = labels_map["degree1"]

    type_label_lst = np.load(os.path.join(pred_out_dir,"type1.npy"))
    degree_label_lst = np.load(os.path.join(pred_out_dir,"degree1.npy"))

    good_cancer = set(["乳头状瘤", "囊腺瘤", "炎性假瘤", "畸胎瘤", "胸腺瘤", "腺瘤", "错构瘤"])
    bad_cancer = set(["小细胞癌", "神经内分泌肿瘤", "类癌", "肉瘤样癌", "腺癌", "腺鳞癌", "鳞癌"])
    merge_result = []
    for tmp_type, tmp_degree, st, idx in tqdm.tqdm(zip(type_label_lst, degree_label_lst, sql_text, ids)):
        tmp_type, tmp_degree = type_labels[tmp_type[0]], degree_labels[tmp_degree[0]]

        if "瘤" not in st and "癌" not in st or tmp_type == '其他' \
                or tmp_type in good_cancer and '瘤' not in st \
                or tmp_type in bad_cancer and ('癌' not in st and '神经' not in st):
            res = '其他'
        elif tmp_type in good_cancer:
            res = tmp_type
        else:
            res = tmp_type + ";" + tmp_degree
        merge_result.append(res)
        break
        # mysqlexec.update_sql_by_words(name1=value, value1=res, name2=id, value2=idx)
    saveJson(merge_result,"./merge_result.json",lines=True)
def PipeLine(config_path = "./pipeline.conf"):
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')
    project_name = config['user_config']['project_name']
    assert project_name==task_name,print("project must be same as task")
    labels_set = [x.split(":") for x in config["user_config"]['labels_set'].split()]
    train_raw_path=config["user_config"]["train_raw_path"]
    train_config_path=config["user_config"]["train_config_path"]
    host = config["user_config"]['host']
    user = config["user_config"]['user']
    password = config["user_config"]['password']
    database = config["user_config"]['database']
    port = int(config["user_config"]['port'])
    charset = config["user_config"]['charset']
    tablename = config["user_config"]['tablename']
    sql_words = config["user_config"]['sql_words'].split(",")
    id_name = config["user_config"]['id']
    value_name = config["user_config"]['value']
    labels_map = {}
    for ls in labels_set:
        labels_map[ls[0]]=ls[1].split(",")
    cpprint(labels_map)
    print("******* PreProcess (including split labels and dataset) *********")
    from copy import deepcopy as cp
    PreProcess(train_raw_path, cp(labels_map), config_path=train_config_path,project_name=project_name)
    for k,v in labels_map.items():
        print("*******************start train {}*****************************".format(k))
        Train(v,k,train_config_path,project_name)
        Test(host, user, password, database, port, charset, tablename, sql_words, id_name, value_name,v,k,config_path=train_config_path,project_name=project_name)
    mysqlexec = MysqlExec(host, user, password, database, port, charset, tablename)
    ids,sql_text = list(zip(*mysqlexec.get_sql_result_by_words(sql_words)))
    print('******** PostProcess (add some rules)  **********************')
    PostProcess(host, user, password, database, port, charset, tablename, sql_words, id_name, value_name,
                labels_map=labels_map,
                sql_text=sql_text,
                ids=ids,
                config_path=train_config_path,
                project_name=project_name
                )


#
if __name__ == '__main__':
    PipeLine()

# if __name__ == '__main__':
#     print(task_name)

