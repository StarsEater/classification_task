import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tools import *
from utils.mysql_util import *
from utils.samples_generate import wash_data
class BertDataset(Dataset):
    """
    Bert 模型的数据集
    一条原始数据格式
    text（str）：“按顺序实现”
    label(list):["其他"]

    数据格式规定
    json
    [{"text":"飞啊试驾评测基础精品剧场","label":["其他"]}]

    """
    def __init__(self,tokenizer=None,label2num=None,train_path=None,val_path=None,test_path=None,exector=None,sql_words= ["id", "Exam_Result"]):
        self.tokenizer = tokenizer
        self.label2num = label2num
        self.process_data = []
        if train_path:
            self.path = train_path
            raw_data = readJson(self.path, lines=True)
            for rd in raw_data:
                t, l = wash_data(rd["text"]), rd["label"]
                if len(t) < 1:
                    continue
                self.process_data.append(self.convert_data2bertFormat(t, l))
        elif val_path:
            self.path = val_path
            raw_data = readJson(self.path, lines=True)
            for rd in raw_data:
                t, l = rd["text"], rd["label"]
                if len(t) < 1:
                    continue
                self.process_data.append(self.convert_data2bertFormat(t, l))
        else:
            self.path = test_path
            raw_mysql_res = exector.get_sql_result_by_words(sql_words)
            for (id, er) in raw_mysql_res:
                er = wash_data(er.strip())
                self.process_data.append(self.convert_data2bertFormat(er, ["其他"]))
        print("total data num is {}".format(len(self.process_data)))
    def __getitem__(self, item):
        return self.process_data[item]
    def __len__(self):
        return len(self.process_data)

    def convert_data2bertFormat(self,text,label,one_label=True):
        text = ['[CLS]']+self.tokenizer.tokenize(text)[:500]+['[SEP]']
        text_index = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(text))
        mask_index = torch.ByteTensor([1] * len(text_index))
        label = [self.label2num[ll] for ll in label]
        if one_label:
            label_index = torch.LongTensor(list(label)[0:1])
        else:
            label_index = torch.LongTensor([1 if i in label else 0 for i in range(len(self.label2num))])
        return text_index,mask_index,label_index
def BertCollate(batch):
    text_mini_batch,mask_mini_batch,label_mini_batch = zip(*batch)
    text_mini_batch = pad_sequence(text_mini_batch,batch_first=True,padding_value=0)
    mask_mini_batch = pad_sequence(mask_mini_batch,batch_first=True,padding_value=0)
    label_mini_batch = pad_sequence(label_mini_batch,batch_first=True,padding_value=0)
    return text_mini_batch,mask_mini_batch,label_mini_batch

if __name__ == '__main__':
    Example_dir = "../dataFormatExamples/"
    example_json_nolines_path = os.path.join(Example_dir,"json_example_nolines.json")
    example_json_lines_path = os.path.join(Example_dir, "json_example_lines.json")
    pd_path = os.path.join(Example_dir,"pd_example.csv")
    example_json = [{"text":"2021年1月21日0-24时，石家庄市新增新冠肺炎确诊病例15例。","label":["其他;肺癌"]},\
                    {"text":"其中，最小的年龄只有1岁。目前，对所有追踪到的密接者已全部采取集中隔离医学观察措施","label":["其他","细胞癌"]}]
    example_csv = {"text":["2021年1月21日0-24时，石家庄市新增新冠肺炎确诊病例15例。","其中，最小的年龄只有1岁。目前，对所有追踪到的密接者已全部采取集中隔离医学观察措施"],
                   "label":[["其他","肺癌"],["其他","细胞癌"]]}
    saveJson(example_json,example_json_lines_path,lines=True)
    saveJson(example_json,example_json_nolines_path,lines=False)
    dict_list2csv(example_csv,pd_path)

