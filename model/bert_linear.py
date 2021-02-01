import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
class BertLinear(nn.Module):
    def __init__(self,
                 labelset_size,
                 load_pretrain=False,
                 pretrain_ckpt='',
                 bert_path='bert-base-chinese',
                 loss_weight = None):
        super(BertLinear, self).__init__()
        self.labelset_size = labelset_size
        model = BertForSequenceClassification.from_pretrained(bert_path,num_labels=labelset_size)
        # if load_pretrain:
        #     print('load pretrain weight: %s'%(pretrain_ckpt))
        #     pretrain_dic = torch.load(pretrain_ckpt)
        #     model_dic = model.state_dict()
        #     new_dic = {}
        #     for key in model_dic.keys():
        #         if key.split(".")[0] == 'bert':
        #             new_dic[key] = pretrain_dic[key]
        #         else:
        #             new_dic[key] = model_dic[key]
        #     model.load_state_dict(new_dic)
        self.bert = model
        # self.criterion = nn.CrossEntropyLoss(weight=loss_weight)
    def forward(self, input_ids,mask,label=None):
        loss,logits= self.bert(input_ids=input_ids,attention_mask = mask,labels=label)[:2]
        pre_label = torch.argmax(logits.detach(),dim=-1,keepdim=True)
        return loss,pre_label