"""
随机从数据库抽取100条进行评估，前提是文本中含有瘤和癌的
"""
from utils.mysql_util import *
from prettyprinter import cpprint
import random


def sample_and_evaluation(num=100):
    all_res = get_sql_result("SELECT id,Exam_Result,stand_sign FROM Pathology")
    all_res_filter = list(filter(lambda x:"瘤" in x[1] or "癌" in x[1],all_res))
    rx = random.sample(all_res_filter,num)
    cpprint(rx)

if __name__ == '__main__':
    sample_and_evaluation()
