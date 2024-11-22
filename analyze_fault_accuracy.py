
from stgcn_utils import *


import torch


import pandas as pd

def save_time():
    cg = torch.load('./checkpoints/STGCN_model.pth')

    test_input = cg['test_X']
    test_target = np.array(cg['test_label'])
    y_test_pred = cg['test_pred']
    test_original_timestamp = cg['test_time']

    print(sum(test_target==0))
    print(sum(test_target==1))

    time_list = []
    for target, pred, time in zip(test_target, y_test_pred, test_original_timestamp):
        if (target == pred) and (target == 1):
            time_list.append(time)


    fault_list = pd.read_csv('./ground_truth_total.csv',encoding='gbk')
    fault_type=[]
    for i in time_list:
        timestamp = int(i[:10])
        platform = i[10:]
        temp=fault_list[fault_list['cluster']==platform]
        for index,row in temp.iterrows():
            if timestamp>=int(row['timestamp']) and timestamp<=int(row['timestamp'])+300:
                fault_type.append(row['failure_type'])

    return fault_type



if __name__ == '__main__':
    save_time()
