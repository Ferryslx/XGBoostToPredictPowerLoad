import os
from time import strftime

import pandas as pd
import matplotlib.pyplot as plt
import datetime

from matplotlib.ticker import MultipleLocator

from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error,mean_absolute_percentage_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

#配置电力负荷预测类
class PowerLoadPredict(object):
    def __init__(self,filename):
        logfile_name = 'predict' + datetime.datetime.now().strftime('%Y%m%d')
        self.logger =Logger('../',logfile_name).get_logger()
        self.data_source=data_preprocessing(filename)
        #把历史数据转为字典，key：时间，value：负荷，目的是避免频繁的操作dataframe，提高效率
        self.time_load_dict=self.data_source.set_index('time').power_load.to_dict()

#预测数据解析特征，保持与模型训练时的特征一致
def pred_feature_extract(time_dict,time,logger):
    logger.info(f'-----正在解析时间{time}对应的特征-----')
    feature_names=[['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                    'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                    'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                    'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                    'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                    'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                    '前1小时', '前2小时', '前3小时', 'yesterday_load']]
    #解析时间特征
    #截取time字段的小时信息
    pre_hour=time[11:13]    #例如'2026-06-20 19:00:00'->'19'
    time_list=[]
    for i in range(24):
        if i==int(pre_hour):
            time_list.append(1)
        else:
            time_list.append(0)

    #截取time字段的月份信息
    pre_month=time[5:7]
    for i in range(1,13):
        if i==int(pre_month):
            time_list.append(1)
        else:
            time_list.append(0)

    #截取time前3个小时的负荷信息(窗口特征)
    last_1h_ago=(pd.to_datetime(time)-datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load=time_dict.get(last_1h_ago,500) #获取前1个小时对应的负荷，如果没有就用500填充

    last_2h_ago = (pd.to_datetime(time) - datetime.timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = time_dict.get(last_2h_ago, 500)  # 获取前2个小时对应的负荷，如果没有就用500填充

    last_3h_ago = (pd.to_datetime(time) - datetime.timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = time_dict.get(last_3h_ago, 500)  # 获取前3个小时对应的负荷，如果没有就用500填充

    #获取昨天同一时刻的负荷
    yesterday_time=(pd.to_datetime(time)-datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    yesterday_load=time_dict.get(yesterday_time,500)

    feature_data=time_list+[last_1h_load,last_2h_load,last_3h_load,yesterday_load]

    #转成df对象，返回
    feature_df=pd.DataFrame([feature_data],columns=feature_names)
    return feature_df

#将实际与预测结果可视化
def visual_pred(data):
    plt.figure(figsize=(30,20))
    plt.title('电力负荷预测结果可视化')
    plt.xlabel('time')
    plt.ylabel('load')
    plt.plot(data['time'],data['real_load'],'mo-',label='real')
    plt.plot(data['time'],data['pred_load'],'co-',label='predict')
    plt.legend()
    #设置x轴刻度间隔，以及x轴标签值旋转角度
    plt.gca().xaxis.set_major_locator(MultipleLocator(base=50))
    plt.xticks(rotation=45, ha='right')
    plt.savefig('../diagrams/预测结果图')
    plt.show()


if __name__ == '__main__':
    power_load=PowerLoadPredict('../data/test.csv')
    logger=power_load.logger
    logger.info('开始创建模型预测对象')
    data=power_load.data_source
    time_load_dict=power_load.time_load_dict
    #加载模型对象
    estimator=joblib.load('../model/model.pkl')

    #确定要预测的时间段（2015/8/1 00:00:00及以后的时间）
    pre_times=data['time'][data['time']>='2015-08-01 00:00:00']

    #为了模拟实际场景的预测，把要预测的时间以及以后的负荷都覆盖掉，因此新建一个数据字典，只保存预测时间以前的数据字典
    #pre_time是要预测的时间，time_load_dict_masked是要预测的时间之前的所有时间（掩盖预测时间之后的数据）
    evaluate_list=[]
    for pre_time in pre_times:
        logger.info(f'正在预测{pre_time}的负荷...')
        time_load_dict_masked = {k: v for k, v in time_load_dict.items() if k < pre_time}

        #预测
        proceed_data=pred_feature_extract(time_load_dict_masked,pre_time,logger)

        y_pred=estimator.predict(proceed_data).item()   #predict的结果都是ndarry类型的，即[[1234]]，所以要调用item函数取出其中的值,但是item只能用于shape为1的量

        real_load = time_load_dict.get(pre_time,500)

        evaluate_list.append({
            'time':pre_time,
            'pred_load':y_pred,
            'real_load':real_load
        })
    evaluate_df=pd.DataFrame(evaluate_list)
    visual_pred(evaluate_df)
    logger.info(f'模型预测结果的均方误差为:{mean_squared_error(evaluate_df['pred_load'],evaluate_df['real_load'])}')
    logger.info(f'模型预测结果的均方根误差为:{root_mean_squared_error(evaluate_df['pred_load'],evaluate_df['real_load'])}')
    logger.info(f'模型预测结果的平均绝对误差为:{mean_absolute_error(evaluate_df['pred_load'],evaluate_df['real_load'])}')