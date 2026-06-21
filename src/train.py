import os
from time import strftime

import pandas as pd
import matplotlib.pyplot as plt
import datetime

from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error,mean_absolute_percentage_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

#定义电力负荷模型类，配置日志，获取数据源
class PowerLoadModel(object):
    #初始化属性信息
    def __init__(self,filename):
        log_file_name='train'+datetime.datetime.now().strftime('%Y%m%d')
        self.logfile=Logger('../',log_file_name).get_logger()
        self.data_source=data_preprocessing(filename)

#查看数据的整体分布情况
def analysis_data(data):
    """
    :param data:数据源
    :return:
    """
    #防止修改源数据，做一次拷贝
    ana_data=data.copy()
    #查看数据整体情况
    ana_data.info()
    #1.负荷整体的分布情况(直方图)
    fig=plt.figure(figsize=(20,40))
    #添加子图
    ax1=fig.add_subplot(411)#411表示分成4行1列，占第一个子图
    ax1.hist(ana_data['power_load'],bins=100)   #bins表示100个区间
    ax1.set_title('负荷整体情况')
    ax1.set_xlabel('负荷')

    #2.各个小时的平均负荷趋势，查看负荷在一天中的变化情况
    ana_data['hour']=ana_data['time'].str[11:13]        #新增一列，充当小时列
    hour_load_mean=ana_data.groupby('hour',as_index=False)['power_load'].mean()     #as_index表示是否把分组字段作为索引列
    #画出折线图
    ax2=fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'],hour_load_mean['power_load'])
    ax2.set_title('各个小时的平均负荷趋势')
    ax2.set_xlabel('hour')

    #3.各个月份的平均负荷趋势，查看负荷在一年中的变化情况
    ana_data['month'] = ana_data['time'].str[5:7]  # 新增一列，充当月份列
    month_load_mean = ana_data.groupby('month', as_index=False)['power_load'].mean()
    # 画出折线图
    ax3 = fig.add_subplot(413)
    ax3.plot(month_load_mean['month'], month_load_mean['power_load'])
    ax3.set_title('各个月份的平均负荷趋势')
    ax3.set_xlabel('month')

    #4.工作日与周末的平均负荷情况，查看工作日负荷与周末是否有区别
    ana_data['weekday']=ana_data['time'].apply(lambda x:pd.to_datetime(x).weekday())    #新增一列，记录星期几（0~6,实际就是周一~周日）
    ana_data['is_holiday']=ana_data['weekday'].apply(lambda x:1 if x in [5,6] else 0)   #0:工作日，1：周末
    work_load_mean=ana_data[ana_data['is_holiday']==0].power_load.mean()
    holiday_load_mean=ana_data[ana_data['is_holiday']==1].power_load.mean()
    ax4 = fig.add_subplot(414)
    ax4.bar(['工作日','周末'], [work_load_mean, holiday_load_mean])
    ax4.set_title('工作日与周末平均负荷对比')

    plt.savefig('../diagrams/电力负荷数据分析图')
    plt.show()

#特征工程
def feature_engineering(data,logger):
    feature_data = data.copy()

    #1.提取出小时和月份
    feature_data['hour']=feature_data['time'].str[11:13]
    feature_data['month']=feature_data['time'].str[5:7]
    #one-hot热编码
    feature_data=pd.get_dummies(feature_data,columns=['hour','month'])

    #将热编码后的bool值转成0，1
    dummy_cols = [col for col in feature_data.columns
                  if col.startswith('hour_') or col.startswith('month_')]

    feature_data[dummy_cols] = feature_data[dummy_cols].astype(int)

    #2.提取出相近时间窗口中的负荷特征
    load_1h_ago=feature_data['power_load'].shift(1).rename('前1小时') #前1个小时的负荷
    load_2h_ago=feature_data['power_load'].shift(2).rename('前2小时') #前2个小时的负荷
    load_3h_ago=feature_data['power_load'].shift(3).rename('前3小时') #前3个小时的负荷
    feature_data=pd.concat([feature_data,load_1h_ago,load_2h_ago,load_3h_ago],axis=1)

    #3.提取出昨日同时刻的负荷特征

    #给特征新增1列，yesterday_time
    # feature_data['yesterday_time']=feature_data['time'].apply(lambda x:(pd.to_datetime(x)-datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    #把所有的日期和负荷拼接成字典，方便查找
    # time_load_dict=feature_data.set_index('time')['power_load'].to_dict()
    # feature_data['yesterday_load']=feature_data['yesterday_time'].apply(lambda x:time_load_dict[x])

    #上面的三行代码可以直接由下一行代码实现
    feature_data['yesterday_load'] = feature_data['power_load'].shift(24)

    #4.剔除空样本
    feature_data=feature_data.dropna()

    #5.整理时间特征，并返回
    feature_columns=[col for col in feature_data.columns if col not in ['time','power_load']]
    return feature_data,feature_columns

#模型训练,评估及保存
def model_training(data,features,logger):
    x=data[features]
    y=data['power_load']

    #划分数据集为训练集和测试集，最好不要用train_test_split，因为它是随机划分的，对于时间序列数据，随机划分可能会导致未来预测过去，导致数据泄露,或者使用train_test_split，设置其中的参数shuffle为False
    split_idx = int(len(feature_data) * 0.8)

    train = feature_data.iloc[:split_idx]
    test = feature_data.iloc[split_idx:]

    x_train = train[feature_columns]
    y_train = train['power_load']

    x_test = test[feature_columns]
    y_test = test['power_load']

    #寻找最优超参（网格搜索与交叉验证）
    logger.info('-'*6+'网格搜索+交叉验证 寻找最优超参组合'+'-'*6)
    logger.info(f'开始时间:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    #定义参数字典
    params_dict={'n_estimators':[200,300,400,500,600],'learning_rate':[0.005,0.01,0.05,0.1,0.2],'max_depth':[3,5,7,9]}

    #创建XGBoost对象
    estimator=XGBRegressor()
    gs=GridSearchCV(param_grid=params_dict,estimator=estimator,cv=5)
    gs.fit(x_train,y_train)

    #打印最优参数组合
    logger.info(f'最优参数组合为：{gs.best_params_}')
    logger.info(f'结束时间:{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    gs.fit(x_train, y_train)
    y_pred = gs.predict(x_test)

    # 模型评估
    print(f'均方误差为:{mean_squared_error(y_test, y_pred)}')
    print(f'均方根误差为:{root_mean_squared_error(y_test, y_pred)}')
    print(f'平均绝对误差为:{mean_absolute_error(y_test, y_pred)}')
    print(f'平均绝对百分比误差为:{mean_absolute_percentage_error(y_test, y_pred)}')

    joblib.dump(gs,'../model/model.pkl')
    logger.info(f'模型保存成功，保存路径为:{os.path.abspath("../model/model.pkl")}')


if __name__ == '__main__':
    pm=PowerLoadModel('../data/train.csv')
    logger=pm.logfile
    logger.info('开始创建 电力负荷模型类 对象')

    #查看数据分布
    #analysis_data(pm.data_source)

    #特征工程
    feature_data,feature_columns = feature_engineering(pm.data_source, logger)

    #模型训练
    model_training(feature_data,feature_columns,logger)