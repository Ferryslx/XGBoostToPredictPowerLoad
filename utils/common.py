import numpy as np
import pandas as pd

"""对数据做预处理->时间格式化，按照时间升序，且对数据去重"""

def data_preprocessing(filename):
    data=pd.read_csv(filename)

    #时间格式化，转为:'%Y-%m-%d %H:%M:%S'
    data['time']=pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    #按照时间升序排列
    data.sort_values('time',ascending=True,inplace=True)

    #去重
    data.drop_duplicates(inplace=True)

    return data

if __name__ == '__main__':
    data_preprocessing()