import pandas as pd
import numpy as np
from datetime import *
import datetime as dtime
import time


biensoxe = '32-D3 1111'
pathin = "123.jpg"
pathout = "1233.jpg"


def xevao():
    data_XeVao = pd.read_csv("./reservedData/CacheFile.csv")
    data = {'key': biensoxe, 'timein': datetime.now().strftime('%d/%m/%Y %H:%M:%S'), 'timeout': 'None',
            'pathin': pathin, 'pathout': 'None',
            'money': '0'}
    data_XeVao = data_XeVao.append(pd.Series(data, index=['key', 'timein', 'timeout', 'pathin', 'pathout', 'money']),
                                   ignore_index=True)
    data_XeVao.to_csv('./reservedData/CacheFile.csv', index=False)


def xera():
    cache = pd.read_csv("./reservedData/CacheFile.csv")
    archive = pd.read_csv("./reservedData/ArchiveFile.csv")
    for i in range(cache.shape[0]):
        if cache.at[i, 'key'] == biensoxe:
            timeout = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            cache.at[i, 'timeout'] = timeout
            cache.at[i, 'pathout'] = '15.jpg'
            cache.at[i, 'money'] = (pay_bill(cache.at[i, 'timein'], timeout))
            archive = archive.append(cache.loc[cache['key'] == biensoxe], ignore_index=True)
            cache = cache[cache.key != biensoxe]
            break
    cache.to_csv('./reservedData/CacheFile.csv', index=False)
    archive.to_csv('./reservedData/ArchiveFile.csv', index=False)


def pay_bill(timein, timeout):
    pay_time = datetime.strptime(timeout, "%d/%m/%Y %H:%M:%S") - datetime.strptime(timein, "%d/%m/%Y %H:%M:%S")
    if pay_time < dtime.timedelta(seconds=21600):
        return 5
    elif pay_time < dtime.timedelta(seconds=43200):
        return 10
    elif pay_time < dtime.timedelta(seconds=86400):
        return 15
    else:
        return 15 + (pay_time.days * 10)


if __name__ == '__main__':
    timeStart = datetime.now()
    # xevao()
    xera()
    timeStop = datetime.now()
    print('timeRun = ', timeStop - timeStart)
