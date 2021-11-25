'''
pip install openpyxl
pip install tushare
'''
import tushare as ts
from multiprocessing.pool import ThreadPool

ts.set_token('9203526c35de212c7bb198947d2961fb20cf93beafcf547f3e50b24f')
pro = ts.pro_api()

ts_code = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')['ts_code'].values

def df(ts_code):
    a = pro.daily(ts_code=ts_code, start_date='19900101', end_date='20200101')
    a.to_excel(str(ts_code)+'.xlsx')
    print(ts_code+'done')
# The number of threads
pool_size = 15
'''
A thread pool object which controls a pool of worker threads to which jobs can be submitted. 
ThreadPool instances are fully interface compatible with Pool instances, 
and their resources must also be properly managed, 
either by using the pool as a context manager or by calling close() and terminate() manually.
'''
pool = ThreadPool(pool_size)
pool.map(df, ts_code)
pool.close()
pool.join()
