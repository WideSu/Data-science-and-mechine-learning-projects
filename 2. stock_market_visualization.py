import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import matplotlib.dates as mdates

SH_df = pd.read_csv('/Users/yinxiangyang/Desktop/000001.csv')
Iflytek_df = pd.read_csv('/Users/yinxiangyang/Desktop/002030.csv')
VisualChina_df = pd.read_csv('/Users/yinxiangyang/Desktop/000681.csv')
Moutai_df = pd.read_csv('/Users/yinxiangyang/Desktop/600519.csv')
CITIC_df = pd.read_csv('/Users/yinxiangyang/Desktop/600300.csv')


df_dict = {"SH": SH_df, "Iflytek": Iflytek_df, "VisualChina": VisualChina_df, "Moutai": Moutai_df, "CITIC": CITIC_df}
n = 1
for df in df_dict.values():
    df = df.iloc[::-1]
    sns.set()
    warnings.filterwarnings('ignore')
    plt.figure(figsize=(24, 14), dpi=100, num=4)
    # volume chart
    x = df['trade_date']
    y1 = df['vol']
    plt.plot(y1, color='r', label='volume nums', linewidth=2)
    plt.xlabel('date', fontsize=15)
    plt.ylabel('volume', fontsize=15)
    plt.title('volume chart', fontsize=15)
    print('volume chart')
    plt.savefig(f'figure{n}.jpg')
    plt.show()
    n += 1
    # OHLC chart
    plt.figure(figsize=(24, 14), dpi=100, num=4)
    x = df['trade_date']
    plt.plot(df.loc[:, 'open'], label='open price', linewidth=2)
    plt.plot(df.loc[:, 'high'], label='high price', linewidth=2)
    plt.plot(df.loc[:, 'low'], label='low price', linewidth=2)
    plt.plot(df.loc[:, 'close'], label='close price', linewidth=2)
    plt.xlabel('date', fontsize=15)
    plt.ylabel('price/exchange', fontsize=15)
    plt.title('OHLC chart', fontsize=15)
    print('OHLC chart')
    plt.savefig(f'figure{n}.jpg')
    plt.show()
    n += 1
