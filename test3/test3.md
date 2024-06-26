```python
import time
time.sleep(3)

```


```python
import numpy as np
def square(x):
    return x * x
```


```python
x = np.random.randint(1, 10)
y = square(x)
print('%d squared is %d' % (x, y))
```

    7 squared is 49
    


```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:\\Users\\WSR\\Desktop\\software_practice\\3\\fortune500.csv')

```


```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']

```


```python
len(df)

```




    25500




```python
df.dtypes

```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numberic_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numberic_profits].head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>228</th>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>290</th>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>294</th>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>296</th>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>352</th>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df.profit[non_numberic_profits])

```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numberic_profits], bins=range(1955, 2006))

```


    
![png](output_9_0.png)
    



```python
df = df.loc[~non_numberic_profits]
df.profit = df.profit.apply(pd.to_numeric)

```


```python
len(df)

```




    25131




```python
df.dtypes

```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x=0, y=0)

```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')

```


    
![png](output_14_0.png)
    



```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (millions)')

```


    
![png](output_15_0.png)
    



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha=0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols=2)
title = 'Increase in mean and std Fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14, 4)
fig.tight_layout()

```


    
![png](output_16_0.png)
    



```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        # 寻找最小元素的索引
        min_index = i
        for j in range(i+1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        # 交换元素
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

```


```python
def test():
    # 用户输入数据
    data_input = input("请输入要排序的数字，用空格分隔：")
    # 将输入的数据转换为列表
    arr = list(map(int, data_input.split()))
    # 调用selection_sort函数进行排序
    sorted_arr = selection_sort(arr)
    # 输出排序结果
    print("排序后的结果：", sorted_arr)

```


```python
# 定义选择排序函数
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        # 寻找最小元素的索引
        min_index = i
        for j in range(i+1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        # 交换元素
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

# 定义测试函数
def test():
    # 用户输入数据
    data_input = input("请输入要排序的数字，用空格分隔：")
    # 将输入的数据转换为列表
    arr = list(map(int, data_input.split()))
    # 调用selection_sort函数进行排序
    sorted_arr = selection_sort(arr)
    # 输出排序结果
    print("排序后的结果：", sorted_arr)

# 调用测试函数
test()


```

    请输入要排序的数字，用空格分隔： 1 6 2 3 8 5 9
    

    排序后的结果： [1, 2, 3, 5, 6, 8, 9]
    


```python

```
