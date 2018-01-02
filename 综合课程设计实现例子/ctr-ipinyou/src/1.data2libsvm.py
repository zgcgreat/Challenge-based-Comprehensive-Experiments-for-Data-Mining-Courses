# _*_ coding: utf-8 _*_

import collections
import operator
from csv import DictReader
from datetime import datetime

'''
将csv格式数据转化为libsvm格式数据，相当于以稀疏矩阵的格式存储one-hot编码格式数据
仅保存one-hot编码后值为1的位置，节省内存空间
'''

train_path = '../data/train.csv'
test_path = '../data/test.csv'
train_libsvm = '../output/train.libsvm'
test_libsvm = '../output/test.libsvm'
vali_path = '../output/validation.csv'
feature_index = '../output/feat_index.txt'

field = ['hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
         'slotheight', 'slotvisibility', 'slotformat', 'creative', 'keypage', 'usertag']

table = collections.defaultdict(lambda: 0)


# 为特征名建立编号, filed
def field_index(x):
    index = field.index(x)
    return index


# 为特征值编号
def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

# 为每个特征增加一个新特征值other， 在测试集中如果有特征值在训练集中没出现过，则将这个特征值变为other
for f in field:
    getIndices(str(field_index(f))+':other')


# 训练集
with open(train_libsvm, 'w') as outfile:
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = str(field_index(k)) + ':' + v
                    features.append('{0}:1'.format(getIndices(kv)))

        if e % 100000 == 0:
            print(datetime.now(), 'creating train.libsvm...', e)
            break
        outfile.write('{0} {1}\n'.format(row['click'], ' '.join('{0}'.format(val) for val in features)))

with open(test_libsvm, 'w') as f1, open(vali_path, 'w') as f2:
    f2.write('id,label'+'\n')
    for t, row in enumerate(DictReader(open(test_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = str(field_index(k)) + ':' + v
                    if kv in table.keys():
                        features.append('{0}:1'.format(getIndices(kv)))
                    else:
                        kv = str(field_index(k)) + ':' + 'other'
                        features.append('{0}:1'.format(getIndices(kv)))

        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.libsvm...', t)
            break
        f1.write('{0} {1}\n'.format(row['click'], ' '.join('{0}'.format(val) for val in features)))
        f2.write(str(t) + ',' + row['click'] + '\n')


# 将特征及其对应的编号写入文件保存起来
featvalue = sorted(table.items(), key=operator.itemgetter(1))
fo = open(feature_index, 'w')
for fv in featvalue:
    fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
fo.close()
