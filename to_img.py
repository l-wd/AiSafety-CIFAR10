# -*- coding: utf-8 -*-
import imageio  # 引入imageio包
import numpy as np
import pickle

# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

# 生成测试集图片
testXtr = unpickle("data/cifar-10-batches-py/test_batch")
for i in range(0, 1000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'images/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
    imageio.imsave(picName, img)
print("test_batch loaded.")