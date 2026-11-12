# 这是一个测试文档 用于测试各个安装包的安装情况
# 由 @Classmateliu 编写

# 库的import部分 ------------------------------------------------------------
import nltk

import sklearn

import pandas as pd

import numpy as np
import string
import json
import re

# NLTK库测试部分 ------------------------------------------------------------
#nltk.download()  # 打开NLTK下载器
from nltk.corpus import brown    # 导入brown语料库
print(brown.words())             # 查看brown库中所有的单词
print(brown.categories())        # 通过categories()函数查看brown中包含的类别
print('brown中一共有{}个句子'.format(len(brown.sents())))  
# 通过sents()函数查看brown中包含的句子
print('brown中一共有{}个单词'.format(len(brown.words())))
# 通过words()函数查看brown中包含的单词

# sklearn库测试部分 ------------------------------------------------------------
print(sklearn.__version__)

# pandas库测试部分 ------------------------------------------------------------
pd.__version__  # 查看版本
mydataset = {
  'sites': ["Google", "Runoob", "Wiki"],
  'number': [1, 2, 3]
}

myvar = pd.DataFrame(mydataset)
print(myvar)

# numpy库测试部分 ------------------------------------------------------------
print(np.__version__)

