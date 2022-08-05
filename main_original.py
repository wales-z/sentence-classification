

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 23:55:56 2022

@author: why
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import re
from gensim.models import Word2Vec
from wordcloud import WordCloud
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.model_selection import GridSearchCV            #网格搜索
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix,classification_report #预测效果的评估
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import xgboost, numpy, textblob, string
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers, models, optimizers
import multiprocessing
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping


plt.rcParams['font.sans-serif']=['SimHei']  #这两行用于plt图像显示中文，否则plt无法显示中文
plt.rcParams['axes.unicode_minus']=False
# sentences = [["粉", "砂锅土豆粉", "砂锅米线"], ["肉", "鸡腿", "炸鸡排"]]
# model = Word2Vec(sentences, min_count=1)


def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*:", "", text)  # 去除正文中的@和回复/转发中的用户名
    # text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    text = text.replace("转发微博", "")  # 去除无意义的词语
    text = text.replace("O网页链接?", "")
    text = text.replace("?展开全文c", "")
    text = text.replace("网页链接", "")
    text = text.replace("展开全文", "")
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./，。、《》？；：‘“{【】}|、！@#￥%……&*（）——+=-"
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    return text.strip()

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


train_df = pd.read_csv('nCoV_100k_train.labled.csv')
example = train_df
test_df = pd.read_csv('nCov_10k_test.csv')

# text = "征战四海只为今日一胜，我不会再败了。"
# # jieba.cut直接得到generator形式的分词结果
# seg = jieba.cut(text)  
# print(' '.join(seg)) 

# # 也可以使用jieba.lcut得到list的分词结果
# seg = jieba.lcut(text)
# print(seg)

train_df.columns = ['id', 'time', 'name', 'content', 'image', 'video', 'label']
train_df = train_df[train_df['label'].isin(['-1', '0', '1'])]
train_df = train_df.reset_index(drop=True)

# 缺失值处理
train_df["content"] = train_df["content"].fillna('')
# 数据清洗
train_df["content"] = train_df["content"].apply(clean)
# 删除重复数据
train_df = train_df[['content', 'label']].drop_duplicates()

# plt.rcParams['font.sans-serif'] = ['SimHei']   #解决中文显示问题
# plt.rcParams['axes.unicode_minus'] = False    # 解决中文显示问题
# s=[list(train_df['label']).count('1'),list(train_df['label']).count('0'),list(train_df['label']).count('-1')]
# l=['积极','中性','消极']
# plt.pie(s,labels=l,autopct='%3.1f%%')
# plt.title('情感倾向分布饼图')

# 去除中性数据
train_df=train_df[train_df['label']!='0']
# S=[list(train_df['label']).count('1'),list(train_df['label']).count('-1')]
# L=['积极','消极']
# plt.pie(S,labels=L,autopct=make_autopct(S))
# plt.title('情感倾向分布饼图')

#停用词读取
with open('stopwords.txt','r',encoding='utf-8') as f:
    stopwords = f.readlines()
stopwords = [i.strip() for i in stopwords]
#结巴分词并去除停用词
def jieba_stop(each):
    result = []
    jieba_line = jieba.lcut(each)  #jieb.cut()返回迭代器，节省空间；jieba.lcut() 直接返回分词后的列表。三种分词模式：默认精确模式，用于文本分析  
    for l in jieba_line:
        if (l not in stopwords) and (l != ''): # 去除停用词以及空格
            result.append(l.strip())
    return result
train_df.content = train_df.content.apply(jieba_stop)  #注意 apply 的用法
train_df.content = train_df.content.apply(lambda x: ','.join([i for i in x if i !=''])) #反复去除空字符

all_words = [i.strip() for line in train_df.content for i in line.split(',')]
all_df = pd.DataFrame({'words':all_words})
all_df.groupby(['words'])['words'].count().sort_values(ascending = False)[:10].plot.bar()  #表示停用词去除完全，分词效果尚可


#数据集划分
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['content'], train_df['label'])
# label编码为目标变量
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)         
 
#词语级tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train_df['content'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
 
# ngram 级tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(train_df['content'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
 
#词性级tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(train_df['content'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
# fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
 
# predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
 
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
 
    return metrics.accuracy_score(predictions, valid_y)

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print( "NB, WordLevel TF-IDF: ", accuracy)

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)

accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors: ", accuracy)

#特征为词语级别TF-IDF向量的RF
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)


#特征为词语级别TF-IDF向量的Xgboost
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print ("Xgb, WordLevel TF-IDF: ", accuracy)

#特征为词性级别TF-IDF向量的Xgboost
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print ("Xgb, CharLevel Vectors: ", accuracy)

# def create_model_architecture(input_size):
# # create input layer
#     input_layer = layers.Input((input_size, ), sparse=True)

# # create hidden layer
#     hidden_layer = layers.Dense(100, activation="relu")(input_layer)

# # create output layer
#     output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

#     classifier = models.Model(inputs = input_layer, outputs = output_layer)
#     classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
#     return classifier

# classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
# accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
# print ("NN, Ngram Level TF IDF Vectors", accuracy)

MAX_NB_WORDS = 90000
MAX_SEQUENCE_LENGTH = 60

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_df['content'].values)

X = tokenizer.texts_to_sequences(train_df['content'].values)
#填充X,让X的各个列的长度统一
X = pad_sequences(X, MAX_SEQUENCE_LENGTH)

#多类标签的onehot展开
Y = pd.get_dummies(train_df['label']).values



#拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

EMBEDDING_DIM = 256

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1])) # 嵌入层将维数降到128
model.add(Bidirectional(LSTM(64))) # 双向LSTM层
model.add(Dropout(0.4)) # 随机失活
model.add(Dense(2,activation='softmax')) # 稠密层 将情感分类0或1
model.compile('adam','binary_crossentropy',metrics=['accuracy']) # 二元交叉熵

epochs = 20
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


y_pred = model.predict(X_test)
for i in range(len(y_pred)):
    max_value = max(y_pred[i])
    for j in range(len(y_pred[i])):
        if max_value == y_pred[i][j]:
            y_pred[i][j] = 1
        else:
            y_pred[i][j] = 0
# target_names = ['负面', '正面']
print(classification_report(Y_test, y_pred))