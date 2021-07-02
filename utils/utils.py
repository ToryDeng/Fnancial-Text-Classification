from typing import Union, List
import os
import tensorflow as tf
import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras_bert import Tokenizer
from stylecloud import gen_stylecloud
from config import cfg


def setup():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.style.use("ggplot")
    # for low memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def preprocess_raw_data(save_info=True):
    # 导入三个数据集
    train_data = pd.read_csv("data/event_type_entity_extract_train.csv", header=None, usecols=[1, 2])
    eval_data = pd.read_csv("data/event_type_entity_extract_eval.csv", header=None, usecols=[1, 2])
    test_data = pd.read_csv("data/event_type_entity_extract_test.csv", header=None, usecols=[1, 2])

    # 合并，预处理
    all_data = pd.concat([train_data, eval_data, test_data]).drop_duplicates().reset_index(drop=True)  # 三行重复
    all_data.rename(columns={1: 'content', 2: 'type'}, inplace=True)
    all_data['content'] = all_data['content'].str.strip()
    all_data['content'] = all_data['content'].str.replace(
        r"【[\u4e00-\u9fa5]{1,2}】|【?[0-9OI\-]{9,20}】?|【?w{0,3}.\w+.[a-z]{2,3}】?|\[\[\+_\+]]|\(\d{6}\)", "", regex=True
    )  # 【头条】、【电话】、【网址】、(股票代码)
    all_data['type'] = all_data['type'].replace('公司股市异常', '股市异常')
    wrong_types = all_data['type'][~all_data['type'].duplicated(keep=False)].values
    all_data = all_data[~all_data['type'].isin(wrong_types)].replace('', np.nan).dropna()

    if save_info:
        # 文本类型计数柱状图
        type_counts = all_data['type'].value_counts(ascending=True)
        ax = type_counts.plot(kind='barh', figsize=(6, 9))
        [plt.text(num + 20, i, num, ha='left', va='center') for i, num in enumerate(type_counts.values)]
        ax.set(**{'xlabel': '数量'})
        plt.tight_layout()
        plt.savefig("img/type_counts.jpg", dpi=150, bbox_inches='tight')
        plt.close()

        # 保存20条样本记录
        all_data.sample(20).to_csv('data/data_samples.csv')

        # 文本长度直方图
        bin_cut = pd.cut(
            all_data['content'].map(lambda x: len(x)), bins=[0, 50, 100, 1000, 5000], right=False
        ).value_counts(sort=False)
        ax = bin_cut.plot(kind='bar', rot=0, color='green')
        [plt.text(i, num + 200, num, ha='center', va='bottom') for i, num in enumerate(bin_cut.values)]
        ax.set(**{'xlabel': '数量', 'ylabel': '长度', 'ylim': [0, 1e5]})
        plt.savefig("img/content_len.jpg", dpi=150, bbox_inches='tight')
        plt.close()

    # 带停用词的分词， 保存
    jieba.load_userdict('data/companies.txt')  # 加载搜狗细胞词库转化而成的上市公司名称
    jieba.add_word('企查查')  # '企查查'、'天眼查'这两个APP在金融文本中较为常见，且容易分错
    jieba.add_word('天眼查')
    stopwords = [line.strip() for line in open('data/停用词.txt', encoding='gbk').readlines()]
    all_data['sep_content'] = all_data['content'].map(
        lambda x: ' '.join([wd for wd in jieba.cut(x) if len(wd) > 1 and wd not in stopwords])
    )
    all_data.replace('', np.nan).dropna().to_csv("data/cleaned_data.csv")  # 仅含停用词的行变成空行，需要被删掉
    print("数据预处理已完成！")


def plot_word_cloud():
    data = pd.read_csv("data/cleaned_data.csv", index_col=0)
    all_content = ' '.join(data['sep_content'].to_list())
    gen_stylecloud(text=all_content,
                   font_path="C:/Windows/Fonts/simhei.ttf",
                   collocations=False,  # 避免关键词重复
                   size=1024,
                   custom_stopwords=['公司', '有限公司'],  # '有限公司'、'公司'对分析没有意义
                   icon_name='fas fa-hand-holding-usd',
                   output_name="img/wordcloud.jpg")
    print('词云图已生成！')


def train_val_test_split(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, shuffle=True,
                                                                test_size=cfg.test_size,
                                                                random_state=cfg.random_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, shuffle=True,
                                                      test_size=cfg.val_size / (1 - cfg.test_size),
                                                      random_state=cfg.random_seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_D2M():
    data = pd.read_csv("data/cleaned_data.csv", index_col=0, usecols=[0, 2, 3])
    countvec = CountVectorizer(max_features=cfg.max_features)
    wordmtx = countvec.fit_transform(data['sep_content'])
    return train_test_split(wordmtx, data.type.values, test_size=cfg.test_size, random_state=cfg.random_seed)


def load_bert_data(tokenizer, nrows=None):
    def seq_padding(X, padding=0):  # 填充序列，默认填充0
        L = [len(x) for x in X]  # 长度列表
        ML = max(L)  # 最大长度
        return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

    X1, X2, Y = [], [], []
    if nrows is not None:
        data = pd.read_csv("data/cleaned_data.csv", index_col=0, usecols=[0, 1, 2], nrows=nrows).dropna()
    else:
        data = pd.read_csv("data/cleaned_data.csv", index_col=0, usecols=[0, 1, 2]).dropna()
    le = LabelEncoder()
    labels = to_categorical(le.fit_transform(data['type']))

    for i in range(data.shape[0]):
        text = data.iloc[i, 0][:cfg.sequence_len]  # 截取到最大长度，可能有的评论小于这个长度
        x1, x2 = tokenizer.encode(first=text)
        X1.append(x1)
        X2.append(x2)  # 第几句，应该都是0
        Y.append(labels[i])

    X1 = seq_padding(X1)  # 开始填充
    X2 = seq_padding(X2)
    Y = seq_padding(Y)

    # 分割训练集、验证集、测试集
    X1_train, X1_val, X1_test, y_train, y_val, y_test = train_val_test_split(X1, Y)
    X2_train, X2_val, X2_test, *rest = train_val_test_split(X2, Y)
    return X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test


def evaluate(model, X_test: Union[np.ndarray, List[np.ndarray]], y_test: np.ndarray) -> pd.DataFrame:
    if isinstance(model, naive_bayes.MultinomialNB):
        model_name = 'Naive Bayes'
    elif isinstance(model, LogisticRegression):
        model_name = 'Logistic Regression'
    elif isinstance(model, Sequential):
        model_name = 'LSTM'
    elif isinstance(model, Model):
        model_name = 'RBTL3'
    else:
        raise NotImplementedError("This model can not use evaluate function.")

    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    eval_result = pd.DataFrame(
        data=[accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')],
        columns=[model_name], index=['Accuracy', 'F1 score']
    )
    if model_name == 'RBTL3':
        print(eval_result)
        eval_result.to_csv('results/bert_result.csv')
    print(f"{model_name} has been evaluated.")
    return eval_result


class MyTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R
