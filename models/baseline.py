import os
import pandas as pd
import tensorflow as tf
from utils.utils import load_D2M, evaluate, train_val_test_split
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import to_categorical

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression

from config import cfg

tf.compat.v1.enable_eager_execution()


def run_naive_bayes():
    X_train, X_test, y_train, y_test = load_D2M()
    NB_model = naive_bayes.MultinomialNB()
    NB_model.fit(X_train, y_train)
    return evaluate(NB_model, X_test, y_test)


def run_logistic():
    X_train, X_test, y_train, y_test = load_D2M()

    scalar = MaxAbsScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    logistic_model = LogisticRegression(max_iter=2000)
    logistic_model.fit(X_train, y_train)
    return evaluate(logistic_model, X_test, y_test)


def run_lstm():
    raw_data = pd.read_csv("data/cleaned_data.csv", index_col=0, usecols=[0, 2, 3])

    tokenizer = Tokenizer(num_words=cfg.max_features)
    tokenizer.fit_on_texts(texts=raw_data['sep_content'].to_list())
    sequences = tokenizer.texts_to_sequences(texts=raw_data['sep_content'].to_list())  # 转化为整数数列
    seq_data = pad_sequences(sequences, maxlen=cfg.sequence_len)  # 截取序列

    le = LabelEncoder()
    labels = to_categorical(le.fit_transform(raw_data['type']))

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(seq_data, labels)

    if f'lstm-{cfg.lstm_epoch}.h5' in os.listdir('ckpt/'):
        model = load_model(f'ckpt/lstm-{cfg.lstm_epoch}.h5')
    else:
        model = Sequential()
        model.add(Embedding(input_dim=cfg.max_features + 1, output_dim=32, input_length=cfg.sequence_len))
        model.add(LSTM(units=200, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dropout(0.2))
        # model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=labels.shape[1], activation='softmax'))
        # model.summary()
        # f1_weighted = tfa.metrics.F1Score(num_classes=labels.shape[1], average='macro', threshold=0.5)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=cfg.lstm_epoch,
                  batch_size=cfg.lstm_batch_size, verbose=2)
        model.save(f'ckpt/lstm-{cfg.lstm_epoch}.h5')
    return evaluate(model, X_test, y_test)


def run_baselines():
    NB_result = run_naive_bayes()
    logi_result = run_logistic()
    lstm_result = run_lstm()

    baseline_results = pd.concat([NB_result, logi_result, lstm_result], axis=1)
    print(baseline_results)
    baseline_results.to_csv('results/baseline_results.csv')
