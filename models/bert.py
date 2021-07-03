import os
from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, get_custom_objects
import codecs
from utils.utils import MyTokenizer, load_bert_data, evaluate
from config import cfg


def run_bert():
    token_dict = {}
    with codecs.open(cfg.bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = MyTokenizer(token_dict)
    X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = load_bert_data(tokenizer)
    print(f"There are {X1_test.shape[0]} samples in test set.")
    if f'rbtl3-{cfg.bert_epoch}.h5' in os.listdir('ckpt/'):
        model = load_model(f'ckpt/rbtl3-{cfg.bert_epoch}.h5', custom_objects=get_custom_objects())
    else:
        bert_model = load_trained_model_from_checkpoint(cfg.bert_cfg_path, cfg.bert_ckpt_path, seq_len=None)

        for layer in bert_model.layers:
            layer.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda y: y[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        p = Dense(y_train.shape[1], activation='softmax')(x)

        model = Model([x1_in, x2_in], p)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(cfg.bert_lr),  # 用足够小的学习率
            metrics=['accuracy']
        )
        model.fit(x=[X1_train, X2_train], y=y_train,
                  validation_data=([X1_val, X2_val], y_val),
                  epochs=cfg.bert_epoch, batch_size=cfg.bert_batch_size)
        model.save(f'ckpt/rbtl3-{cfg.bert_epoch}.h5')
    evaluate(model, [X1_test, X2_test], y_test)
