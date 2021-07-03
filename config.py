class Config:
    def __init__(self):
        self.random_seed = 2021
        # preprocessing params
        self.max_features = 5000
        self.sequence_len = 128
        self.val_size = 0.2
        self.test_size = 0.2

        # training params
        # LSTM specific
        self.lstm_epoch = 10
        self.lstm_batch_size = 32
        # BERT specific
        self.bert_epoch = 2
        self.bert_lr = 2e-5
        self.bert_batch_size = 32
        self.bert_cfg_path = 'ckpt/bert_config_rbtl3.json'
        self.bert_ckpt_path = 'ckpt/bert_model.ckpt'
        self.bert_dict_path = 'ckpt/vocab.txt'


cfg = Config()
