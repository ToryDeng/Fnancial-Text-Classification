import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils.utils import preprocess_raw_data, plot_word_cloud, setup
from models.baseline import run_baselines
from models.bert import run_bert

setup()
preprocess_raw_data()
plot_word_cloud()
run_baselines()
run_bert()
