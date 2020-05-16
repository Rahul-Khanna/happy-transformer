import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import sys
sys.path.append('../')
from happytransformer import HappyROBERTA
if len(sys.argv):
    model_name = sys.argv[0]
else:
    model_name = "roberta-base"

roberta = HappyROBERTA(model_name)

word_prediction_args = {
"batch_size": 1,

"epochs": 20,

"lr": 1e-5,

"adam_epsilon": 1e-6

}

roberta.init_train_mwp(word_prediction_args)

train_path = '../../data/finetune_data/train_sentences.txt'
eval_path = '../../data/finetune_data/eval_sentences.txt'
train_masked_path = '../../data/finetune_data/train_m_sentences.txt'
eval_masked_path = '../../data/finetune_data/eval_m_sentences.txt'
output_dir = '../../data/finetune_data/{}'.format(model_name)

roberta.train_mwp(train_path, eval_path, train_masked_path, eval_masked_path, output_dir)