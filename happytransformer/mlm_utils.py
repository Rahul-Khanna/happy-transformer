"""
BERT and ROBERTA masked language model fine-tuning:

Credit: This code is a modified version of the code found in this repository
under
        https://github.com/huggingface/transformers/blob/master/examples
        /run_lm_finetuning.py

"""

import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm import trange
from tqdm.notebook import tqdm_notebook
from transformers import (AdamW)

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    from transformers import WarmupLinearSchedule \
        as get_linear_schedule_with_warmup

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Used to turn .txt file into a suitable dataset object
    """

    def __init__(self, tokenizer, file_path, mask_path, block_size=512):
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        self.examples = []
        for line in lines:
            tokenized_text = tokenizer.encode(line, max_length=block_size,
                                              add_special_tokens=True, pad_to_max_length=True)  # Get ids from text
            self.examples.append(tokenized_text)

        with open(mask_path, encoding="utf-8") as f:
            text = f.read()
        words = text.split("\n")
        self.positions_to_mask = get_masked_position_per_sentence(lines, words, tokenizer, block_size)
        print(self.positions_to_mask[0:10])
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def set_seed(seed=42):
    """
    Sets seed for all random number generators available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        print('Cuda manual seed is not set')


def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original.
    * The standard implementation from Huggingface Transformers library *
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    # MLM Prob is 0.15 in examples
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in
        labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with
    # tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens
    # unchanged
    return inputs, labels

def custom_mask_tokens(inputs, tokenizer, positions_to_mask):
    """ Prepare masked tokens inputs/labels for masked language modeling
        Assumes batch-size of 1
    """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, 0.0)
    masked_positions = [1 if i in positions_to_mask else 0 for i in range(labels.shape[0])]

    probability_matrix.masked_fill_(torch.tensor(
        masked_positions, dtype=torch.bool), value=1.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with
    # tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

def get_masked_position_per_sentence(sentences, words_to_mask, tokenizer, block_size):
    """
        Given a masked sentence returns the position of the mask for each sentence
    """

    masked_positions = []
    for i, sentence in enumerate(sentences):
        tokens = tokenizer.tokenize(sentence, max_length=block_size,
                                  add_special_tokens=True, pad_to_max_length=True)
        
        words_to_mask_sent = words_to_mask[i].split(",")
        masked_positions_sent = []
        for j, val in enumerate(tokens):
            for word in words_to_mask_sent:
                if word in val:
                    masked_positions_sent.append(j)

        masked_positions.append(set(masked_positions_sent))

    return masked_positions

def save_model_checkpoint(model, optimizer, global_step, epoch_info, file_name):
    """
        Function to create a checkpoint storing model and optimizer progress,
        the number of steps taken and the stats so far
    """
    output = {
              "model"       : model.state_dict(),
              "optimizer"   : optimizer.state_dict(),
              "global_step" : global_step + 1,
              "epoch_info" : epoch_info
            }
    torch.save(output, file_name)

def eval_and_save_model(output_dir, eval_dataset, global_step, epoch_info, model, optimizer, tokenizer):
    
    # adds / to output_dir
    full_output_dir = os.path.join(output_dir, 'save_step_{}'.format(global_step))
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    output_model_file = os.path.join(full_output_dir, "checkpoint.pt")
    info = evaluate(model, tokenizer, eval_dataset, batch_size=1)
    
    epoch_info.append(info)

    save_model_checkpoint(model, optimizer, global_step, epoch_info, output_model_file)

    return epoch_info

def train(model, tokenizer, train_dataset, eval_dataset, batch_size, lr, adam_epsilon,
          epochs, output_dir):
    """

    :param model: Bert Model to train
    :param tokenizer: Bert Tokenizer to train
    :param train_dataset:
    :param batch_size: Stick to 1 if not using using a high end GPU
    :param lr: Suggested learning rate from paper is 5e-5
    :param adam_epsilon: Used for weight decay fixed suggested parameter is
    1e-8
    :param epochs: Usually a single pass through the entire dataset is
    satisfactory
    :return: Loss
    """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size)
    train_positions_to_mask = train_dataset.positions_to_mask

    t_total = len(train_dataloader) // batch_size  # Total Steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, t_total)

    # ToDo Case for fp16

    # Start of training loop
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)

    model.train()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    train_iterator = trange(int(epochs), desc="Epoch")
    epoch_info = []
    proceed = False
    tmp_global_step = 0
    for _ in train_iterator:
        epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
        for i, batch in enumerate(epoch_iterator):
            
            if tmp_global_step >= global_step or proceed:
                proceed = True
            else:
                tmp_global_step += 1
            
            if proceed:
                inputs, labels = custom_mask_tokens(batch, tokenizer, train_positions_to_mask[i])
                inputs = inputs.to('cuda')  # Don't bother if you don't have a gpu
                labels = labels.to('cuda')

                outputs = model(inputs, masked_lm_labels=labels)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                loss.backward()
                tr_loss += loss.item()

                # if (step + 1) % 1 == 0: # 1 here is a placeholder for gradient
                # accumulation steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        if proceed:
            epoch_info = eval_and_save_model(output_dir, eval_dataset, global_step, epoch_info, model, optimizer, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            tr_loss = 0

    return model, tokenizer


def create_dataset(tokenizer, file_path, mask_path, block_size=512):
    """
    Creates a dataset object from file path.
    :param tokenizer: Bert tokenizer to create dataset
    :param file_path: Path where data is stored
    :param block_size: Should be in range of [0,512], viable choices are 64,
    128, 256, 512
    :return: The dataset
    """
    dataset = TextDataset(tokenizer, file_path=file_path, mask_path=mask_path,
                          block_size=block_size)
    return dataset


def evaluate(model, tokenizer, eval_dataset, batch_size):
    """

    :param model: Newly trained Bert model
    :param tokenizer:Newly trained Bert tokenizer
    :param eval_dataset:
    :param batch_size: More flexible than training, the user can get away
    with picking a higher batch_size
    :return: The perplexity of the dataset
    """
    eval_sampler = SequentialSampler(eval_dataset)  # Same order samplinng
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    positions_to_mask = eval_dataset.masked_positions

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    # Evaluation loop
    i = 0
    for batch in tqdm_notebook(eval_dataloader, desc='Evaluating'):
        inputs, labels = custom_mask_tokens(batch, tokenizer, positions_to_mask[i])
        i += 1
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    result = {
        'perplexity': perplexity,
        'eval_loss': eval_loss
    }

    return result


word_prediction_args = {
    "batch_size": 1,
    "epochs": 1,
    "lr": 5e-5,
    "adam_epsilon": 1e-8

}


class FinetuneMlm():
    """

    :param train_path: Path to the training file, expected to be a .txt or
    similar
    :param test_path: Path to the testing file, expected to be a .txt or
    similar

    Default parameters for effortless finetuning
    batch size = 1
    Number of epochs  = 1
    Learning rate = 5e-5
    Adam epsilon = 1e-8
    model_name = Must be in default transformer name. ie: 'bert-base-uncased'

    """

    def __init__(self, mlm, args, tokenizer, logger):
        self.mlm = mlm
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger

    def train(self, train_path, eval_path, train_masked_path, eval_masked_path, output_dir):
        self.mlm.resize_token_embeddings(len(self.tokenizer))
        # Start Train
        self.mlm.cuda()
        train_dataset = create_dataset(
            self.tokenizer, file_path=train_path, mask_path=train_masked_path)
        eval_dataset = create_dataset(
            self.tokenizer, file_path=eval_path, mask_path=eval_masked_path)

        self.mlm, self.tokenizer = train(self.mlm, self.tokenizer,
                                         train_dataset,
                                         eval_dataset,
                                         batch_size=self.args["batch_size"],
                                         epochs=self.args["epochs"],
                                         lr=self.args["lr"],
                                         adam_epsilon=self.args[
                                             "adam_epsilon"],
                                         output_dir=output_dir)

        del train_dataset
        del eval_dataset
        self.mlm.cpu()
        return self.mlm, self.tokenizer

    def evaluate(self, test_path, masked_position_path, batch_size):
        self.mlm.cuda()
        test_dataset = create_dataset(self.tokenizer, file_path=test_path, mask_path=masked_position_path)
        result = evaluate(self.mlm, self.tokenizer, test_dataset,
                          batch_size=batch_size)
        del test_dataset
        self.mlm.cpu()
        return result
