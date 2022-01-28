import os
import gc
import json
import re
import string
import torch
import numpy as np
import logging
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel, AdamW, BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score, accuracy_score
import argparse
import wandb
from bert_model import bertModel
import warnings


warnings.filterwarnings('ignore')


def logging_storage(logfile_path):
    logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
    logging.info(torch.__version__)
    logging.info(DEVICE)


def preprocess(text, tokenizer):
    print(text)

    # convert to lowercase
    text_lower = text.lower()
    print(text_lower)

    # remove numbers
    text_no_number = re.sub(r'\d+', '', text_lower)
    print(text_no_number)

    # remove punctuation
    text_no_punc = text_no_number.translate(str.maketrans("", "", string.punctuation))
    print(text_no_punc)

    # remove whitespaces
    text_no_whitespace = text_no_punc.strip()
    print(text_no_whitespace)

    # tokenize
    text_tokenize = tokenizer.tokenize(text_no_whitespace)

    # remove stop words
    # stop_words = set(stopwords.words(‘english’))
    # result = [i for i in text_tokenize if not i in stop_words]

    return text_tokenize


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten().cpu().detach().numpy()
    return np.sum(pred_flat == labels_flat), pred_flat, labels_flat


def train(i, t_dataloader, loss_new):
    model.train()
    total_loss = 0.0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    crossEntropy = nn.CrossEntropyLoss()

    ## adaptive lr
    optimizer.param_groups[0]['lr'] *= (0.1) ** (1 / denom)

    for step, (b_input_ids, b_input_mask, b_labels) in enumerate(t_dataloader):
        b_input_ids = b_input_ids.to(DEVICE)
        b_input_mask = b_input_mask.to(DEVICE)
        b_labels = b_labels.to(DEVICE)
        if b_labels.size(0) <= 1:
            continue

        optimizer.zero_grad()

        torch.cuda.empty_cache()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        pred = np.argmax(outputs.detach().cpu().numpy(), axis=1).flatten()

        loss = crossEntropy(outputs, b_labels)
        loss.sum().backward()
        optimizer.step()

        total_actual_label = np.concatenate((total_actual_label, b_labels))
        total_predicted_label = np.concatenate((total_predicted_label, pred))

        total_loss += loss.item()

    precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label, total_predicted_label,
                                                                       average='macro')
    acc = accuracy_score(total_actual_label, total_predicted_label)

    epoch_len = len(total_actual_label)

    logging.info(
        "Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" % (
        epoch_len * 100.0 / train_inputs.size(0), i+1, step, total_loss / epoch_len, acc * 100.,
        precision * 100., recall * 100., f1_measure * 100.))
    k_result['t_loss'].append(total_loss / epoch_len)
    k_result['t_accuracy'].append(acc * 100.)
    k_result['t_precision'].append(precision * 100.)
    k_result['t_recall'].append(recall * 100.)
    k_result['t_F1_measure'].append(f1_measure * 100.)

    # wandb log
    wandb.log({"loss": total_loss / epoch_len, "accuracy": acc, "precision": precision * 100.,
               "recall": recall * 100., "F1_measure": f1_measure * 100.})

    # early stop
    flag = False

    loss_last = loss_new
    loss_new = total_loss / epoch_len

    if loss_last < early_stop and loss_new < early_stop:
        logging.info("Epoch: %d\tearly stopped at loss: %5.5f" % (i + 1, total_loss / epoch_len))

        flag = True
    return flag, loss_new


def eva(v_dataloader):
    model.eval()

    total_loss = 0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    crossEntropy = nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, (b_input_ids, b_input_mask, b_labels) in enumerate(v_dataloader):
            b_input_ids = b_input_ids.to(DEVICE)
            b_input_mask = b_input_mask.to(DEVICE)
            b_labels = b_labels.to(DEVICE)

            optimizer.zero_grad()

            torch.cuda.empty_cache()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = np.argmax(outputs.detach().cpu().numpy(), axis=1).flatten()
            loss = crossEntropy(outputs, b_labels)

            total_actual_label = np.concatenate((total_actual_label, b_labels))
            total_predicted_label = np.concatenate((total_predicted_label, pred))

            total_loss += loss.item()

        precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label, total_predicted_label,
                                                                           average='macro')
        acc = accuracy_score(total_actual_label, total_predicted_label)
        epoch_len = len(total_actual_label)
        logging.info(
            "Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" % (
            epoch_len * 100.0 / val_inputs.size(0), i+1, step, total_loss / epoch_len, acc * 100.0,
            precision * 100., recall * 100., f1_measure * 100.))

        # wandb log
        wandb.log({"e_loss": total_loss / epoch_len, "e_accuracy": acc * 100.0, "e_precision": precision * 100.,
                   "e_recall": recall * 100., "e_F1_measure": f1_measure * 100.})

        k_result['e_loss'].append(total_loss / epoch_len)
        k_result['e_accuracy'].append(acc * 100.0)
        k_result['e_precision'].append(precision * 100.)
        k_result['e_recall'].append(recall * 100.)
        k_result['e_F1_measure'].append(f1_measure * 100.)
    return acc * 100.0


# check gpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.device_count() > 1:

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='Learning Rate', default=5, type=float)
parser.add_argument('--epochs', help='Number of Epochs', default=20, type=int)
parser.add_argument('--ml', help='Max Len of Sequence', default=1024, type=int)
parser.add_argument('--bs', help='Batch Size', default=8, type=int)
parser.add_argument('--es', help='Early Stopping Loss Criteria ', default=2, type=float)
parser.add_argument('--adaptive', help='Adaptive LR', default='20', type=float)

args = parser.parse_args()

lr = 10 ** (-args.lr)
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
early_stop = 10 ** (-args.es)
# test_size = args.ts
# model_str = 'xlnet8'
model_str = 'test'
# num_labels = 4
denom = args.adaptive

# set path
trg_path = "moody_test.json"
ending_path = ('%s_%d_bs_%d_adamw_lr_%s_es_%s_%d' %(model_str, MAX_LEN, batch_size, str(lr).replace("-",""), str(early_stop).replace("-",""), denom))
save_model_path = "models/" + ending_path
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
if not os.path.exists("logs/"):
    os.mkdir("logs/")
logfile_path = "logs/" + ending_path
logging_storage(logfile_path)
logging.info("Using %d GPUs!", torch.cuda.device_count())
# result_path = "result_json/" + ending_path
if not os.path.exists("result_json/"):
    os.makedirs("result_json/")


# fetch data
with open(trg_path) as f:
    song_info = json.load(f)
lyrics = song_info["Lyric"][0]
labels = song_info["Mood"][0]

# convert categorical labels to numerical labels
# Angry = 0; Happy = 1; Relaxed = 2; Sad = 3
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

attention_masks = []
# input_ids = []
# tokenize
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(lyric) for lyric in lyrics]
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# train-validation test split
# 6 2 2
train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(input_ids, encoded_labels,
                                                                                random_state=SEED, test_size=0.2)
train_val_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                     random_state=SEED, test_size=0.2)

# Convert to Tensor
train_val_inputs = torch.tensor(train_val_inputs)
train_val_labels = torch.tensor(train_val_labels)
train_val_masks = torch.tensor(train_val_masks)

test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

k_folds = 5
results = []
result_json = {}
kfold = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_inputs)):
    logging.info('------------------------------------------------------')
    logging.info('%d FOLD', fold+1)

    # wandb init
    wandb_pj = ending_path + '_fold_'+ str(fold)
    wandb.init(project=wandb_pj, entity="yinanazhou")

    # write results of each fold into a dic
    k_result = {'t_loss': [], 't_accuracy': [], 't_precision': [], 't_recall': [], 't_F1_measure': [],
                'e_loss': [], 'e_accuracy': [], 'e_precision': [], 'e_recall': [], 'e_F1_measure': [],
                'epoch': []}

    # Dataset.select(indices=train_idx)
    train_inputs, val_inputs = train_val_inputs[train_idx], train_val_inputs[val_idx]
    train_masks, val_masks = train_val_masks[train_idx], train_val_masks[val_idx]
    train_labels, val_labels = train_val_labels[train_idx], train_val_labels[val_idx]

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # define model
    # xlnet
    # xlnet_transformer = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=8)
    # BERT
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=8)
    model = bertModel(model)
    model = nn.DataParallel(model)
    model.to(DEVICE)

    # define optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    es_flag = False
    loss_default = 5

    wandb.config = {
        "model": model_str,
        "learning_rate": lr,
        "adaptive_learning_rate": (0.1) ** (1 / denom),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "max_len": MAX_LEN,
        "early_stop_criteria": early_stop
    }

    for i in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        es_flag, loss_default = train(i, train_dataloader, loss_default)
        acc = eva(val_dataloader)

        if es_flag:
            break

    wandb.finish()

    k_result['epoch'].append(i+1)
    gc.collect()
    torch.cuda.empty_cache()
    results.append(acc)
    result_json[str(fold+1)] = []
    result_json[str(fold+1)].append(k_result)

logging.info("AVERAGE ACCURACY: %5.3f", sum(results) / len(results))
result_json['average accuracy'] = []
result_json['average accuracy'].append(sum(results) / len(results))
result_path = "result_json/" + ending_path + '.json'
with open(result_path, 'w') as f:
    json.dump(result_json, f, indent=4)
