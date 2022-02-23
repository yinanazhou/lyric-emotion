import os
import gc
import json
import re
import string
import torch
import numpy as np
import logging
from EarlyStoppingPytorch.pytorchtools import EarlyStopping
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel, AdamW, BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score
import argparse
import wandb
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def logging_storage(logfile_path):
    logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
    logging.info(torch.__version__)
    logging.info(DEVICE)


def noiseRemoval(text):
    # print(text)

    # remove numbers
    text_no_number = re.sub(r'\d+', '', text)
    # print(text_no_number)

    # remove punctuation
    text_no_punc = text_no_number.translate(str.maketrans("", "", string.punctuation))
    # print(text_no_punc)

    # remove whitespaces
    text_no_whitespace = text_no_punc.strip()
    # print(text_no_whitespace)

    return text_no_whitespace


def train(i, t_dataloader, loss_new):
    model.train()
    total_loss = 0.0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    train_len = 0  # training lyric number
    train_steps = 0

    for step, (b_input_ids, b_input_mask, b_labels) in enumerate(t_dataloader):
        b_input_ids = b_input_ids.to(DEVICE)
        b_input_mask = b_input_mask.to(DEVICE)
        b_labels = b_labels.to(DEVICE)
        if b_labels.size(0) <= 1:
            continue

        optimizer.zero_grad()

        torch.cuda.empty_cache()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        pred = outputs[1].detach().cpu().numpy()
        pred_flat = np.argmax(pred, axis=1).flatten()
        loss = outputs[0]
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        labels_flat = b_labels.flatten().cpu().detach().numpy()
        total_actual_label = np.concatenate((total_actual_label, labels_flat))
        total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

        train_len += b_input_ids.size(0)
        total_loss += loss.item()
        train_steps += 1

    f1 = f1_score(total_actual_label, total_predicted_label, average="macro")

    # wandb log
    wandb.log({"t_loss": total_loss / train_steps, "t_F1": f1 * 100.})

    return loss_new, f1 * 100.


def eva(v_dataloader):
    model.eval()
    val_len = 0
    total_loss = 0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    val_steps = 0

    with torch.no_grad():
        for step, (b_input_ids, b_input_mask, b_labels) in enumerate(v_dataloader):
            b_input_ids = b_input_ids.to(DEVICE)
            b_input_mask = b_input_mask.to(DEVICE)
            b_labels = b_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = outputs[1].detach().cpu().numpy()
            pred_flat = np.argmax(pred, axis=1).flatten()

            labels_flat = b_labels.flatten().cpu().detach().numpy()
            total_actual_label = np.concatenate((total_actual_label, labels_flat))
            total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

            val_len += b_input_ids.size(0)
            total_loss += float(outputs[0].sum())
            val_steps += 1

        val_f1 = f1_score(total_actual_label, total_predicted_label, average="macro")

        # wandb log
        wandb.log({"e_loss": total_loss / val_steps, "e_F1": val_f1 * 100.})

    return total_loss / val_steps, val_f1 * 100.


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
parser.add_argument('--lc', help='Lowercase Conversion', default=False, type=bool)
parser.add_argument('--nr', help='Noise Removal', default=False, type=bool)
parser.add_argument('--stop', help='Stop Words Removal', default=False, type=bool)
parser.add_argument('--stem', help='Stemming', default=False, type=bool)
parser.add_argument('--lemma', help='Lemmatization', default=False, type=bool)

args = parser.parse_args()

lr = 10 ** (-args.lr)
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
early_stop = 10 ** (-args.es)
lc = args.lc
nr = args.nr
stop = args.stop
stem = args.stem
lemma = args.lemma
# test_size = args.ts
model_str = 'xl'
num_labels = 4
denom = args.adaptive

# set path
trg_path = "LastFM_cleaned_train.json"
ending_path = ('%s_%d_bs_%d_lc_%s_nr_%s_stem_%s_lemma_%s' %(model_str, MAX_LEN, batch_size, lc, nr, stem, lemma))
save_model_path = "models/" + ending_path + '.pt'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
if not os.path.exists("logs/"):
    os.mkdir("logs/")
logfile_path = "logs/" + ending_path
logging_storage(logfile_path)
# result_path = "result_json/" + ending_path
if not os.path.exists("result_json/"):
    os.makedirs("result_json/")
logging.info("Using %d GPUs!" % (torch.cuda.device_count()))

# fetch data
with open(trg_path) as f:
    song_info = json.load(f)
lyrics = song_info["Lyric"]
labels = song_info["Mood"]
labels = np.array(labels)

# text preprocessing
logging.info("lc: %s, nr: %s, stop: %s, stem: %s, lemma: %s" % (lc, nr, stop, stem, lemma))
if nr:
    lyrics = [noiseRemoval(lyric) for lyric in lyrics]

if stop or stem or lemma:

    if stop:
        stop_words = set(stopwords.words('english'))
        for i in range(len(lyrics)):
            lyrics[i] = [word for word in lyrics[i] if word not in stop_words]

    if stem:
        stemmer = PorterStemmer()
        for i in range(len(lyrics)):
            lyrics[i] = [stemmer.stem(word) for word in lyrics[i]]

    if lemma:
        lemmatizer = WordNetLemmatizer()
        for i in range(len(lyrics)):
            lyrics[i] = [lemmatizer.lemmatize(word) for word in lyrics[i]]

    for i in range(len(lyrics)):
        lyrics[i] = ' '.join(lyrics[i])

# tokenize
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=lc)
tokenized_texts = [tokenizer.tokenize(lyric) for lyric in lyrics]

# convert tokens to index number in the XLNet vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# Pad input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# convert to tensor
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)
labels = torch.sub(labels, 1)

nSplits = 2
nRepeats = 1
repeaded_kfold = RepeatedStratifiedKFold(n_splits=nSplits, n_repeats=nRepeats, random_state=SEED)

results = []
result_json = {}
for fold, (train_idx, test_idx) in enumerate(repeaded_kfold.split(input_ids, labels)):
    logging.info('-------------------------------------------------')
    logging.info('Repeat: %d, Fold: %d' % (fold//nSplits + 1, (fold)%nSplits + 1))

    k_result = {'t_F1': [], 'e_F1': [], 'epoch': []}

    # wandb init
    wandb_pj = ending_path + '_repeat_' + str(fold//nSplits + 1) + '_fold_' + str((fold)%nSplits + 1)
    wandb.init(project=wandb_pj, entity="yinanazhou")

    # Dataset.select(indices=train_idx)
    train_inputs, test_inputs = input_ids[train_idx], input_ids[test_idx]
    train_masks, test_masks = attention_masks[train_idx], attention_masks[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # model
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)



    wandb.config = {
        "model": model_str,
        "learning_rate": lr,
        "lowercase_conversion": lc,
        "noise_removal": nr,
        "stop": stop,
        "stemming": stem,
        "lemmatization": lemma,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "max_len": MAX_LEN,
        "early_stop_criteria": early_stop
    }

    early_stopping = EarlyStopping(patience=3, verbose=True, path=save_model_path)

    loss_default = 5.0
    train_f1 = 0.0
    test_f1 = 0.0

    for i in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        loss_default, train_f1 = train(i, train_dataloader, loss_default)
        test_loss, test_f1 = eva(test_dataloader)

        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            logging.info("Early stopping at epoch %i" % i)
            break

    wandb.finish()
    logging.info("t_F1 = %5.3f, e_f1 = %5.3f" % (train_f1, test_f1))
    k_result['t_F1'].append(train_f1)
    k_result['e_F1'].append(test_f1)
    k_result['epoch'].append(i + 1)

    results.append(test_f1 * 100)
    result_json[str(fold + 1)] = []
    result_json[str(fold + 1)].append(k_result)

logging.info("AVERAGE F1: %5.3f", sum(results) / len(results))
result_json['average f1'] = []
result_json['average f1'].append(sum(results) / len(results))
result_path = "result_json/" + ending_path + '.json'
with open(result_path, 'w') as f:
    json.dump(result_json, f, indent=4)


