import os
import gc
import json
import re
import string
import torch
import numpy as np
import logging
from EarlyStoppingPytorch.pytorchtools import EarlyStopping
from transformers import AdamW, BertTokenizer, BertForSequenceClassification
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
from sklearn.model_selection import StratifiedShuffleSplit

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
parser.add_argument('--es', help='Early Stopping Loss Patience ', default=7, type=float)
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
es = args.es
lc = args.lc
nr = args.nr
stop = args.stop
stem = args.stem
lemma = args.lemma
# test_size = args.ts
model_str = 'bertTest'
num_labels = 4
denom = args.adaptive

# set path
train_path = "LastFM_full_cleaned.json"
test_path = "AllMusic_cleaned.json"
ending_path = ('%s_%d_bs_%d_lr_%s_es_%i_lc_%s_nr_%s_sr_%s_stem_%s_lemma_%s' %(model_str, MAX_LEN, batch_size,str(lr).replace("-",""), es, lc, nr, stop, stem, lemma))
model_path = ending_path + '.ckpt'
if not os.path.exists('test_models/'):
    os.makedirs('test_models/')
save_model_path = os.path.join('test_models', model_path)
if not os.path.exists("test_logs/"):
    os.mkdir("test_logs/")
logfile_path = "test_logs/" + ending_path
logging_storage(logfile_path)
# result_path = "result_json/" + ending_path
if not os.path.exists("test_result_json/"):
    os.makedirs("test_result_json/")
logging.info("Using %d GPUs!" % (torch.cuda.device_count()))

# fetch training data
with open(train_path) as f:
    trainSet = json.load(f)
trainLyrics = trainSet["Lyric"]
trainLabels = trainSet["Mood"]
trainLabels = np.array(trainLabels)
# fetch test data
with open(test_path) as f:
    testSet = json.load(f)
testLyrics = testSet["Lyric"]
testLabels = testSet["Mood"]
testLabels = np.array(testLabels)

# text preprocessing
logging.info("es: %i, lc: %s, nr: %s, stop: %s, stem: %s, lemma: %s" % (es, lc, nr, stop, stem, lemma))
# noise reduction
if nr:
    # train
    trainLyrics = [noiseRemoval(trainLyric) for trainLyric in trainLyrics]
    # test
    testLyrics = [noiseRemoval(testLyric) for testLyric in testLyrics]

# tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=lc)
tokenizedTrain = [tokenizer.tokenize(lyric) for lyric in trainLyrics]
# tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=lc)
tokenizedTest = [tokenizer.tokenize(lyric) for lyric in testLyrics]

if stop or stem or lemma:
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # train
    for i in range(len(tokenizedTrain)):
        if stop: tokenizedTrain[i] = [word for word in tokenizedTrain[i] if word not in stop_words]
        if stem: tokenizedTrain[i] = [stemmer.stem(word, to_lowercase=lc) for word in tokenizedTrain[i]]
        if lemma: tokenizedTrain[i] = [lemmatizer.lemmatize(word) for word in tokenizedTrain[i]]

    # test
    for i in range(len(tokenizedTest)):
        if stop: tokenizedTest[i] = [word for word in tokenizedTest[i] if word not in stop_words]
        if stem: tokenizedTest[i] = [stemmer.stem(word, to_lowercase=lc) for word in tokenizedTest[i]]
        if lemma: tokenizedTest[i] = [lemmatizer.lemmatize(word) for word in tokenizedTest[i]]

# train
# convert tokens to index number in the Bert vocabulary
inputIdTrain = [tokenizer.convert_tokens_to_ids(x) for x in tokenizedTrain]
# Pad input tokens
inputIdTrain = pad_sequences(inputIdTrain, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# test
# convert tokens to index number in the Bert vocabulary
inputIdTest = [tokenizer.convert_tokens_to_ids(x) for x in tokenizedTest]
# Pad input tokens
inputIdTest = pad_sequences(inputIdTest, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attentionMaskTrain = []
attentionMaskTest = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in inputIdTrain:
    seq_mask = [float(i > 0) for i in seq]
    attentionMaskTrain.append(seq_mask)
for seq in inputIdTest:
    seq_mask = [float(i > 0) for i in seq]
    attentionMaskTest.append(seq_mask)

# convert to tensor
inputIdTrain = torch.tensor(inputIdTrain)
attentionMaskTrain = torch.tensor(attentionMaskTrain)
trainLabels = torch.tensor(trainLabels)
trainLabels = torch.sub(trainLabels, 1)
# test
inputIdTest = torch.tensor(inputIdTest)
attentionMaskTest = torch.tensor(attentionMaskTest)
testLabels = torch.tensor(testLabels)
testLabels = torch.sub(testLabels, 1)

# results = []
# result_json = {}
# for fold, (train_idx, test_idx) in enumerate(repeaded_kfold.split(input_ids, labels)):
logging.info('-------------------------------------------------')
# logging.info('Repeat: %d, Fold: %d' % (fold//nSplits + 1, (fold)%nSplits + 1))

k_result = {'t_F1': [], 'e_F1': [], 'epoch': []}

# wandb init
wandb_pj = ending_path
wandb.init(project=wandb_pj, entity="yinanazhou")

# Dataset.select(indices=train_idx)
# train_inputs, test_inputs = input_ids[train_idx], input_ids[test_idx]
# train_masks, test_masks = attention_masks[train_idx], attention_masks[test_idx]
# train_labels, test_labels = labels[train_idx], labels[test_idx]

# split train and validation set
train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
for train_index, test_index in train_val_split.split(inputIdTrain, trainLabels):
    train_inputs, val_inputs = inputIdTrain[train_index], inputIdTrain[test_index]
    train_masks, val_masks = attentionMaskTrain[train_index], attentionMaskTrain[test_index]
    train_labels, val_labels = trainLabels[train_index], trainLabels[test_index]

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(inputIdTest, attentionMaskTest, testLabels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# model
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
model.to(DEVICE)

for name, param in model.named_parameters():
    # if 'transformer' in name and '11' not in name:  # classifier layer
    if 'bert' in name and 'pooler' not in name and '11' not in name:
        param.requires_grad = False

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
    "early_stop_criteria": es
}

early_stopping = EarlyStopping(patience=es, verbose=True, path=save_model_path)

loss_default = 5.0
train_f1 = 0.0
val_f1 = 0.0
test_f1 = 0.0

for i in range(num_epochs):
    gc.collect()
    torch.cuda.empty_cache()
    loss_default, train_f1 = train(i, train_dataloader, loss_default)
    val_loss, val_f1 = eva(val_dataloader)

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        logging.info("Early stopping at epoch %i" % i)
        break

logging.info('-------------------------------------------------')
logging.info("Start testing")
test_loss, test_f1 = eva(test_dataloader)


wandb.finish()
logging.info("t_F1 = %5.3f, v_f1 = %5.3f, test_f1 = %5.3f" % (train_f1, val_f1, test_f1))



