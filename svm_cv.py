import os
import gc
import json
import re
import string
import torch
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score, accuracy_score
import argparse
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
# import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')


def logging_storage(logfile_path):
    logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
    logging.info(torch.__version__)
    logging.info(DEVICE)


def preprocess(text):
    # print(text)

    # convert to lowercase
    text_lower = text.lower()
    # print(text_lower)

    # remove numbers
    text_no_number = re.sub(r'\d+', '', text_lower)
    # print(text_no_number)

    # remove punctuation
    text_no_punc = text_no_number.translate(str.maketrans("", "", string.punctuation))
    # print(text_no_punc)

    # remove whitespaces
    text_no_whitespace = text_no_punc.strip()
    # print(text_no_whitespace)

    return text_no_whitespace


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    labels_flat = labels_flat.cpu().detach().numpy()
    return np.sum(pred_flat == labels_flat), pred_flat


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='Learning Rate', default=2e-5, type=float)
parser.add_argument('--epochs', help='Number of Epochs', default=20, type=int)
parser.add_argument('--ml', help='Max Len of Sequence', default=1024, type=int)
parser.add_argument('--bs', help='Batch Size', default=8, type=int)
# parser.add_argument('--ts', help='Test Size (0-1)', default=0.2, type=float)
parser.add_argument('--adaptive', help='Adaptive LR', default='20', type=float)

args = parser.parse_args()

lr = args.lr
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
# test_size = args.ts
model_str = 'svm_linear'
num_labels = 4
denom = args.adaptive
remove_stop_words = False
stemming = False
lemma = False

# set path
trg_path = "moody_lyrics.json"
ending_path = ('%s_%d_bs_%d' %(model_str, MAX_LEN, batch_size))
save_model_path = "models/" + ending_path
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
if not os.path.exists("logs/"):
    os.mkdir("logs/")
logfile_path = "logs/" + ending_path
logging_storage(logfile_path)
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

# lyric preprocessing
lyrics = [preprocess(lyric) for lyric in lyrics]

if remove_stop_words or stemming or lemma:

    lyrics = [word_tokenize(lyric) for lyric in lyrics]

    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        for i in range(len(lyrics)):

            lyrics[i] = [word for word in lyrics[i] if word not in stop_words]

    if stemming:
        stemmer = PorterStemmer()
        for i in range(len(lyrics)):
            lyrics[i] = [stemmer.stem(word) for word in lyrics[i]]

    if lemma:
        lemmatizer = WordNetLemmatizer()
        for i in range(len(lyrics)):
            lyrics[i] = [lemmatizer.lemmatize(word) for word in lyrics[i]]

    for i in range(len(lyrics)):
        lyrics[i] = ' '.join(lyrics[i])

# train-validation test split
# 6 2 2
train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(lyrics, encoded_labels,
                                                                                random_state=SEED, test_size=0.2)

# Convert to vector
tfidf_vect = TfidfVectorizer(max_features=MAX_LEN, lowercase=False)
tfidf_vect.fit(lyrics)

train_val_inputs = tfidf_vect.transform(train_val_inputs)
test_inputs = tfidf_vect.transform(test_inputs)

k_folds = 5
results = []
result_json = {}
kfold = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_inputs)):
    logging.info('-------------------------------------------------')
    logging.info('%d FOLD', fold+1)

    # write results of each fold into a dic
    # k_result = {'t_loss': [], 't_accuracy': [], 't_precision': [], 't_recall': [], 't_F1_measure': [],
    #             'e_loss': [], 'e_accuracy': [], 'e_precision': [], 'e_recall': [], 'e_F1_measure': []}

    k_result = {'accuracy': [], 'precision': [], 'recall': [], 'F1_measure': []}

    # Dataset.select(indices=train_idx)
    train_inputs, val_inputs = train_val_inputs[train_idx], train_val_inputs[val_idx]
    # train_masks, val_masks = train_val_masks[train_idx], train_val_masks[val_idx]
    train_labels, val_labels = train_val_labels[train_idx], train_val_labels[val_idx]

    # define model
    # xlnet
    # model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)
    # BERT
    # model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
    # model = nn.DataParallel(model)
    model = svm.SVC(kernel="linear")
    # model.to(DEVICE)

    model.fit(train_inputs, train_labels)

    pred = model.predict(val_inputs)

    k_acc = accuracy_score(val_labels, pred)
    precision, recall, f1_measure, _ = precision_recall_fscore_support(val_labels, pred, average='macro')
    logging.info("Acc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" % (
            k_acc * 100, precision * 100., recall * 100., f1_measure * 100.))

    k_result['accuracy'].append(k_acc * 100)
    k_result['precision'].append(precision * 100.)
    k_result['recall'].append(recall * 100.)
    k_result['F1_measure'].append(f1_measure * 100.)

    results.append(k_acc * 100)
    result_json[str(fold+1)] = []
    result_json[str(fold+1)].append(k_result)

logging.info("AVERAGE ACCURACY: %5.3f", sum(results) / len(results))
result_json['average accuracy'] = []
result_json['average accuracy'].append(sum(results) / len(results))
result_path = "result_json/" + ending_path + '.json'
with open(result_path, 'w') as f:
    json.dump(result_json, f, indent=4)
