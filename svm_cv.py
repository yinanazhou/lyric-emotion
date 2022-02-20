import os
import gc
import json
import re
import string
import torch
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score, accuracy_score
import argparse
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')


def logging_storage(logfile_path):
    logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
    logging.info(torch.__version__)
    logging.info(DEVICE)


def noise_removal(text):
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
model_str = 'svm_rbf_lc_nr_lemma'
num_labels = 4
denom = args.adaptive
remove_stop_words = False
stemming = False
lemma = True

# set path
trg_path = "MER_dataset.json"
ending_path = ('%s_ml_%d' %(model_str, MAX_LEN))
if not os.path.exists("MER_logs/"):
    os.mkdir("MER_logs/")
logfile_path = "MER_logs/" + ending_path
logging_storage(logfile_path)
# result_path = "result_json/" + ending_path
if not os.path.exists("MER_result_json/"):
    os.makedirs("MER_result_json/")


# fetch data
with open(trg_path) as f:
    song_info = json.load(f)
lyrics = song_info["Lyric"]
labels = song_info["Mood"]
labels = np.array(labels)


# noise reduction
lyrics = [noise_removal(lyric) for lyric in lyrics]

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
# Convert to vector
tfidf_vect = TfidfVectorizer(max_features=MAX_LEN, lowercase=True)
lyrics = tfidf_vect.fit_transform(lyrics).toarray()

nSplits = 2
nRepeats = 5
repeaded_kfold = RepeatedStratifiedKFold(n_splits=nSplits, n_repeats=nRepeats, random_state=SEED)

results = []
result_json = {}
for fold, (train_idx, test_idx) in enumerate(repeaded_kfold.split(lyrics, labels)):
    logging.info('-------------------------------------------------')
    logging.info('Repeat: %d, Fold: %d' % (fold//nSplits + 1, (fold)%nSplits + 1))

    k_result = {'accuracy': []}

    # Dataset.select(indices=train_idx)
    train_inputs, test_inputs = lyrics[train_idx], lyrics[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    # print("TRAIN:", train_idx, "TEST:", test_idx)

    # define model
    model = svm.SVC(kernel="rbf")
    # model.to(DEVICE)

    model.fit(train_inputs, train_labels)

    pred = model.predict(test_inputs)

    k_acc = accuracy_score(test_labels, pred)
    logging.info("Acc= %5.3f" % (k_acc * 100))

    k_result['accuracy'].append(k_acc * 100)

    results.append(k_acc * 100)
    result_json[str(fold+1)] = []
    result_json[str(fold+1)].append(k_result)

logging.info("AVERAGE ACCURACY: %5.3f", sum(results) / len(results))
result_json['average accuracy'] = []
result_json['average accuracy'].append(sum(results) / len(results))
result_path = "MER_result_json/" + ending_path + '.json'
with open(result_path, 'w') as f:
    json.dump(result_json, f, indent=4)
