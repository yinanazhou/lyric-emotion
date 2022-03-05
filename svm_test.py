import os
import gc
import json
import re
import string
import torch
import numpy as np
import logging
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
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
parser.add_argument('--lc', help='Lowercase Conversion', default=False, type=bool)
parser.add_argument('--nr', help='Noise Removal', default=False, type=bool)
parser.add_argument('--stop', help='Stop Words Removal', default=False, type=bool)
parser.add_argument('--stem', help='Stemming', default=False, type=bool)
parser.add_argument('--lemma', help='Lemmatization', default=False, type=bool)

args = parser.parse_args()

lr = args.lr
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
# test_size = args.ts
model_str = 'All_test_svm_rbf_lc_nr_sr_stem'
num_labels = 4
denom = args.adaptive
remove_stop_words = args.stop
stemming = args.stem
lemma = args.lemma
lc = args.lc
nr = args.nr

# set path
train_path = "LastFM_full_cleaned.json"
test_path = "AllMusic_cleaned.json"
ending_path = ('%s_ml_%d' %(model_str, MAX_LEN))
if not os.path.exists("LastFM_logs_F1/"):
    os.mkdir("LastFM_logs_F1/")
logfile_path = "LastFM_logs_F1/" + ending_path
logging_storage(logfile_path)
# result_path = "result_json/" + ending_path
# if not os.path.exists("MER_result_F1_json/"):
#     os.makedirs("MER_result_F1_json/")


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

# noise reduction
if nr:
    # train
    trainLyrics = [noise_removal(trainLyric) for trainLyric in trainLyrics]
    # test
    testLyrics = [noise_removal(testLyric) for testLyric in testLyrics]

if remove_stop_words or stemming or lemma:

    trainLyrics = [word_tokenize(trainLyric) for trainLyric in trainLyrics]
    testLyrics = [word_tokenize(testLyric) for testLyric in testLyrics]

    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        for i in range(len(trainLyrics)):
            trainLyrics[i] = [word for word in trainLyrics[i] if word not in stop_words]
        for i in range(len(testLyrics)):
            testLyrics[i] = [word for word in testLyrics[i] if word not in stop_words]

    if stemming:
        stemmer = PorterStemmer()
        for i in range(len(trainLyrics)):
            trainLyrics[i] = [stemmer.stem(word, to_lowercase=lc) for word in trainLyrics[i]]
        for i in range(len(testLyrics)):
            testLyrics[i] = [stemmer.stem(word, to_lowercase=lc) for word in testLyrics[i]]

    if lemma:
        lemmatizer = WordNetLemmatizer()
        for i in range(len(trainLyrics)):
            trainLyrics[i] = [lemmatizer.lemmatize(word) for word in trainLyrics[i]]
        for i in range(len(testLyrics)):
            testLyrics[i] = [lemmatizer.lemmatize(word) for word in testLyrics[i]]


    for i in range(len(trainLyrics)):
        trainLyrics[i] = ' '.join(trainLyrics[i])
    for i in range(len(testLyrics)):
        testLyrics[i] = ' '.join(testLyrics[i])

# Convert to vector
tfidf_vect = TfidfVectorizer(max_features=MAX_LEN, lowercase=lc)
trainLyrics = tfidf_vect.fit_transform(trainLyrics).toarray()
testLyrics = tfidf_vect.transform(testLyrics).toarray()

# results = []
# result_json = {}

logging.info('-------------------------------------------------')
logging.info('Test')
# define model
model = svm.SVC(kernel="rbf")
model.fit(trainLyrics, trainLabels)
pred = model.predict(testLyrics)
f1 = f1_score(testLabels, pred, average='macro')
logging.info("F1 = %5.3f" % (f1 * 100))

