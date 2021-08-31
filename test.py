import os
import json
import re
import string
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel, AdamW
from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from keras.preprocessing.sequence import pad_sequences
import argparse


#  preprocessing
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


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='Learning Rate', default=2e-5, type=float)
parser.add_argument('--epochs', help='Number of Epochs', default=20, type=int)
parser.add_argument('--ml', help='Max Len of Sequence', default=1024, type=int)
parser.add_argument('--bs', help='Batch Size', default=8, type=int)
parser.add_argument('--ts', help='Test Size', default=0.2, type=float)
parser.add_argument('--adaptive', help='Adaptive LR', default='20', type=float)

args = parser.parse_args()

lr = args.lr
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
test_size = args.ts
model = 'xlnet'
num_labels = 4
denom = args.adaptive

# set path
trg_path = "moody_test.json"
ending_path = ('%s_%d_bs_%d_adamw_data_%d_lr_%s_%d' %(model, MAX_LEN, batch_size,(1 - test_size)*100, str(lr).replace("-",""),denom))
save_model_path = "/models/" + ending_path
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

# fetch data
with open(trg_path) as f:
    song_info = json.load(f)
lyrics = song_info["Lyric"][0]
labels = song_info["Mood"][0]

# convert categorical labels to numerical labels
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(labels)

# tokenize
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(lyric) for lyric in lyrics]
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# for lyric in lyrics:
#     lyric = preprocess(lyric, tokenizer)
#     print(lyric)

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

