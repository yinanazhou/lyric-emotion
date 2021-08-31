from sklearn.preprocessing import LabelEncoder
import json

trg_path = "moody_test.json"

with open(trg_path) as f:
    song_info = json.load(f)
labels = song_info["Mood"][0]


#use encoder and transform
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(labels)

print(encoded_values)