# DeepPass

Dockerized application that analyzes documents for password candidates.

To run: `docker-compose up`

This will expose http://localhost:5000 where documents can be uploaded.

The API can manually be used at `http://localhost:5000/api/passwords` :

```
C:\Users\harmj0y\Documents\GitHub\DeepPass>curl -F "file=@test_doc.docx" http://localhost:5000/api/passwords
[{"file_name": "test_doc.docx", "model_password_candidates": [{"left_context": ["for", "the", "production", "server", "is:"], "password": "P@ssword123!", "right_context": ["Please", "dont", "tell", "anyone", "on"]}, {"left_context": ["that", "the", "other", "password", "is"], "password": "LiverPool1", "right_context": [".", "This", "is", "our", "backup."]}], "regex_password_candidates": [{"left_context": ["for", "the", "production", "server", "is:"], "password": "P@ssword123!", "right_context": ["Please", "dont", "tell", "anyone", "on"]}], "custom_regex_matches": null}]
```

[Apache Tika](https://hub.docker.com/r/apache/tika) is used to extract data from [various document formats](https://tika.apache.org/0.9/formats.html). [Tensorflow Serving](https://hub.docker.com/r/tensorflow/serving) is used for serving the model.

The neural network is Bidirectional LSTM:

```
embedding_dimension = 20
dropout = 0.5
cells = 200

model = Sequential()
model.add(Embedding(total_chars, embedding_dimension, input_length=32, mask_zero=True))
model.add(Bidirectional(LSTM(cells)))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
```

It was trained on 2,000,000 passwords randomly selected from [this leaked password list](https://crackstation.net/files/crackstation-human-only.txt.gz) and 2,000,000 extracted terms from various Google dorked documents. The stats for the .1 test set are:

```
------------------
loss       :  0.04804224148392677
tn         :  199446.0
fp         :  731.0
fn         :  3281.0
tp         :  196542.0
------------------
accuracy   :  0.9899700284004211
precision  :  0.9962944984436035
recall     :  0.983580470085144
------------------
F1 score.     :  0.9898966618590025
------------------
```

The training notebook for the model is in  `./notebooks/password_model_bilstm.ipynb`