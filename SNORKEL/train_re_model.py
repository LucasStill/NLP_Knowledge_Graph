import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, InputLayer, LSTM
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from snorkel.utils import probs_to_preds
import ast

# Pad or truncate sentences
def pad_or_truncate(l, max_length=128):
    return l[:max_length] + [0] * (max_length - len(l))

# LOAD DATA
# Load created train set
df = pd.read_csv("annotated_all.csv", sep="|")

# Set correct type for list
df['Word'] = df['Word'].apply(ast.literal_eval)
df['Word_idx'] = df['Word_idx'].apply(ast.literal_eval)
df['Tag'] = df['Tag'].apply(ast.literal_eval)

# Remove labels if they don't occur enough (800 times, for undersampling later on)
df = df[df.Relation.isin(list(df.Relation.value_counts().loc[lambda x: x > 800].index))].reset_index(drop=True)

# Create padded / truncated input
X = np.array(list(map(pad_or_truncate, df.Word_idx)))

# Create one hot encoding for relations
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df.Relation)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

# CREATE TRAIN, TEST AND VALIDATION SET (80, 10, 10, RESPECTIVELY)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8, random_state=12345)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size=.5, random_state=12345)

# BALANCING THE DATA
under_sampler = RandomUnderSampler(random_state=42)
X_train, Y_train = under_sampler.fit_resample(X_train, Y_train)

# CREATE MODEL
# Parameters
embedding_size = 256
rnn_state_size = 128
dense_layer_one = 128
dense_layer_two = 64

batch_size = 32
epochs = 15

# Design model
model = Sequential()

# Create input layer
model.add(InputLayer((None,)))
model.add(Embedding(X.max()+1, embedding_size))
model.add(Bidirectional(LSTM(rnn_state_size, activation=tf.nn.relu)))

# Use fully connected layers to output
model.add(Dense(dense_layer_one, activation=tf.nn.relu))
model.add(Dense(dense_layer_two, activation=tf.nn.relu))
model.add(Dense(Y.shape[1], activation=tf.nn.softmax))

# Assign optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAIN MODEL
history = model.fit(X_train, Y_train, batch_size, epochs=epochs, verbose=1, validation_data=(X_val, Y_val))

# SAVE MODEL
model.save("benchmark_re")

# MEASURE PERFORMANCE TESTSET
probs_test = model.predict(X_test)
Y_Pred = probs_to_preds(probs_test)

print("Test Accuracy: {:.4f}".format(accuracy_score([np.where(r == 1)[0][0] for r in Y_test], Y_Pred)))
print("Test Precision: {:.4f}".format(precision_score([np.where(r == 1)[0][0] for r in Y_test], Y_Pred, average='weighted')))
print("Test Recall: {:.4f}".format(recall_score([np.where(r == 1)[0][0] for r in Y_test], Y_Pred, average='weighted')))
print("Test F1-score: {:.4f}".format(f1_score([np.where(r == 1)[0][0] for r in Y_test], Y_Pred, average='weighted')))

# PLOT RESULTS
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
