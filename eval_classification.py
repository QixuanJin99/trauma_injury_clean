import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle
import argparse
from pathlib import Path

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from vae_arch import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_path', type=str, default = "")
parser.add_argument('--output_dir', type=str, default = "")
parser.add_argument('--model_prefix', type=str, default = "")
args = parser.parse_args()
print(args)

embedding_test = pd.read_pickle(args.embedding_path) 
print(embedding_test.shape)

embedding_train = pd.read_pickle(args.embedding_path.replace("test", "train"))
print(embedding_train.shape)

test_index = np.loadtxt("final/new_test_index.csv").astype(int)
train_index = np.loadtxt("final/new_train_index.csv").astype(int)

df_embed_test = pd.DataFrame(embedding_test, index = test_index)
df_embed_test.index.name = 'INC_KEY'

df_embed_train = pd.DataFrame(embedding_train, index = train_index)
df_embed_train.index.name = 'INC_KEY'

demo = pd.read_pickle("final/demo.pkl")
gcs = pd.read_pickle("aux_signal/gcs_signal.pkl")
mech = pd.read_pickle("aux_signal/mech_coarse_signal.pkl")
risk = pd.read_pickle("aux_signal/high_risk_icd.pkl")

# Don't include race or gender
# demo = demo[['age']]
# Normalize age 
# min_age = demo['age'].min()
# max_age = demo['age'].max()
# demo['age'] = (demo['age'] - min_age)/(max_age - min_age)
# cond = pd.merge(demo, gcs[['mild', 'moderate', 'severe']], 
#                 how = 'inner', left_index=True, right_index=True)
cond = gcs[['mild', 'moderate', 'severe']]
cond = pd.merge(cond, mech, how='inner',
                left_index=True, right_index=True)
cond = pd.merge(cond, risk, how='inner',
                left_index=True, right_index=True)

cond[cond > 1] = 1
cond_train = cond.loc[train_index]
cond_test = cond.loc[test_index]

X_train = tf.cast(df_embed_train.values, tf.float32)
X_test = tf.cast(df_embed_test.values, tf.float32)
y_train = tf.cast(cond_train.values, tf.float32)
y_test = tf.cast(cond_test.values, tf.float32)

args.num_classes = cond.shape[1]
n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]

model = keras.Sequential()
model.add(Dense(128, input_dim=n_inputs, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = "adam")
model.fit(X_train, y_train, epochs = 100, batch_size = 256, verbose = 2) 

yhat_prob = model.predict(X_test)
yhat = yhat_prob.round()

df = pd.DataFrame(index = cond.columns)
df['auc'] = roc_auc_score(y_test, yhat_prob, average = None)
df['f1'] = f1_score(y_test, yhat, average = None)
df['recall'] = recall_score(y_test, yhat, average=None)
df['precision'] = precision_score(y_test, yhat, average=None, zero_division = 0.)
df = df.round(3)

vectors = {"yhat": yhat, 
           "yhat_prob": yhat_prob}

out_dir = Path(f"{args.output_dir}aux/")
out_dir.mkdir(exist_ok = True, parents = True)

df.to_pickle(f"{args.output_dir}aux/{args.model_prefix}_auxclassification_metrics.pkl") 
with open(f"{args.output_dir}aux/{args.model_prefix}_auxclassification_vectors.pkl", "wb") as f: 
    pickle.dump(vectors, f)
print("Finished classification!")