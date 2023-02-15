import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import pickle
import itertools
import argparse
from pathlib import Path
import uuid

import tensorflow as tf
from tensorflow_probability import distributions as tfd
print(tf.__version__)

from vae_arch import Encoder, Decoder, Classifier
from vae_arch import BetaVAE_Classifier
from callbacks import CustomEarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--model_arch', type=str, default='betavae-classifier-v2',
                    help = 'model architecture')
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--layer_sizes', nargs='+', 
                    help = 'dimensions for 2 layer encoder and decoder')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--output_dir', type=str, default = "")
parser.add_argument('--nohead', action="store_true")

## ---------------------------------------------------------------------
parser.add_argument('--repeats', type=int, default=1, 
                    help= 'number of repeated randomized runs')
parser.add_argument('--beta', type=float, default=4., 
                    help = 'disentanglement hyperparam for BetaVAE')
parser.add_argument('--gamma', type=float, default=1., 
                    help = 'weight hyperparam for classifier in loss function')
parser.add_argument('--classifier_layer_size', type=int, default=128, 
                    help = 'layer size for simple classifier')
parser.add_argument('--cond_features', type=str, default = 'demo_gcs_mech_risk',  
                    help = 'options: demo, gcs, mech, risk')
parser.add_argument('--patience', type=int, default = 5, 
                    help = 'hyperparam for callback')

parser.add_argument('--unweighted_classifier_loss', action = 'store_true')
parser.add_argument('--demo_idx', nargs='+', default = [0, 1]) 
parser.add_argument('--gcs_idx', nargs='+', default = [1, 4]) 
parser.add_argument('--mech_idx', nargs='+', default = [4, 12]) 
parser.add_argument('--risk_idx', nargs='+', default = [12, 13]) 
parser.add_argument('--reg_demo', type=float, default=1.)
parser.add_argument('--reg_gcs', type=float, default=1.)
parser.add_argument('--reg_mech', type=float, default=1.)
parser.add_argument('--reg_risk', type=float, default=1.)
args = parser.parse_args()
print(args)

model_id = str(uuid.uuid4())
print(model_id)
args.model_id = model_id

args.layer_sizes = [int(i) for i in args.layer_sizes]
args.demo_idx = [int(i) for i in args.demo_idx]
args.gcs_idx = [int(i) for i in args.gcs_idx]
args.mech_idx = [int(i) for i in args.mech_idx]
args.risk_idx = [int(i) for i in args.risk_idx]
args.cond_features = args.cond_features.split('_')

out_dir = Path(args.output_dir)
out_dir.mkdir(exist_ok = True, parents = True)

## Data Preprocessing 
if args.nohead: 
    print("No Head Injuries")
    X_train = pd.read_pickle("final/new_X_train_nohead.pkl")
    X_test = pd.read_pickle("final/new_X_test_nohead.pkl")
    with open("final/minmax_train_nohead.pkl", "rb") as f: 
        X_train_svd_norm = pickle.load(f)
    with open("final/minmax_test_nohead.pkl", "rb") as f: 
        X_test_svd_norm = pickle.load(f)
        
X_train = pd.read_pickle("final/new_X_train.pkl")
X_test = pd.read_pickle("final/new_X_test.pkl")
with open("final/minmax_train.pkl", "rb") as f: 
    X_train_svd_norm = pickle.load(f)
with open("final/minmax_test.pkl", "rb") as f: 
    X_test_svd_norm = pickle.load(f)
demo = pd.read_pickle("final/demo.pkl")
gcs = pd.read_pickle("aux_signal/gcs_signal.pkl")
mech = pd.read_pickle("aux_signal/mech_coarse_signal.pkl")
risk = pd.read_pickle("aux_signal/high_risk_icd.pkl")

# Don't include race or gender
demo = demo[['age']]
# Normalize age 
min_age = demo['age'].min()
max_age = demo['age'].max()
demo['age'] = (demo['age'] - min_age)/(max_age - min_age)
cond = pd.merge(demo, gcs[['mild', 'moderate', 'severe']], 
                how = 'inner', left_index=True, right_index=True)
cond = pd.merge(cond, mech, how='inner',
                left_index=True, right_index=True)
cond = pd.merge(cond, risk, how='inner',
                left_index=True, right_index=True)

# Subset conditional values 
include_cond_cols = []
if "demo" in args.cond_features: 
    include_cond_cols.extend(['age'])
if "gcs" in args.cond_features:
    include_cond_cols.extend(['mild', 'moderate', 'severe']) 
if "mech" in args.cond_features: 
    include_cond_cols.extend(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']) 
if "risk" in args.cond_features: 
    include_cond_cols.extend(['high_risk'])
    
cond = cond[include_cond_cols]
cond[cond > 1] = 1
cond_train = cond.loc[X_train.index]
cond_test = cond.loc[X_test.index]

X_traint = tf.cast(np.concatenate([X_train_svd_norm, X_train.values, cond_train.values], axis = 1), tf.float32)
X_testt = tf.cast(np.concatenate([X_test_svd_norm, X_test.values, cond_test.values], axis = 1), tf.float32)

args.input_dim = X_train.shape[1]+X_train_svd_norm.shape[1]
args.num_classes = cond.shape[1]

if args.output_dir[-1] != "/": 
    args.output_dir = args.output_dir + "/" 
    
log_dir = Path(f"{args.output_dir}log/")
log_dir.mkdir(exist_ok = True, parents = True)

model_dir = Path(f"{args.output_dir}model/")
model_dir.mkdir(exist_ok = True, parents = True)

repr_dir = Path(f"{args.output_dir}repr/")
repr_dir.mkdir(exist_ok = True, parents = True)

print("Start training VAE!")
for i in range(args.repeats):
    log = {}
    log['args'] = args
    log['run'] = i
    log['model_id'] = model_id
    log['model_arch'] = args.model_arch
    
    encoder = Encoder(input_shape=(args.input_dim,), 
                      latent_dim = args.latent_dim, 
                      layer_sizes = args.layer_sizes)
    decoder = Decoder(output_shape=args.input_dim, 
                      latent_dim = args.latent_dim, 
                      layer_sizes = args.layer_sizes)
    classifier = Classifier(layer_size = args.classifier_layer_size,
                            num_classes = args.num_classes, 
                            classifier_activation = "sigmoid")

    if args.unweighted_classifier_loss: 
        model = BetaVAE_Classifier(encoder = encoder, 
                                decoder=decoder, classifier=classifier, 
                                beta = args.beta, gamma = args.gamma, 
                                classifier_loss = "binary crossentropy", 
                                classifier_activation= "sigmoid", 
                                num_classes = int(args.num_classes),
                                unweighted_classifier_loss = True,)
    else: 
        model = BetaVAE_Classifier(encoder = encoder, 
                                decoder=decoder, classifier=classifier, 
                                beta = args.beta, gamma = args.gamma, 
                                classifier_loss = "binary crossentropy", 
                                classifier_activation= "sigmoid", 
                                num_classes = int(args.num_classes), 
                                unweighted_classifier_loss = False, 
                                demo_idx = args.demo_idx, 
                                gcs_idx = args.gcs_idx, 
                                mech_idx = args.mech_idx, 
                                risk_idx = args.risk_idx, 
                                reg_demo = args.reg_demo, 
                                reg_gcs = args.reg_gcs, 
                                reg_mech = args.reg_mech, 
                                reg_risk = args.reg_risk,)

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr))
    history = model.fit(X_traint, 
                        epochs = args.epochs, 
                        batch_size = args.batch_size, 
                        verbose = 2, 
                        callbacks = [CustomEarlyStopping(patience = args.patience, kl_loss ="kl loss",
                                                         kl_threshold = 100, verbose = 1)])
    embedding, _, _ = model.encoder(X_traint[:, :-int(args.num_classes)])
    embedding = embedding.numpy()
    test_embedding, _, _ = model.encoder(X_testt[:, :-int(args.num_classes)])
    test_embedding = test_embedding.numpy()

    log['total loss'] = history.history['total loss'] 
    log['kl loss'] = history.history['kl loss']
    log['reconstruction loss'] = history.history['reconstruction loss']
    log['classifier loss'] = history.history['classifier loss']
    
    with open(f"{args.output_dir}log/log_{args.model_arch}_{model_id}_run{i}.pkl", "wb") as f: 
        pickle.dump(log, f)
    with open(f"{args.output_dir}repr/train_embed_{args.model_arch}_{model_id}_run{i}.pkl", "wb") as f: 
        pickle.dump(embedding, f)
    with open(f"{args.output_dir}repr/test_embed_{args.model_arch}_{model_id}_run{i}.pkl", "wb") as f: 
        pickle.dump(test_embedding, f)
    model.save_weights(f"{args.output_dir}model/model_{args.model_arch}_{model_id}_run{i}")
print("Finished training VAE!")