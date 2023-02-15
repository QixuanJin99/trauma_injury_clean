import numpy as np 
import pandas as pd 
import pickle
import argparse
from pathlib import Path
import umap 

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_path', type=str, default = "")
parser.add_argument('--output_dir', type=str, default = "")
parser.add_argument('--model_prefix', type=str, default = "")
args = parser.parse_args()
print(args)

embedding = pd.read_pickle(args.embedding_path) 
print(embedding.shape)

print("Starting computing UMAP")
reducer = umap.UMAP(n_neighbors = 20, 
                    min_dist = 0.1)
 
u = reducer.fit_transform(embedding)
print(u.shape)

out_dir = Path(f"{args.output_dir}visual/")
out_dir.mkdir(exist_ok = True, parents = True)
with open(f"{args.output_dir}visual/{args.model_prefix}_umap_embed.pkl", "wb") as f: 
    pickle.dump(u, f)
with open(f"{args.output_dir}visual/{args.model_prefix}_umap_model.pkl", "wb") as f: 
    pickle.dump(reducer, f)
    
print("Finished computing UMAP!")