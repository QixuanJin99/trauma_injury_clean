import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle
from pathlib import Path
import os
import argparse

from sklearn import metrics as skm 
from sklearn.preprocessing import MinMaxScaler
import umap
from scipy.stats import pearsonr
from metrics import _phys_distance, _risk_score, _downweight_head

import networkx as nx
from networkx.algorithms.components.connected import connected_components

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def generate_pattern_desc(pattern): 
    pattern_desc = []
    for s in pattern: 
        pattern_desc.append([mapping[icd] for icd in s])
    return pattern_desc

def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_path', type=str, default = "")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--umap_embed_path', type=str, default = "")
parser.add_argument('--num_samples', type=int, default = 10000)
parser.add_argument('--cluster_alg', type=str, default = "kmeans")
parser.add_argument('--num_clusters', type=int, default = 30)
parser.add_argument('--min_cluster_size', type=int, default=259)
parser.add_argument('--cluster_freq_threshold', type=float, default=0.05) 
parser.add_argument('--pvalue_threshold', type=float, default=0.05)
parser.add_argument('--num_top_corr_pairs', type=int, default=50)
parser.add_argument('--min_corr_threshold', type=float, default=0.3) 
parser.add_argument('--body_spatial_weight', type=float, default=0.5) 
parser.add_argument('--internal_external_weight', type=float, default=0.2) 
parser.add_argument('--risk_weight', type=float, default=0.2) 
parser.add_argument('--corr_weight', type=float, default=0.1) 
parser.add_argument('--head_weight', type=float, default=0.3) 
parser.add_argument('--save_fig', action = 'store_true') 
parser.add_argument('--nohead', action = 'store_true')
parser.add_argument('--not_cr', action = 'store_true') 
args = parser.parse_args()
print(args)

t = args.embedding_path
# args.output_dir = t.split("/")[0] + "/" + t.split("/")[1] + "/"

out_dir = Path(f"{args.output_dir}cr/")
out_dir.mkdir(exist_ok = True, parents = True)

if args.save_fig:
    out_dir_fig = Path(f"{args.output_dir}cr/fig/")
    out_dir_fig.mkdir(exist_ok = True, parents = True)

embed = pd.read_pickle(t) 
print(embed.shape)

scaler = MinMaxScaler()
embed_norm = scaler.fit_transform(embed)

index = np.random.choice(embed.shape[0], args.num_samples, replace=False)

# Load precomputed UMAP 
if args.save_fig:
    umap_embed = pd.read_pickle(args.umap_embed_path)

# Load labels 
test_index = np.loadtxt("final/new_test_index.csv").astype(int)
demo = pd.read_pickle("final/demo.pkl")
gcs = pd.read_pickle("aux_signal/gcs_signal.pkl")
mech = pd.read_pickle("aux_signal/mech_coarse_signal.pkl")
risk = pd.read_pickle("aux_signal/high_risk_icd.pkl")

mech[mech > 1] = 1
demo = demo[['age', 'gender']]
cond = pd.merge(demo, gcs[['mild', 'moderate', 'severe']], 
                how = 'inner', left_index=True, right_index=True)
cond = pd.merge(cond, mech, how='inner',
                left_index=True, right_index=True)
cond = pd.merge(cond, risk, how='inner',
                left_index=True, right_index=True)
cond = cond.loc[test_index]

patient_char = ['age', 'gender', 'mild', 'moderate', 'severe', 
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'high_risk']
cond = cond[patient_char]

## Clustering -------------------------------------------------------------
print("Performing Clustering")
clustering_results = {}
if args.cluster_alg == "kmeans": 
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = args.num_clusters, n_init = 10, 
                    max_iter = 500, random_state=42, verbose=0).fit(embed[index, :])
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    chunk_size = 10000
    all_labels = np.array([])
    for i in range(int(embed.shape[0] / chunk_size) + 1): 
        high = min((i+1)*chunk_size, embed.shape[0])
        all_labels = np.append(all_labels, kmeans.predict(embed[i*chunk_size:high]))
else: 
    raise Exception("Invalid Clustering Alg")
    
clustering_results['labels'] = labels
clustering_results['all_labels'] = all_labels
clustering_results['centroids'] = centroids

## Unsupervised Clustering Metrics ----------------------------------------
print("Computing Unsupervised Clustering Metrics")
model_summary = {}
model_summary['model_full_ch_index'] = skm.calinski_harabasz_score(embed_norm, all_labels)

approx_silh = []
for i in range(10): 
    sindex = np.random.choice(embed.shape[0], 10000, replace=False)
    approx_silh.append(skm.silhouette_score(embed_norm[sindex, :], all_labels[sindex]))
model_summary['model_avg_silh_score'] = np.mean(approx_silh)

model_summary['model_ch_index'] = skm.calinski_harabasz_score(embed_norm[index, :], labels)
model_summary['model_silh_score'] = skm.silhouette_score(embed_norm[index, :], labels)

## Clinical Relevance Score -----------------------------------------------

print("Computing CR Score")
if args.nohead: 
    X_test = pd.read_pickle("final/new_X_test_nohead.pkl")
else: 
    X_test = pd.read_pickle("final/new_X_test.pkl")
sample = X_test
sample['label'] = all_labels

discovered_clusters = {}
for i in range(args.num_clusters): 
    df = sample[sample['label'] == i].drop(columns = ['label'])
    if len(df) < args.min_cluster_size: 
        continue
    # Select conditions above a certain frequency threshold 
    high_freq_cols = df.columns[df.mean(axis = 0) > args.cluster_freq_threshold].tolist()
    df = df[high_freq_cols]
    
    pcorr = df.corr(method=pearsonr_pval)
    # Mask out pairs below a certain pvalue threshold
    pcorr = pcorr[pcorr < args.pvalue_threshold].fillna(1)
    
    corr = df.corr()
    remove_cols = corr.columns[corr.isna().all()]
    corr = corr.drop(remove_cols, axis = 0)
    corr = corr.drop(remove_cols, axis = 1)
    for c in corr: 
        corr.loc[c, c] = 0.
    tmp = corr.stack().nlargest(args.num_top_corr_pairs).index.tolist()
    
    discovered_clusters[i] = []
    every_other = True
    for (r, c) in tmp: 
        if every_other: 
            every_other = False 
            continue
        else: 
            every_other = True
        
        if pcorr.loc[r, c] >= 1 or corr.loc[r, c] < args.min_corr_threshold: 
            continue
        discovered_clusters[i].append(((r, c), pcorr.loc[r, c], corr.loc[r, c].round(3))) 

if args.not_cr:
    df_results = pd.DataFrame(columns = ['cluster-num', 'num-patients', 'pattern'] + patient_char)
    j = 0
    for i, pairs in discovered_clusters.items(): 
        if len(pairs) == 0: 
            continue 

        df_pairs = pd.DataFrame(columns = ['m1', 'm2'])
        for c in pairs: 
            m1, m2 = c[0]
            df_pairs = df_pairs.append({'m1': m1, 'm2': m2}, ignore_index=True)

        num_patients = len(all_labels[all_labels == i])
        pairs_list = list(df_pairs[['m1', 'm2']].itertuples(index=False, name=None))
        G = to_graph(pairs_list)
        pattern = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        pattern = [list(s) for s in pattern]

        # Group into larger pattern 
        df_results = df_results.append({
            'cluster-num': i, 
            'num-patients': num_patients, 
            'pattern': pattern,
        }, ignore_index=True)

        df_results.loc[j, patient_char] = cond[all_labels == i][patient_char].mean(axis=0).round(3).values
        j += 1
else: 
    df_results = pd.DataFrame(columns = ['cluster-num', 'cr-score', 'num-patients', 'pattern'] + patient_char)
    num_meaningful = {}
    num_meaningful_clusters = 0
    avg_model_score = 0
    j = 0
    for i, pairs in discovered_clusters.items(): 
        if len(pairs) == 0: 
            continue 

        print("\nCluster {}".format(i))
        df_pairs = pd.DataFrame(columns = ['m1', 'm2', 'body-spatial', 
                                           'internal-external', 'high-risk', 
                                           'correlation', 'head', 'pvalue', 
                                           'cr-score'])
        num_meaningful[i] = 0
        cluster_avg_score = 0
        for c in pairs: 
            m1, m2 = c[0]
            body_spatial_score, internal_external_score = _phys_distance(m1, m2)
            risk_score = _risk_score(m1, m2)
            head_score = _downweight_head(m1, m2)

            correlation = c[2]
            weighted_score = args.body_spatial_weight * body_spatial_score + \
                             args.internal_external_weight * internal_external_score + \
                             args.risk_weight * risk_score + \
                             args.head_weight * head_score

            if weighted_score > 0:
                # Add correlation factor after filtering, since correlation always positive 
                weighted_score += args.corr_weight * correlation
                num_meaningful[i] += 1
                cluster_avg_score += round(weighted_score, 4)
                print("Pair ({}, {})".format(m1, m2))
                print("Body-spatial: {}".format(body_spatial_score))
                print("Internal-External: {}".format(internal_external_score))
                print("High-Risk: {}".format(risk_score))
                print("Correlation: {}".format(correlation))
                print("Weighted score: {}".format(round(weighted_score, 4)))
                df_pairs = df_pairs.append({'m1': m1, 'm2': m2, 
                                'body-spatial': body_spatial_score, 
                                'internal-external': internal_external_score, 
                                'high-risk': risk_score, 
                                'correlation': correlation, 
                                'head': head_score,
                                'pvalue': c[1],
                                'cr-score': weighted_score}, ignore_index=True)

        if num_meaningful[i] == 0: 
            continue

        num_meaningful_clusters += 1
        cluster_avg_score /= num_meaningful[i]
        avg_model_score += cluster_avg_score
        num_patients = len(all_labels[all_labels == i])

        cluster_dict = {}
        cluster_dict['cluster-num'] = i 
        cluster_dict['num-patients'] = num_patients
        cluster_dict['avg-cr-score'] = round(cluster_avg_score, 4)
        cluster_dict['df_pairs'] = df_pairs

        print("Num of Patients in Test: {}".format(num_patients))
        print("Avg Cluster Score: {}".format(round(cluster_avg_score, 4)))
        model_summary[i] = cluster_dict

        pairs_list = list(df_pairs[['m1', 'm2']].itertuples(index=False, name=None))
        G = to_graph(pairs_list)
        pattern = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        pattern = [list(s) for s in pattern]

        # Group into larger pattern 
        df_results = df_results.append({
            'cluster-num': i, 
            'cr-score': round(cluster_avg_score, 4), 
            'num-patients': num_patients, 
            'pattern': pattern,
        }, ignore_index=True)

        df_results.loc[j, patient_char] = cond[all_labels == i][patient_char].mean(axis=0).round(3).values
        j += 1

        if args.save_fig: 
            plt.figure(figsize=(7, 5), dpi = 200)
            plt.scatter(
                umap_embed[index, 0][labels != i],
                umap_embed[index, 1][labels != i], 
                alpha = 1., 
                s = 0.3, 
                c = labels[labels != i], 
            )
            plt.scatter(
                umap_embed[index, 0][labels == i],
                umap_embed[index, 1][labels == i], 
                alpha = 1., 
                s = 1, 
                color = "red", 
            )
            plt.savefig(f"{args.output_dir}cr/fig/umap_highlight_cluster{i}.png", dpi = 200)
    model_summary['model_cr_score'] = round(avg_model_score / num_meaningful_clusters, 4)
    model_summary['model_cr_score_k30'] = round(avg_model_score / 30, 4)
    with open(args.embedding_path.replace("repr", "cr").replace("test_embed", "model-summary"), "wb") as f: 
        pickle.dump(model_summary, f)


df_desc = pd.read_csv('old/exp1_repeated/unique_icds_desc.csv')
mapping = dict(zip(df_desc['ICD10_DCODE_CLEANED'], df_desc['ICD_DESC']))

df_results['pattern-desc'] = df_results['pattern'].apply(lambda p: generate_pattern_desc(p))
if args.not_cr: 
    df_results = df_results[['cluster-num', 'num-patients', 'pattern', 'pattern-desc'] + patient_char]
else: 
    df_results = df_results[['cluster-num', 'cr-score', 'num-patients', 'pattern', 'pattern-desc'] + patient_char]
    df_results = df_results.sort_values(by=["cr-score"], ascending=False)
df_results = df_results.reset_index(drop=True)

print("Saving results!")
## Save everything ----------------------------------------------------------------------------------------------
if args.not_cr:
    np.savetxt(args.embedding_path.replace("repr", "cr").replace("test_embed", "index-NOT").replace(".pkl", ".csv"), index, delimiter=",")
    df_results.to_csv(args.embedding_path.replace("repr", "cr").replace("test_embed", "df-cr-NOT").replace(".pkl", ".csv"))
    with open(args.embedding_path.replace("repr", "cr").replace("test_embed", "cluster-results-NOT"), "wb") as f:
        pickle.dump(clustering_results, f)
else:
    if args.nohead:
        df_results.to_csv(args.embedding_path.replace("repr", "cr").replace("test_embed", "df-cr").replace(".pkl", ".csv"))
    else: 
        np.savetxt(args.embedding_path.replace("repr", "cr").replace("test_embed", "index").replace(".pkl", ".csv"), index, delimiter=",")
        df_results.to_csv(args.embedding_path.replace("repr", "cr").replace("test_embed", "df-cr-HEAD").replace(".pkl", ".csv"))
        with open(args.embedding_path.replace("repr", "cr").replace("test_embed", "cluster-results"), "wb") as f:
            pickle.dump(clustering_results, f)

print("FINISHED!")