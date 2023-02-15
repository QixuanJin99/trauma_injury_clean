import pandas as pd
import numpy as np
import itertools

data_folder = 'data/old'
s_first_mapping = pd.read_csv(data_folder + '/preprocessed/phys_distance_s_first.csv', index_col = 0)
s_first_mapping.columns = s_first_mapping.columns.astype(str)
s_first_mapping.index = s_first_mapping.index.astype(str)

s_second_mapping = pd.read_csv(data_folder + '/preprocessed/phys_distance_s_second.csv')
s_second_mapping['first_code'] = s_second_mapping['first_code'].astype(str)
s_second_mapping['second_code'] = s_second_mapping['second_code'].astype(str)
s_second_mapping['internal'] = s_second_mapping['internal'].astype(int)

high_risk_icd = pd.read_csv("final/high_risk_50.csv", index_col = 0)
high_risk_index = high_risk_icd.index.tolist()

def _commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

def icd_hierarchy_metric(m, inverse = True):
    # The maximum number of matches 
    max_letters = 7
    
    if len(m) <= 1: 
        # Clusters that are single are returned as -1 (for sorting later)
        return -1
    if len(m) == 2: 
        # Just return the number of characters in overlapping prefix
        if inverse: 
            return max_letters - len(_commonprefix(m).replace('.',''))
        else: 
            return len(_commonprefix(m).replace('.',''))
    # If cluster has more than 2 entries 
    all_scores = []
    for i in range(2, len(m)):
        tmp = list(itertools.combinations(m, i))
        scores = np.array([len(_commonprefix(list(mi)).replace('.','')) for mi in tmp])
        all_scores.append(np.mean(scores)) 
    
    if inverse:
        return max_letters - np.mean(np.array(all_scores))
    else: 
        return np.mean(np.array(all_scores))

def _internal_external_score(m1, m2): 
    # -1 for superficial injury, 1 for internal, 0 for other, 0.5 for nerves or blood vessels
    m1_score = s_second_mapping[(s_second_mapping['first_code'] == m1[1]) 
                                & (s_second_mapping['second_code'] == m1[2])]['internal'].values[0]

    m2_score = s_second_mapping[(s_second_mapping['first_code'] == m2[1]) 
                                & (s_second_mapping['second_code'] == m2[2])]['internal'].values[0]
    
    if m1_score == -1 or m2_score == -1: 
        return -1 
    elif m1_score == m2_score: 
        return m1_score * 0.5
    else: 
        return abs(m1_score - m2_score)

def _phys_distance(m1, m2): 
    if m1[0] == 'S' and m2[0] == 'S': 
        # look at the first number after S, graph of physical distances 
        # Scale from 0 to 7 -> normalize
        first_score = s_first_mapping.loc[m1[1], m2[1]] / 7
        if first_score == 0: 
            first_score = 0.01
        second_score = _internal_external_score(m1, m2)
        return first_score, second_score
    elif m1[0] == 'T' and m2[0] == 'T':
        return 0, 0
    # Other diseases in cluster
    else: 
        if m1[0] == 'S' or m2[0] == 'S':
            return 0.5, 0
        else: 
            return 0, 0

def _risk_score(m1, m2): 
    score = 0
    if m1 in high_risk_index: 
        score += 0.5
    if m2 in high_risk_index: 
        score += 0.5
    return score

def _downweight_head(m1, m2): 
    score = 0
    if m1[0:2] == 'S0': 
        score -= 0.5
    if m2[0:2] == 'S0': 
        score -= 0.5 
    return score
    
def phys_distance_metric(m): 
    if len(m) <= 1: 
        # Clusters that are single are returned as -1 (for sorting later)
        return -1
    if len(m) == 2: 
        # Just return the number of characters in overlapping prefix
        return _phys_distance(m[0], m[1])
    # If cluster has more than 2 entries 
    # All subsets of certain length 2
    tmp = list(itertools.combinations(m, 2))
    scores = np.array([_phys_distance(mi[0], mi[1]) for mi in tmp])
    return np.mean(scores)

def get_count_frequency(df, codes): 
    # Given a list of injury codes (cleaned). e.g. ['S36.4', 'S36.5', 'S36.8'] 
    # Returns the number of patients with a subset of patterns 
    
    set_codes = set(codes)
    return len(df[df.apply(lambda x: set_codes.issubset(set(x)))])