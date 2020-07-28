import pandas as pd
import sys
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import random
from collections import defaultdict 

predictions = np.load(sys.argv[1])
annotation = pd.read_csv('rareact.csv')
if len(sys.argv) > 2:
  n_sampling = int(sys.argv[2])
else:
  n_sampling = 0

assert len(predictions) == len(annotation)
video_ids = annotation['video_id'].values
positive_negative_ind = {}
action_to_classid = {}
for i in range(len(annotation)):
  action = (annotation['verb'].values[i], annotation['noun'].values[i])
  action_to_classid[action] = annotation['class_id'].values[i]
  if action not in positive_negative_ind:
    positive_negative_ind[action] = {'positive': [], 'negative': [], 'hard negative': []}

for i in range(len(annotation)):
  v = annotation['verb'].values[i]
  n = annotation['noun'].values[i]
  annot = annotation['annotation'].values[i]
  if annot == 0:
    positive_negative_ind[(v,n)]['negative'].append(i)
  elif annot == 1:
    positive_negative_ind[(v,n)]['positive'].append(i)
    for action in positive_negative_ind:
      if (action[0] == v and action[1] != n) or (action[0] != v and action[1] == n):
        positive_negative_ind[action]['hard negative'].append(i)
      elif action[0] != v and action[1] != n:
        positive_negative_ind[action]['negative'].append(i)
  else:
    positive_negative_ind[(v,n)]['hard negative'].append(i)

def vid_sampling(vid):
  all_vid = set()
  for v in vid:
    all_vid.add(v)
  out = np.zeros((len(all_vid)), dtype=int)
  for i, v in enumerate(all_vid):
    inds = np.where(vid == v)[0]
    out[i] = inds[random.randint(0, len(inds) - 1)]
  return out

def normalize_nid(l):
  count = defaultdict(int)
  for el in l:
    count[el] += 1
  out = np.ones((len(l)))
  for i in range(len(l)):
    out[i] = 1.0 / count[l[i]]
  return out

all_scores = {}
mAP = 0

for action in positive_negative_ind:
  ind = positive_negative_ind[action]
  class_id = action_to_classid[action]
  pos_score = predictions[ind['positive'], class_id]
  if len(ind['positive']) > 0: 
    hneg_score = predictions[ind['hard negative'], class_id] 
    neg_score = predictions[ind['negative'], class_id]
    vid_pos = video_ids[ind['positive']]
    vid_hneg = video_ids[ind['hard negative']]
    vid_neg = video_ids[ind['negative']]
    if n_sampling > 0:
      score = 0
      for i in range(n_sampling):
        pos_sampling = vid_sampling(vid_pos)
        neg_sampling = vid_sampling(vid_neg)
        hneg_sampling = vid_sampling(vid_hneg)
        sampled_pos_score = pos_score[pos_sampling]
        sampled_vid_pos = vid_pos[pos_sampling]
        sampled_neg_score = neg_score[neg_sampling]
        sampled_vid_neg = vid_neg[neg_sampling]
        sampled_hneg_score = hneg_score[hneg_sampling]
        sampled_vid_hneg = vid_hneg[hneg_sampling]

        scores = np.concatenate([sampled_pos_score, sampled_neg_score, sampled_hneg_score], axis=0) 
        label = np.zeros((len(scores)), dtype=int)
        label[:len(sampled_pos_score)] = 1 
        score += average_precision_score(label, scores)
      score /= n_sampling
    else:
      sample_weight_pos = normalize_nid(vid_pos)
      sample_weight_neg = normalize_nid(vid_neg)
      sample_weight_hneg = normalize_nid(vid_hneg)
      sample_weight = np.concatenate([sample_weight_pos, sample_weight_neg, sample_weight_hneg], axis=0) 
    
      scores = np.concatenate([pos_score, neg_score, hneg_score], axis=0) 
      label = np.zeros((len(scores)), dtype=int)
      label[:len(pos_score)] = 1 
      score = average_precision_score(label, scores, sample_weight=sample_weight)
    all_scores[action] = score
    mAP += score

if n_sampling > 0:
    print('mSAP (n = {}) {}'.format(n_sampling, mAP / float(len(all_scores))))
else:
    print('mWAP {}'.format(mAP / float(len(all_scores))))
