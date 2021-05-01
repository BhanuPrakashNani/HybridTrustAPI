from typing import Optional
from fastapi import FastAPI

import numpy as np
import pandas as pd

app = FastAPI()

user_weights = pd.read_csv('../userWeights.csv')
user_feedback = pd.read_csv('../CSP_QoS_UserFeedback1.csv')
csp_promised_parameters = pd.read_csv('../CSP_Promised_Parameters (1).csv')

from sklearn.metrics.pairwise import cosine_similarity
import operator

def calculate_tuci(user_inp, csp_params):
  return np.stack((csp_params[:,0], np.dot(csp_params[:,1:], user_inp.T)/100), axis =1)

def similar_users(input_weights, other_users, k=3):
    similarities = cosine_similarity(input_weights,other_users)[0].tolist()
    indices = other_users.index.tolist()
    index_similarity = dict(zip(indices, similarities))

    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]
    return top_users_similarities

def getlist(input, topk, user_feedback):
  csp_set = set()
  dummy = user_feedback.drop(["Cloud_Consumer_Name","Cloud_Service_Name", "Timestamp"],axis=1)
  # print(similar_users(input, dummy, topk))
  for k,v in similar_users(input, dummy, topk):
    csp_set.add((user_feedback.loc[k]['Cloud_Service_Name'],v, user_feedback.loc[k]['Timestamp'].strip()))
  return list(csp_set)

def average_trust_per_csp(cllb_filter_output):
    csp_set = set()
    for i in cllb_filter_output:
        csp_set.add(i[0])
    csp_set = list(csp_set)
    # print(csp_set)

    weighted_list = []
    for csp in csp_set:
      filter_csp=[]
      scores = []
      
      for i in cllb_filter_output:
        if(i[0]==csp):
          filter_csp.append(i)
      
      if(len(filter_csp) != 1):
        # print(filter_csp)
        for i in filter_csp:
          scores.append(i[1])
        # print(csp)
        # print(min(scores), max(scores))
        weighted_list.append([csp, sum(scores)/len(scores)])
      else:
        weighted_list.append([csp,filter_csp[0][1]])

    weighted_list = np.array(weighted_list)
    return weighted_list

def merge_tuci_tfci(tfci, tuci, weights, topk):
  final_trust = []
  for i in range(len(tuci)):
    row =[]
    if(len(tfci)!= 0):
      row = tfci[np.where(tfci[:,0] == tuci[i][0])] 
    f = 0

    if(len(row)!=0):
      f = float(row[0][1])
    u = tuci[i][1]
    final_trust.append([tuci[i][0],u,f, weights[0]*u + weights[1]*f])
    final_trust.sort(key = lambda x : x[3], reverse = True)
  return final_trust[:topk]

@app.get('/get_topk')
def read_topk(weights_vals: str, topk: int):
    weights = np.array(list(map(int, weights_vals.split(' '))))
    ret = merge_tuci_tfci(average_trust_per_csp(getlist(weights.reshape(1,4), 100, user_feedback)), calculate_tuci(weights, csp_promised_parameters.values
), [0.75, 0.25], topk)
    return ret

@app.get('/feedback_val')
def validation(csp_name: str, vals: str):
    vals = list(map(int, vals.split(' ')))
    vals = (pd.DataFrame(vals)).T
    vals.columns = ['Uptime', 'Downtime', 'Fault_Tolerance_Capability', 'Application_Update_Frequency']

    data1 = user_feedback[user_feedback['Cloud_Service_Name']==csp_name]
    data1.drop(['Cloud_Consumer_Name', 'Cloud_Service_Name', 'Timestamp'], axis=1)

    Q1 = data1.quantile(0.25)
    Q3 = data1.quantile(0.75)
    IQR = Q3 - Q1

    count = (vals < (Q1 - 1.5 * IQR)) | (vals > (Q3 + 1.5 * IQR))
    count = count*1
    count["sum"] = count.sum(axis=1)

    if (count['sum']>2).bool():
        # ind.append(j)
        return {'bool':False}
    return {'bool':True}
    