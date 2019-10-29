import pdb
import numpy as np
import json
import scipy.sparse as sp
import sys 
from collections import defaultdict, deque
import grakel_all
import time

print("loaded text")

import sys

def graphOfWords(text):
    text_words = text#.split()
    graph_window = 3
    # Graph of words 
    vertices_words = list(set(text_words))
    vertices_index = []

    edges = []
    for index1 in range(min([len(text_words), max([1,len(text_words)-graph_window])])):
        word1 = text_words[index1]
        
        for index2 in range(1, min([len(text_words)-index1, graph_window])):
            word2 = text_words[index1 + index2]
            vertex1 = word1
            vertex2 = word2
            edges.append((vertex1, vertex2)) 
         
    # GraKeL graphs
    A = {}
    for ed in edges:
        A[ed[0]] = []
        A[ed[1]] = []
    
    for ed in edges:
        if ed[1] not in A[ed[0]]:
            A[ed[0]].append(ed[1])
        if ed[0] not in A[ed[1]]:
            A[ed[1]].append(ed[0])

    B = {vw: vw for vw in vertices_words}

    if len(B) == 1:
        return [{vertices_words[0]: [vertices_words[0]]}, B]

    if len(B) == 0:
        B = {0: 0}    

    if len(A) == 0:        
        return [{0:[0]},B]
    else:
        return [A,B]


import time

def returnXDataLine(dic_line, K=10):
    # input_dic example: {"mention_id" : 1, "mention_text_features": [0,1,3,4], "score_results": {"E1": [name_score, context_score], "E10": [name_score, context_score]}}
    #dic_line = json.loads(dic_line)

    start = time.time()
    mention_id = dic_line["mention_id"]
    no_kernel = False
    X_mention = []
    candidate_entities = dic_line["score_results"][:K]
   
    #text = mentionIdToText[mention_id]
    text = dic_line["mention_text_words"]
    H_gk = graphOfWords(text) 
    
    gk_q = grakel_all.GraphKernel(kernel=dict(name="pyramid_match", with_labels=True), normalize=True)
    gk_q.fit_transform([H_gk])
     
    print("gow done")
    dic_return = {"mention_id": mention_id}
    #candidate_entities = [(key, candidate_entities[key][0], candidate_entities[key][1]) for key in candidate_entities.keys()]

    for ceo in candidate_entities:
        #print(ceo[0])
        X_line = []    
        ce = ceo["entity_id"]

        X_line.append(ceo["name_score"])
        X_line.append(ceo["context_score"])
    
        type_features = []
        indexes = []
        counter_type = -1
        ce_neighbors_total = [(nei, entityIdToType[nei]) for nei in  id_to_neighbors[ce]]
        ce_neighbors_total = [x[0] for x in ce_neighbors_total]
        
        for type_ in types2:
            counter_type += 1
            ce_neighbors = []
            nb_nei_limit = 0 
            ce_neighbors = [nei for nei in  ce_neighbors_total if entityIdToType[nei] == type_]
            neighbors_of_type = {nei: len(id_to_neighbors[nei]) for nei in ce_neighbors}
            sorted_x = sorted(neighbors_of_type.items(), key=lambda x: x[1], reverse=True)
            neighbors_of_type = [sx[0] for sx in sorted_x][:3] 
            type_text = [] 
            for i in range(len(neighbors_of_type)):
                text_limited = " ".join(entityIdToText[neighbors_of_type[i]].split()[:100])
                type_text.append(text_limited)
            type_text = " ".join(type_text)
          
            if len(neighbors_of_type) == 0:
                type_features.append([{0: [0]}, {0 :0}])
            else:
                type_features.append(graphOfWords(type_text.split()))
                indexes.append(counter_type)
            #if type_ == "City":
            #    pdb.set_trace()
    
        if len(indexes) > 0:   
            graph_kernel_scores =  gk_q.transform([type_features[i] for i in indexes]).tolist()
        graph_scores = []
        gks_index = 0
        for i in range(len(type_features)):
            if i in indexes:
                graph_scores.append(graph_kernel_scores[gks_index][0])
                gks_index += 1
            else:
                graph_scores.append(0.)
    
        for gs in graph_scores:
            X_line.append(gs)
    
        dic_return[ce] = X_line

    end = time.time()
    print("TIME: " + str(end - start))
    #pdb.set_trace()    
    return dic_return


def getEntityIds(x):
    dic_x = json.loads(x)
    ers = dic_x["entity_ranks"]
    return [er[0] for er in ers]

def loadF(x):
    dic_x = json.loads(x)
    return (dic_x["gold_entity_id"], dic_x["mention_text_features"])


def getTrainingSamples(x):
    gold_entity_id = x["gold_entity_id"]
    X = []
    Y = []


    if gold_entity_id not in [xx["entity_id"] for xx in x["score_results"]]:
        return [], []

    X_features = returnXDataLine(x) 
    for eid in X_features.keys():
        if eid[0] == "E": 
            if eid == gold_entity_id:
                y = 1
            else:
                y = 0
        
            X.append(X_features[eid])
            Y.append(y)

    n_x = len(X)
    if n_x < 10:
        for ix in range(10-n_x):
            X.append([0. for i in range(len(X[0]))])
            Y.append(0)

    return X, Y    

#########################################################################################

entityIdToText = json.load(open("entityIdToText_micro_kb.json", "r"))
entityIdToTextFeatures = json.load(open("entityIdToTextFeatures.json", "r"))
entityIdToType = json.load(open("entityIdToOntology_micro_kb.json", "r"))
id_to_neighbors = json.load(open("id_to_neighbors_micro_kb.json", "r"))
types_init = open("types_init").read().split(", ")
ontology_map = json.load(open("project-ontology.json", "r"))

#depth = sys.argv[2]
depth=3
types2 = []
for ty in types_init:
    try:
        ty_n = ontology_map[depth][ty]
    except:
        ty_n = ty
    types2.append(ty_n)

types2 = list(set(types2))


# EXAMPLE INFERENCE
#dic_line = {"mention_id"= "0", "mention_text_features": [0,1,3,4], "entity_ranks": {"E1": [name_score, ], "E10": [name_score, context_score]}}

#with open("identification_example.json", "r") as f:
#    dic_line = json.load(f)
#
#mentionIdToText = {dic_line["mention_id"]: dic_line["mention_text_features"]}
#x_dic = returnXDataLine(dic_line)

X_train = []
Y_train = []
index_line = 0
with open("training_samples_micro_kb.json", "r") as f:
    for line in f:
        dic_line = json.loads(line)
        X_line, Y_line = getTrainingSamples(dic_line)
        X_train = X_train + X_line
        Y_train = Y_train + Y_line
        index_line += 1
        print(index_line)
        if index_line == 10:
            break





##################################################################################################################################

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree

from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
# All dataset
results_gold = []
#
if 1:
    # No PCA
    n_samples = len(Y_train)
    nb_samples_entities = n_samples/10
    n_train = int(nb_samples_entities*0.5)*10
    Y_test = Y_train[n_train:]
    Y_train = Y_train[:n_train]
    X_train_dim = X_train[:n_train]
    X_test_dim = X_train[n_train:]

    nb_entities_test = len(X_test_dim)/10
    for penal in [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]:
        scores = 0
        print(penal)
        predictor = 6
        regr_2 = linear_model.LogisticRegression(C = penal, class_weight = "balanced")
        regr_2.fit(X_train_dim,Y_train)
        results = regr_2.predict_proba(X_test_dim).tolist()
        results = [res[1] for res in results]

        n_results_batch = len(results)/10
        for i in range(n_results_batch):
            imax = np.argmax(results[i*10:(i+1)*10])    
            score_i = int(Y_test[i*10:(i+1)*10][imax] == 1)
            scores += 1
            
        pdb.set_trace()        

pdb.set_trace()
