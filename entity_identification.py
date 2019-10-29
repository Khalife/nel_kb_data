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

def returnXDataLine(dic_line):
    # input_dic example: {"mention_id" : 1, "mention_text_features": [0,1,3,4], "entity_ranks": {"E1": [name_score, context_score], "E10": [name_score, context_score]}}
    #dic_line = json.loads(dic_line)

    start = time.time()
    mention_id = dic_line["mention_id"]
    no_kernel = False
    X_mention = []
    candidate_entities = dic_line["entity_ranks"]
   
    #text = mentionIdToText[mention_id]
    text = dic_line["mention_text_features"]
    H_gk = graphOfWords(text) 
    
    gk_q = grakel_all.GraphKernel(kernel=dict(name="pyramid_match", with_labels=True), normalize=True)
    gk_q.fit_transform([H_gk])
     
    print("gow done")
    dic_return = {"mention_id": mention_id}
    candidate_entities = [(key, candidate_entities[key][0], candidate_entities[key][1]) for key in candidate_entities.keys()]
    for ceo in candidate_entities[:10]:
        #print(ceo[0])
        X_line = []    
        ce = ceo[0]

        X_line.append(ceo[1])
        X_line.append(ceo[2])
    
        type_features = []
        indexes = []
        counter_type = -1
        ce_neighbors_total = [(nei, entityIdToType[nei]) for nei in  id_to_neighbors[ce]]
        #types2 = list(set([x[1] for x in ce_neighbors_total]))
        ce_neighbors_total = [x[0] for x in ce_neighbors_total]
        
        for type_ in types2:
            counter_type += 1
            ce_neighbors = []
            nb_nei_limit = 0 
            #pdb.set_trace()
            ce_neighbors = [nei for nei in  ce_neighbors_total if entityIdToType[nei] == type_]
            #neis_types = []
            ##print("ok0")
            #for nei in ce_neighbors:
            #    try: 
            #        neis_types.append(ontology_map[depth][entityIdToType[nei]])
            #    except:
            #        neis_types.append(entityIdToType[nei])
            neighbors_of_type = {nei: len(id_to_neighbors[nei]) for nei in ce_neighbors}
            sorted_x = sorted(neighbors_of_type.items(), key=lambda x: x[1], reverse=True)
            neighbors_of_type = [sx[0] for sx in sorted_x][:3] 
            type_text = [] 
            #print("ok1")
            #print(len(neighbors_of_type))
            for i in range(len(neighbors_of_type)):
                #text_limited = " ".join(entityIdToText[ceo[0]].split()[:100])
                text_limited = " ".join(entityIdToText[neighbors_of_type[i]].split()[:100])
                type_text.append(text_limited)
            type_text = " ".join(type_text)
          
            if len(neighbors_of_type) == 0:
                type_features.append([{0: [0]}, {0 :0}])
            else:
                type_features.append(graphOfWords(type_text))
                indexes.append(counter_type)
    
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
    
    return dic_return


def getEntityIds(x):
    dic_x = json.loads(x)
    ers = dic_x["entity_ranks"]
    return [er[0] for er in ers]

def loadF(x):
    dic_x = json.loads(x)
    return (dic_x["gold_entity_id"], dic_x["mention_text_features"])

#############################################################################################################################
#############################################################################################################################
#print("Getting id to text...")
#rdd_train = sc.textFile("filtering_results_results_filtering_frequency_aida_vectorizer_aida_test_final_clean_names_train_shuffle_clean.csv")
#rdd_test = sc.textFile("filtering_results_results_filtering_frequency_aida_vectorizer_aida_test_final_clean_names.json")
#
#get_entity_ids_train = rdd_train.map(lambda x: getEntityIds(x)).collect()
#mention_ids_train = rdd_train.map(lambda x : json.loads(x)["mention_id"]).collect()
#
#get_entity_ids_test = rdd_test.map(lambda x: getEntityIds(x)).collect()
#mention_ids_test = rdd_test.map(lambda x : json.loads(x)["mention_id"]).collect()
#
#entity_ids_concerned = list(set(sum(get_entity_ids_train + get_entity_ids_test, [])))
#
#mention_ids_concerned = mention_ids_train + mention_ids_test
#mentionsTextFeatures = sc.textFile("all_v_mentionsJsonFileComplete-COnLL-UPDATE-NNIL.json").filter(lambda x : json.loads(x)["mention_id"] in mention_ids_concerned).map(lambda x: loadF(x)).collect()
#
#mentionIdToTextFeatures = {}
#for mf in mentionsTextFeatures:
#    mentionIdToTextFeatures[mf[0]] = mf[1]
#
#json.dump(mentionIdToTextFeatures, open("mentionIdToTextFeatures.json", "w"))
#
#id_to_neighbors = json.load(open("conll_id_to_neighbors.json", "r"))
#entity_ids_neighbors = sum([id_to_neighbors[eic] for eic in entity_ids_concerned], [])
#entity_ids_concerned = entity_ids_concerned + list(set(entity_ids_neighbors))
#print("Step 1")
#kbFileName = "aida_vectorizer_aida_test_final_clean_names.json"
#listIdTextFeatures = sc.textFile(kbFileName).map(lambda x: json.loads(x)).filter(lambda x : x["entity_id"] in entity_ids_concerned).map(lambda x: (x["entity_id"], x["features"])).collect()
#entityIdToTextFeatures = {}
#for litf in listIdTextFeatures:
#    entityIdToText[litf[0]] = litf[1]
#
#json.dump(entityIdToTextFeatures, open("entityIdToTextFeatures.json", "w"))
#assert(0)
#############################################################################################################################

#############################################################################################################################
#############################################################################################################################

#id_to_neighbors = json.load(open("conll_id_to_neighbors.json", "r"))
#id_to_neighbors = sc.broadcast(id_to_neighbors)
#dic_test = json.load(open("idsToTextTest.json", "r"))
#dic_train = json.load(open("idsToTextTrain.json", "r"))
#
#def merge_two_dicts(x, y):
#    z = x.copy()   # start with x's keys and values
#    z.update(y)    # modifies z with y's keys and values & returns None
#    return z
#
#entityIdToText = merge_two_dicts(dic_train, dic_test) 

#mentionIdToText = json.load(open("concat_mentionIdToText_conll.json", "r"))

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



#dic_line = {"mention_id"= "0", "mention_text_features": [0,1,3,4], "entity_ranks": {"E1": [name_score, ], "E10": [name_score, context_score]}}

with open("identification_example.json", "r") as f:
    dic_line = json.load(f)

mentionIdToText = {dic_line["mention_id"]: dic_line["mention_text_features"]}
x_dic = returnXDataLine(dic_line)

pdb.set_trace()
