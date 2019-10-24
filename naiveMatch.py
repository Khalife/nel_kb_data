import pdb
import json
import scipy.sparse as sp
import sys
import numpy as np


#aida_file = open("aida_test_final_clean_names.json", "r")


def getMatchingDicFromKB(kb_filename):
    nb_line = - 1
    non_unique_entity = 0
    
    mentionNameToGoldId = {}
    lookup_id_to_text = {}

    kb_file = open(kb_filename, "r")
    for line in kb_file:
        nb_line += 1
        dic_line = json.loads(line)
        dic_new = dict(dic_line)
        del dic_new["entity_text"]
        mentionNameToGoldId[dic_line["entity_name"]] = [dic_new]
        if nb_line % 100000 == 0:
            print(nb_line)

    return mentionNameToGoldId 


def updateDicFromTrain(file_train, matchingDic):
    print("Using training file")
    mentionsTrainFile = open(file_train, "r")
    
    nb_line = - 1
    non_unique_entity = 0
    supplementary_ids = []
    matchingDicNew = matchingDic
    for line in mentionsTrainFile:
        nb_line += 1
        dic_line = json.loads(line)
        dic_new = dict(dic_line)
        dic_new["entity_type"] = dic_new["gold_entity_type"]
        dic_new["entity_id"] = dic_new["gold_entity_id"]
        del dic_new["mention_full_text"]
        try:
            value_ids = [dic_gid["gold_entity_id"] for dic_gid in matchingDicNew[dic_line["mention_name"]]]
            target_id = dic_line["gold_entity_id"]
            if target_id not in value_ids:
                matchingDicNew[dic_line["mention_name"]].append(dic_new)
        except:
            matchingDicNew[dic_line["mention_name"]] = [dic_new]
        
        if nb_line == 10000:
            print(nb_line)
    
        if len(matchingDicNew[dic_line["mention_name"]]) > 1:
            non_unique_entity += 1
            for eid in matchingDicNew[dic_line["mention_name"]]:
                supplementary_ids.append(eid)
    
    return matchingDicNew



def naiveMatch(file_test, mentionNameToGoldId):
    missed_mentions = []
    dic_predictions = {}
    nb_catched_queries = 0
    entity_presence = 0
    missed_entity = 0   
 
    gold_match = {}
    mentionsTestFile = open(file_test, "r")
    for line in mentionsTestFile:
        
        dic_line = json.loads(line)
        mention_id = dic_line["mention_id"]
        gold_match[mention_id] = dic_line["gold_entity_id"]
        try:
            predictedDics = mentionNameToGoldId[dic_line["mention_name"]]
            predictedTypes = [pd["entity_type"] for pd in predictedDics] 
            predictedId = [pd["entity_id"] for pd in predictedDics]
            nb_catched_queries += 1
        except:
            predictedId = ["ENone"]
            missed_entity += 1
            predictedTypes = []
         
        mention_type = dic_line["gold_entity_type"]
        true_id = dic_line["gold_entity_id"]
    
        if mention_type in predictedTypes:
            final_predicted_index = predictedTypes.index(mention_type) 
            final_predicted_id = predictedId[final_predicted_index]
            entity_presence += 1
            dic_predictions[mention_id] = final_predicted_id 
        else:
            missed_mentions.append(dic_line) 

    return dic_predictions, missed_mentions, gold_match


if __name__ == "__main__":
    kb_file = sys.argv[1]  
    filename_test = sys.argv[2]
    matchingDic = getMatchingDicFromKB(kb_file)
    try:
        file_train = sys.argv[3]
        matchingDic = updateDicFromTrain(file_train, matchingDic) 
    except:
        print("No train file")

    dic_prediction, missed_mentions, gold_match  = naiveMatch(filename_test, matchingDic) 
   
    score_naive_match = 0
    for key in dic_prediction.keys():
        if dic_prediction[key] == gold_match[key]:
            score_naive_match += 1
   
    print("Total queries nb : " + str(len(dic_prediction) + len(missed_mentions))) 
    print("Remaining : " + str(len(missed_mentions)) + ", " + " Naive precision : " + str(score_naive_match/float(len(dic_prediction))))

    #writeFileTest = open("remaining_queries_" + filename_test.split("/")[-1], "w")
    #for mm in missed_mentions:
    #    writeFileTest.write(json.dumps(mm, ensure_ascii=False) + "\n")

    # python naiveMatch.py /home/khalife/ai-lab/data/COnLL/KB/aida_test_final_clean_names.json  /home/khalife/ai-lab/data/COnLL/KB/mentionsJsonFileComplete-COnLL-UPDATE-NNIL-testb-v1.json   
    pdb.set_trace()
    
