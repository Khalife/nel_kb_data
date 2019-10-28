import json 
import time
import numpy as np
import pdb
#import naiveMatch
#import bm25 
import sys
import string
import unicodedata
from math import log
import re
import naiveMatch
rx = re.compile(r"[\W]")


def unicodeName(name):
    #name_clean = unicodedata.normalize('NFD', name)
    #name_clean = name_clean.encode('ascii', 'ignore')
    #return name_clean
    return name

import re
rx = re.compile(r"[\W]")
def processFullText(str1):
    mention_context_text = re.sub(rx, " ", str1.lower())
    words = mention_context_text.split()
    bag_of_words = [word for word in words if (word not in stop_words) and (word != "")]
    return " ".join(bag_of_words)

def acronymTest(str1, main_type):
    distances = []
    nb_letters = len(str1)
    i = 0
    index = -1
    while i < nb_letters:
        if index < 0:
            if str1[i].isupper():
                index += 1
        else:
            if str1[i].isupper():
                distances.append(i-index)
                index = i
        i += 1

    if main_type == "PER":
        max_distances = 4

    if main_type == "ORG":
        max_distances = 5

    if main_type == "GPE":
        max_distances = 4

    if main_type == "UKN":
        max_distances = 4

    return distances, (len(distances) > 0) and ( sum(distances) <= len(distances) ) and ( len(distances) < max_distances ) #and ( len(distances) <= 5 )


def returnLetterNgram1(str1,n):
    lthree_grams1 = []
    str2 = str1.replace(" ", "")
    for i in range(max([len(str2)-n+1,1])):
        lthree_grams1.append(str2[i:i+n])
    return lthree_grams1

def scoreLetterNgram(str1, str2, n):
    LNgram1 = returnLetterNgram1(str1,n)
    LNgram2 = returnLetterNgram1(str2,n)
    L3 = set(LNgram1).intersection(LNgram2)
    L4 = set(LNgram1).union(set(LNgram2))
    return float(len(L3))/len(L4)

def acronymScore1(str1, str2):
    capital_letters = "".join([l for l in str2 if l.isupper()])
    first_words = str2.split()
    first_letters = "".join([fw[0] for fw in first_words])
    norm_nb_letters = min([len(str1), len(str2)])

    total_letters = capital_letters
    for letter in first_letters:
        if letter not in capital_letters:
            total_letters = total_letters + letter

    lcs1 = longest_common_substring(str1.lower(),total_letters.lower())
    acronym_score = lcs1/norm_nb_letters
    return acronym_score

def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return longest


def reverse_insort_dic(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    
    if lo < 0 or hi <0:
        raise ValueError('lo and hi must be non-negative')
    
    while lo < hi:
        mid = (lo+hi)//2
        if x["score"] > a[mid]["score"]: hi = mid
        else: lo = mid+1
    original_length = len(a)
    
    if lo == len(a):
        a.append(x)
    else:
        a[lo+1:] = a[lo:len(a)]
        a[lo] = x
    return a


# need to define:
#                   - ngrams to entity id and freq
#                   - word context to entity id and freq


with open("entityIdToTextFeatures.json", "r") as f:
    entityIdToTextFeatures = json.load(f)

with open("vocabulary_kb.json", "r") as f:
    vocabulary_kb = json.load(f)

with open("wordToIds.json", "r") as f:
    wordToIds = json.load(f)

with open("ngramsToIds.json", "r") as f:
    ngramsToIds = json.load(f)

with open("entityIdToOntology_micro_kb.json", "r") as f:
    idToType = json.load(f)

with open("idf_diag.json", "r") as f:
    idf_diag = json.load(f)

with open("entityIdToName_micro_kb.json", "r") as f:
    idToName = json.load(f)

with open("matchingDict.json", "r") as f:
    matchingDict = json.load(f)

def treatQueryFromJson(query, matchingDict):
    start = time.time()
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0
    time_0cs = 0
    time_1cs = 0
    time_2cs = 0
    time_3cs = 0
    
    result_naive = naiveMatch.naiveMatchQuery(query, matchingDict) 
    if [val for val in result_naive.values()][0] != "ENone":
        return result_naive

    query_type = query["gold_entity_type"]
    query_text = query["query_text"]
    query_features = {}
    query_text_split = query_text.split()
    n_words_query = float(len(query_text_split))
    for word in query_text_split:
        try:
            query_word = vocabulary_kb[word]
        except:
            continue

        try:
            query_features[query_word] += 1
        except:
            query_features[query_word] = 1

    for key in query_features.keys():
        query_features[key] /= n_words_query

    # wordToIds: dict from word_index to [entity_id, freq_word]
    context_scores = {}
    for word in query_text.split():
        try:
            word_index = vocabulary_kb[word]
        except:
            continue
        try:
            potential_entities = [eid for eid in wordToIds[str(word_index)] if idToType[eid] == query_type]
        except:
            pdb.set_trace()

        for pe in potential_entities:
            for pew in entityIdToTextFeatures[pe].keys():
                try:
                    context_scores[pe] += query_features[word_index]*entityIdToTextFeatures[pe][pew]*(idf_diag[word]**2)
                except:
                    context_scores[pe] = query_features[word_index]*entityIdToTextFeatures[pe][pew]*(idf_diag[word]**2)
    

    #######################################
    ############## Name score ##############
    #######################################
    mention_name = query["mention_name"]
    total_gram_scores = {}
    _, acronym_test = acronymTest(mention_name, "PER")
    if acronym_test:
        potential_entities = typeToIds[query_type]
        for le in potential_entities:
            entity_name = idToName[le]
            closest_clean_entity_name = entity_name
            closest_clean_entity_name = re.sub(rx, " ", closest_clean_entity_name)
            closest_clean_mention_name = re.sub(rx, "", query_name)
            name_score = acronymScore1(closest_clean_mention_name, closest_clean_entity_name)
            total_gram_scores[le] = name_score
    else:
        two_grams = []
        three_grams = []
        four_grams = []
        query_name = query["mention_name"].lower()
        query_name =  query_name.replace("_", " ")
        query_name = re.sub(rx, " ", query_name)

                                                                
        for query_name_word in query_name.split():
            for i in range(len(query_name_word)-1):
                two_grams.append(query_name_word[i:(i+2)])
                                                                    
            for i in range(len(query_name_word)-2):
                three_grams.append(query_name_word[i:(i+3)])
                                                                    
            for i in range(len(query_name_word)-3):
                four_grams.append(query_name_word[i:(i+4)])

        
        for n, n_grams in enumerate([two_grams, three_grams, four_grams]):
            n = n + 2
            gram_scores = {}
            for grams in n_grams:
                # we want the number of times grams appear for each candidate entities 
                gram_scores = {eid: ngramsToIds[grams][eid] for eid in ngramsToIds[grams].keys() if idToType[eid] == query_type}
                #pdb.set_trace()

                #urls = [{"url": key, "nbr": value} for key, value in Counter(list_of_urls).items()]
                #for pe in potential_entities:
                #    ggram_scores.get(pe[0], 0) + 1
                #    try:
                #        gram_scores[pe[0]] += pe[1]
                #    except:
                #        gram_scores[pe[0]] = pe[1]

                for key in gram_scores.keys():
                    eid_name = idToName[key].lower()
                    size_union = len(set(n_grams + [eid_name[i:(i+n)] for i in range(len(eid_name)-n+1)]))
                    gram_scores[key] = gram_scores[key]/float(size_union)
                    try:
                        total_gram_scores[key] += gram_scores[key]
                    except:
                        total_gram_scores[key] = gram_scores[key]
         
        for key in total_gram_scores.keys():
            total_gram_scores[key] = total_gram_scores[key]/3.

    lambda_score = 0.9
    #pdb.set_trace()
    total_candidates = reduce(lambda x, y: x.union(y.keys()), [total_gram_scores, context_scores], set())
    results = []
    i = 1
    previous_score = -100
    for tce in total_candidates:
        name_score = total_gram_scores.get(tce, 0.)
        context_score = context_scores.get(tce, 0.)
        final_score = lambda_score*name_score + (1. - lambda_score)*context_score
        tce_score = {"entity_name": idToName[tce], "entity_id": tce, "name_score": name_score, "context_score": context_score, "score": final_score}
        #current_score = results[i-1][3]

        
        reverse_insort_dic(results, tce_score)
    end = time.time()
    print("TIME: " + str(end - start) +  " sec.")
    return results


result = treatQueryFromJson({"mention_id": 0, "mention_name":"United state", "query_text": "this country is blabla", "gold_entity_type": "Country"}, matchingDict)


pdb.set_trace()
