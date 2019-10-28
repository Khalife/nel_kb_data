import json
import pdb
import re
import math

rx = re.compile(r"[\W]")

index = 0
vocabulary = set([])
nb_document_containing_word = {}
entityIdToTextFeatures = {}
idToName = {}
with open("aida_micro_kb.json", "r") as f:
    for line in f:
        dic_line = json.loads(line)
        entity_text = dic_line["entity_text"]
        text_features = {}
        entity_text_split = entity_text.split()
        n_words_entity = float(len(entity_text_split))
        for word in entity_text_split:
            try:
                text_features[word] += 1
            except:
                text_features[word] = 1

        for word in text_features.keys():
            text_features[word] /= n_words_entity

        entityIdToTextFeatures[dic_line["entity_id"]] = text_features

        entity_text_set = set(entity_text.split())
        vocabulary.update(entity_text_set)
        for word in entity_text_set:
            try:
                nb_document_containing_word[word] + 1
            except:
                nb_document_containing_word[word] = 1
        
        idToName[dic_line["entity_id"]] = dic_line["entity_name"]

        if index % 1000 == 0:
            print(index)
        index += 1


idf_diag = {}
for word in nb_document_containing_word.keys():
    ratio = index/nb_document_containing_word[word]
    idf_diag[word] = math.log10(ratio)

with open("entityIdToName_micro_kb.json", "w") as f:
    json.dump(idToName, f)

with open("idf_diag.json", "w") as f:
    json.dump(idf_diag, f)


with open("entityIdToTextFeatures.json", "w") as f:
    json.dump(entityIdToTextFeatures, f)
        
vocabulary = sorted(list(vocabulary))
vocabulary_map = {}
word_index = 0
for word in vocabulary:
    vocabulary_map[word] = word_index
    word_index += 1


with open("vocabulary_kb.json", "w") as f:
    json.dump(vocabulary_map, f)


reverse_dictionnary = {}
idToType = {}
ngramsToIds = {}
index = 0
with open("aida_micro_kb.json", "r") as f:
    for line in f:
        dic_line = json.loads(line)
        entity_text = dic_line["entity_text"].split()
        for word in entity_text:
            try:
                reverse_dictionnary[vocabulary_map[word]].update([dic_line["entity_id"]])
            except:
                reverse_dictionnary[vocabulary_map[word]] = set([dic_line["entity_id"]])

        entity_id = dic_line["entity_id"]
        entity_type = dic_line["entity_type"]
        idToType[entity_id] = entity_type
        entity_name = dic_line["entity_name"].lower()


        clean_entity_name = re.sub(rx, " ", entity_name).replace("_", " ").split()
        for word_entity_name in clean_entity_name:
            if len(word_entity_name) > 1:
                for n in [2,3,4]:
                    for i in range(len(word_entity_name)-n+1):
                        try:
                            #ngramsToIds[word_entity_name[i:(i+n)]].append(entity_id)
                            ngram_to_ids = ngramsToIds[word_entity_name[i:(i+n)]]
                            try:
                                ngramsToIds[word_entity_name[i:(i+n)]][entity_id] += 1
                            except:
                                ngramsToIds[word_entity_name[i:(i+n)]][entity_id] = 1

                        except:
                            ngramsToIds[word_entity_name[i:(i+n)]] = {entity_id: 1}
                            #ngramsToIds[word_entity_name[i:(i+n)]] = [entity_id]

        if index % 1000 == 0:
            print(index)
        index += 1

with open("wordToIds.json", "w") as f:
    for key in reverse_dictionnary.keys():
        reverse_dictionnary[key] = list(reverse_dictionnary[key])
    json.dump(reverse_dictionnary, f)

with open("ngramsToIds.json", "w") as f:
    json.dump(ngramsToIds, f)

with open("entityIdToOntology_micro_kb.json", "w") as f:
    json.dump(idToType, f)



pdb.set_trace()



