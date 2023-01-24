# input: feature map + sentence
# possible arcs 
# 
# output: feature vectors

import pprint

from data import Sentence, Read, Write
from features import Features




def get_fv_one_arc(feat_map, features_one_arc):

    fv = [0 for i in range(len(feat_map))]
    fv_dense = []

    for feature in features_one_arc:
        feat_idx = feat_map[feature]
        
        fv_dense.append(feat_idx)
        fv[feat_idx] = 1

    assert len(fv) == len(feat_map)

    return fv, fv_dense





if __name__ == "__main__":
    
    file_name_train = "wsj_train.first-1k.conll06"
    language = "english"
    mode = "train"

    reader_train = Read(file_name_train, language, mode)
    sent_train = reader_train.all_sentences[:5]

    feat = Features(sent_train)
    feat_map = feat.create_feature_map(train_data=sent_train)

    file_name_test = "wsj_test.conll06.blind"
    language = "english"
    mode = "test"

    reader_test = Read(file_name_test, language, mode)
    sent_test = reader_train.all_sentences[:5]


    index = 0
    for sentence in sent_train:
        #sentence is Sentence object

        '''Sentence attributes'''
        id_list = sentence.id
        form_list = sentence.form
        lemma_list = sentence.lemma
        pos_list = sentence.pos
        #xpos_list = sentence.xpos #always empty?
        #morph_list = sentence.morph #always empty?
        head_list = sentence.head
        rel_list = sentence.rel
        #empty1_list = sentence.empty1 #always empty?
        #empty2_list = sentence.empty2 #always empty?

        # 1	In	in	IN	_	_	43	ADV	_	_
        # 2	an	an	DT	_	_	5	NMOD	_	_


        for token_idx in range(len(id_list)):

            #if token_idx == 10: #debug
            #    break



            head_id = head_list[token_idx]
            dep_id = id_list[token_idx]

            if token_idx == 0: # ROOT
                dform = dlemma = dpos = "_ROOT_"
                hform = hlemma = hpos = "_NULL_" # Doesn't have a head

                # No head => no direction, no distance, no tokens in-between
                direction = distance = "_NULL_"
                between = [None]

                hP1form = hP1lemma = hP1pos = "_NULL_"
                hM1form = hM1lemma = hM1pos = "_NULL_"
                dP1form = dP1lemma = dP1pos = "_NULL_" # token at idx 1?
                dM1form = dM1lemma = dM1pos = "_NULL_" # BOS?
                
            else:
                dform = form_list[token_idx]
                dlemma = lemma_list[token_idx]
                dpos = pos_list[token_idx]

                # get attributes of the head
                hform, hlemma, hpos = feat.get_attributes(
                    head_id, form_list, lemma_list, pos_list
                )

                if hform == "ROOT":
                    hform = hlemma = hpos = "_ROOT_" 

                # direction of arc, distance between head and dep,
                # list of tokens (ids) in between head and dep
                direction, distance, between = feat.get_direction_distance_between(
                    head_id, dep_id
                )

                # get attributes for left/right tokens
                neighbours_attributes = feat.get_neighbours_attributes(
                    head_id, dep_id, form_list, lemma_list, pos_list
                )
            
                # unpack
                hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos = neighbours_attributes[:6]
                dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos = neighbours_attributes[6:]

            
            features_one_arc = feat.get_features(
                form_list, lemma_list, pos_list,
                hform, hlemma, hpos, 
                dform, dlemma, dpos, 
                direction, distance, between,
                hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos,
                dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos
            ) #list

            
            fv, fv_dense = get_fv_one_arc(feat_map, features_one_arc)

            #print(fv_one_arc)
            #print(fv_one_arc_dense)

            
            break
        
        break

        