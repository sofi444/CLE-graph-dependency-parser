
import pprint

from data import Sentence, Read, Write



class Features:

    def __init__(self, data:list) -> None:
        
        self.data = data #all_sentences
        self.feature_map = {} #{feature:index}



    def create_feature_map(self, train_data):
        '''
        Extract features from TRAINING data
        Create feature map
        '''

        fm_index = 0 # index of features in feature_map

        for sentence in train_data:

            id_list, form_list, lemma_list, pos_list, head_list = self.get_sentence_attributes(
                sentence_ob=sentence
            )

            for token_idx in range(len(id_list)):

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
                    hform, hlemma, hpos = self.get_attributes(
                        head_id, form_list, lemma_list, pos_list
                    )

                    if hform == "ROOT":
                        hform = hlemma = hpos = "_ROOT_" 

                    # direction of arc, distance between head and dep,
                    # list of tokens (ids) in between head and dep
                    direction, distance, between = self.get_direction_distance_between(
                        head_id, dep_id
                    )

                    # get attributes for left/right tokens
                    neighbours_attributes = self.get_neighbours_attributes(
                        head_id, dep_id, form_list, lemma_list, pos_list
                    )
                
                    # unpack
                    hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos = neighbours_attributes[:6]
                    dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos = neighbours_attributes[6:]

                
                features_one_arc = self.get_features(
                    form_list, lemma_list, pos_list,
                    hform, hlemma, hpos, 
                    dform, dlemma, dpos, 
                    direction, distance, between,
                    hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos,
                    dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos
                ) #list


                # Add to feature_map
                for feature in features_one_arc:
                    if feature not in self.feature_map:
                        self.feature_map[feature] = fm_index
                        fm_index += 1

            break #debug

            
        #print("Feature map:")
        #pprint.pprint(self.feature_map)
        #print("Number of features:", fm_index)


                
    def get_features(self, 
                    form_list, lemma_list, pos_list,
                    hform, hlemma, hpos, 
                    dform, dlemma, dpos, 
                    direction, distance, between,
                    hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos,
                    dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos) -> list:

        # Templates from McDonald's et al. (2005):
            # Unigram T1-T6
            # Bigram T7-T13
            # -1/+1 tokens T14-T17
            # In between tokens T52
        
        # Other templates:
            # T18-T51
            # In between tokens T53
        
        templates = {

            'T1' : f"{hform},{direction},{distance}",
            'T2' : f"{hpos},{direction},{distance}",
            #'T3' : f"{dform},{direction},{distance}",
            #'T4' : f"{dpos},{direction},{distance}",
            #'T5' : f"{hform},{hpos},{direction},{distance}",
            #'T6' : f"{dform},{dpos},{direction},{distance}",

            #'T7' : f"{hform},{dform},{direction},{distance}",
            #'T8' : f"{hpos},{dpos},{direction},{distance}",
            #'T9' : f"{hform},{hpos},{dpos},{direction},{distance}",
            #'T10' : f"{hform},{hpos},{dform},{direction},{distance}",
            #'T11' : f"{hform},{dform},{dpos},{direction},{distance}",
            #'T12' : f"{hpos},{dform},{dpos},{direction},{distance}",
            #'T13' : f"{hform},{hpos},{dform},{dpos},{direction},{distance}",

            #'T14' : f"{hpos},{dpos},{hP1pos},{dM1pos},{direction},{distance}",
            #'T15' : f"{hpos},{dpos},{hM1pos},{dM1pos},{direction},{distance}",
            #'T16' : f"{hpos},{dpos},{hP1pos},{dP1pos},{direction},{distance}",
            #'T17' : f"{hpos},{dpos},{hM1pos},{dP1pos},{direction},{distance}",

            #'T18' : f"{hlemma},{direction},{distance}",
            #'T19' : f"{dlemma},{direction},{distance}",

            #'T20' : f"{hform},{dpos},{direction},{distance}",
            #'T21' : f"{hpos},{dform},{direction},{distance}",

            #'T22' : f"{hlemma},{dpos},{direction},{distance}",
            #'T23' : f"{hpos},{dlemma},{direction},{distance}",
            #'T24' : f"{hlemma},{hpos},{direction},{distance}",
            #'T25' : f"{dlemma},{dpos},{direction},{distance}",
            #'T26' : f"{hlemma},{dlemma},{direction},{distance}",
            
            #'T27' : f"{hlemma},{hpos},{dlemma},{dpos},{direction},{distance}",
            #'T28' : f"{hpos},{dlemma},{dpos},{direction},{distance}",
            #'T29' : f"{hlemma},{dlemma},{dpos},{direction},{distance}",
            #'T30' : f"{hlemma},{hpos},{dlemma},{direction},{distance}",
            #'T31' : f"{hlemma},{hpos},{dpos},{direction},{distance}",

            #'T32' : f"{hP1form},{direction},{distance}",
            #'T33' : f"{hform},{hP1form},{direction},{distance}",
            #'T34' : f"{hP1pos},{direction},{distance}",
            #'T35' : f"{hpos},{hP1pos},{direction},{distance}",

            #'T36' : f"{hM1form},{direction},{distance}",
            #'T37' : f"{hM1form},{hform},{direction},{distance}",
            #'T38' : f"{hM1pos},{direction},{distance}",
            #'T39' : f"{hM1pos},{hpos},{direction},{distance}",

            #'T40' : f"{dP1form},{direction},{distance}",
            #'T41' : f"{dform},{dP1form},{direction},{distance}",
            #'T42' : f"{dP1pos},{direction},{distance}",
            #'T43' : f"{dpos},{dP1pos},{direction},{distance}",

            #'T44' : f"{dM1form},{direction},{distance}",
            #'T45' : f"{dM1form},{dform},{direction},{distance}",
            #'T46' : f"{dM1pos},{direction},{distance}",
            #'T47' : f"{dM1pos},{dpos},{direction},{distance}",

            #'T48' : f"{hM1form},{hform},{hP1form},{direction},{distance}",
            #'T49' : f"{hM1pos},{hpos},{hP1pos},{direction},{distance}",
            #'T50' : f"{dM1form},{dform},{dP1form},{direction},{distance}",
            #'T51' : f"{dM1pos},{dpos},{dP1pos},{direction},{distance}"

        }


        features_one_arc = [f"{t}:{f}" for t,f in templates.items()]
        

        for token_id in between:

            if token_id is None:
                bform = blemma = bpos = "_NONE_"
            
            else:
                bform, blemma, bpos = self.get_attributes(
                    token_id, form_list, lemma_list, pos_list
                )
            

            between_templates = {
                
                #'T52' : f"{hpos},{bpos},{dpos},{direction},{distance}"

                #'T53' : f"{hform},{bform},{dform},{direction},{distance}"

            }

            features_one_arc.extend([f"{t}:{f}" for t,f in between_templates.items()])


        return features_one_arc



    def get_attributes(self, token_id:str, 
        form_list:list, lemma_list:list, pos_list:list) -> tuple:
        # get attributes of token with id == token_id
        
        token_idx = int(token_id)

        form = form_list[token_idx]
        lemma = lemma_list[token_idx]
        pos = pos_list[token_idx]

        return form, lemma, pos

    

    def get_direction_distance_between(self, head_id:str, dep_id:str) -> tuple:
        
        head_id = int(head_id)
        dep_id = int(dep_id)

        if dep_id < head_id:
            direction = "left"
            distance = head_id - dep_id
            tokens_in_between = [str(i) for i in range(dep_id+1, head_id)]

        elif dep_id > head_id:
            direction = "right"
            distance = dep_id - head_id
            tokens_in_between = [str(i) for i in range(head_id+1, dep_id)]
        
        if distance > 10:
            distance = "10+"

        if tokens_in_between == []:
            tokens_in_between = [None]

        
        return direction, str(distance), tokens_in_between



    def get_neighbours_attributes(self, head_id:str, dep_id:str,
        form_list:list, lemma_list:list, pos_list:list) -> list:

        # get attributes of tokens with ids:
        # token_id+1, token_id-1, dep_id+1, dep_id-1
        
        head_idx = int(head_id)
        dep_idx = int(dep_id)
        
        head_idx_P1 = head_idx +1
        head_idx_M1 = head_idx -1
        dep_idx_P1 = dep_idx +1
        dep_idx_M1 = dep_idx -1

        if head_idx_P1 < len(form_list):
            hP1form = form_list[head_idx_P1]
            hP1lemma = lemma_list[head_idx_P1]
            hP1pos = pos_list[head_idx_P1]
        else:
            hP1form = hP1lemma = hP1pos = "_EOS_"

        hM1form = form_list[head_idx_M1]
        hM1lemma = lemma_list[head_idx_M1]
        hM1pos = pos_list[head_idx_M1]

        if dep_idx_P1 < len(form_list):
            dP1form = form_list[dep_idx_P1]
            dP1lemma = lemma_list[dep_idx_P1]
            dP1pos = pos_list[dep_idx_P1]
        else:
            dP1form = dP1lemma = dP1pos = "_EOS_"

        dM1form = form_list[dep_idx_M1]
        dM1lemma = lemma_list[dep_idx_M1]
        dM1pos = pos_list[dep_idx_M1]

        neighbours_attributes = [
            hP1form, hP1lemma, hP1pos, 
            hM1form, hM1lemma, hM1pos, 
            dP1form, dP1lemma, dP1pos, 
            dM1form, dM1lemma, dM1pos
        ]

        neighbours_attributes = [
            "_NULL_" if i == '_' else i for i in neighbours_attributes
        ]


        return neighbours_attributes



    def get_sentence_attributes(self, sentence_ob:object) -> tuple:
        
        id_list = sentence_ob.id
        form_list = sentence_ob.form
        lemma_list = sentence_ob.lemma
        pos_list = sentence_ob.pos
        #xpos_list = sentence_ob.xpos #always empty?
        #morph_list = sentence_ob.morph #always empty?
        head_list = sentence_ob.head
        #rel_list = sentence_ob.rel
        #empty1_list = sentence_ob.empty1 #always empty?
        #empty2_list = sentence_ob.empty2 #always empty?

        # 1	In	in	IN	_	_	43	ADV	_	_
        # 2	an	an	DT	_	_	5	NMOD	_	_

        return id_list, form_list, lemma_list, pos_list, head_list



    def get_fv_one_arc(self, feat_map, features_one_arc):
        # features_one_arc is the output of get_features

        fv = [0 for i in range(len(feat_map))]
        fv_dense = []

        for feature in features_one_arc:
            feat_idx = feat_map[feature]
            
            fv_dense.append(feat_idx)
            fv[feat_idx] = 1

        assert len(fv) == len(feat_map)

        return fv, fv_dense






if __name__ == "__main__":

    file_name = "wsj_train.first-1k.conll06"
    language = "english"
    mode = "train"

    reader = Read(file_name, language, mode)

    all_sentences = reader.all_sentences

    #print(type(all_sentences)) # <class 'list'> 
    #print(type(all_sentences[0])) # <class 'data.Sentence'>

    feat = Features(data=all_sentences)
    feat.create_feature_map(train_data=all_sentences)
