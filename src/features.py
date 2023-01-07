
from data import Sentence, Read, Write



class FeatureMap:
    
    def __init__(self, train_data:list) -> None:
        self.train_data = train_data #all_sentences
        self.feature_map = {} #{feature:index}
        
        self.index = 0 #index of features in feature_map

        for sentence in self.train_data:
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

                if token_idx == 5: #debug
                    break

                head_id = head_list[token_idx]
                dep_id = id_list[token_idx]

                if token_idx == 0: # ROOT
                    dform = dlemma = dpos = "_ROOT_"
                    hform = hlemma = hpos = "_NULL_" # Doesn't have a head
                    # Only visually at the beginning of the sentence
                    direction = distance = "_NULL_" 

                    hP1form = hP1lemma = hP1pos = "_NULL_"
                    hM1form = hM1lemma = hM1pos = "_NULL_"
                    dP1form = dP1lemma = dP1pos = "_NULL_"
                    dM1form = dM1lemma = dM1pos = "_NULL_"
                    
                else:
                    dform = form_list[token_idx]
                    dlemma = lemma_list[token_idx]
                    dpos = pos_list[token_idx]

                    hform, hlemma, hpos = self.get_head_attributes(
                        head_id, form_list, lemma_list, pos_list
                    )

                    direction, distance = self.get_direction_and_distance(
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
                    hform, hpos, dform, dpos, direction, distance,
                    hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos,
                    dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos
                ) #list


                for feature in features_one_arc:
                    if feature not in self.feature_map:
                        self.feature_map[feature] = self.index
                        self.index += 1


            break #debug
            
        print(self.feature_map)



                
    def get_features(self, hform, hpos, dform, dpos, direction, distance,
                    hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos,
                    P1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos) -> list:

        # Unigram templates T1-T6
        # Bigram templates T7 - T13
        # Other templates (from McDonald's et al. 2005) T14-T17
        
        templates = {

            'T1' : f"{hform},{direction},{distance}",
            'T2' : f"{hpos},{direction},{distance}",
            'T3' : f"{hform},{hpos},{direction},{distance}",
            'T4' : f"{dform},{direction},{distance}",
            'T5' : f"{dpos},{direction},{distance}",
            'T6' : f"{dform},{dpos},{direction},{distance}",

            'T7' : f"{hform},{dform},{direction},{distance}",
            'T8' : f"{hpos},{dpos},{direction},{distance}",
            'T9' : f"{hform},{hpos},{dpos},{direction},{distance}",
            'T10' : f"{hform},{hpos},{dform},{direction},{distance}",
            'T11' : f"{hform},{dform},{dpos},{direction},{distance}",
            'T12' : f"{hpos},{dform},{dpos},{direction},{distance}",
            'T13' : f"{hform},{hpos},{dform},{dpos},{direction},{distance}",

            'T14' : f"{hpos},{dpos},{hP1pos},{dM1pos},{direction},{distance}",
            'T15' : f"{hpos},{dpos},{hM1pos},{dM1pos},{direction},{distance}",
            'T16' : f"{hpos},{dpos},{hP1pos},{dP1pos},{direction},{distance}",
            'T17' : f"{hpos},{dpos},{hM1pos},{dP1pos},{direction},{distance}"
        
        }


        features_one_arc = list(templates.values())

        return features_one_arc








    def get_head_attributes(self, head_id:str, 
        form_list:list, lemma_list:list, pos_list:list) -> tuple:
        # get attributes of token with id == head_id
        
        head_idx = int(head_id)

        form = form_list[head_idx]
        lemma = lemma_list[head_idx]
        pos = pos_list[head_idx]

        return form, lemma, pos

    

    def get_direction_and_distance(self, head_id:str, dep_id:str) -> tuple:
        
        head_id = int(head_id)
        dep_id = int(dep_id)

        if dep_id < head_id:
            direction = "left"
            distance = head_id - dep_id

        elif dep_id > head_id:
            direction = "right"
            distance = dep_id - head_id
        
        else:
            print("Something wrong")
        
        return direction, distance



    def get_neighbours_attributes(self, head_id:str, dep_id:str,
        form_list:list, lemma_list:list, pos_list:list) -> list:

        # get attributes of tokens with ids:
        # head_id+1, head_id-1, dep_id+1, dep_id-1
        
        head_idx = int(head_id)
        dep_idx = int(dep_id)
        
        head_idx_P1 = head_idx +1
        head_idx_M1 = head_idx -1
        dep_idx_P1 = dep_idx +1
        dep_idx_M1 = dep_idx -1

        hP1form = form_list[head_idx_P1]
        hP1lemma = lemma_list[head_idx_P1]
        hP1pos = pos_list[head_idx_P1]

        hM1form = form_list[head_idx_M1]
        hM1lemma = lemma_list[head_idx_M1]
        hM1pos = pos_list[head_idx_M1]

        dP1form = form_list[dep_idx_P1]
        dP1lemma = lemma_list[dep_idx_P1]
        dP1pos = pos_list[dep_idx_P1]

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





if __name__ == "__main__":

    file_name = "wsj_train.first-1k.conll06"
    language = "english"
    mode = "train"

    reader = Read(file_name, language, mode)

    all_sentences = reader.all_sentences

    #print(type(all_sentences)) # <class 'list'> 
    #print(type(all_sentences[0])) # <class 'data.Sentence'>

    FeatureMap(all_sentences)
