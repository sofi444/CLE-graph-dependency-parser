import sys
import collections
import pprint
import random
import numpy as np

from data import Read
from features import Features



class Graph:
    def __init__(self, sentence:object, 
                    feature_map:dict, 
                    weight_vector:list, 
                    graph_type:str) -> None:
        
        global F
        F = Features() # for accessing functions

        self.sentence = sentence

        self.feature_map = feature_map
        self.w = weight_vector
        self.graph_type = graph_type

        self.nodes = self.sentence.id
        self.graph = {}

        # shared across arcs
        self.sent_attributes = F.get_sentence_attributes(self.sentence)

        
        '''
        GRAPH:dict where
        keys are heads
        values are dicts where
            keys are dependents
            values are dicts where
                keys are fv, score
                values are list(fv), float(score)
        '''
        

        if self.graph_type == "fully_connected":

            for node in self.nodes:

                dependents = {} #{dep_node: {fv: [], score: float()}}

                for dep_node in self.nodes:
                    if not (dep_node == '0' or dep_node == node): # impossible edges
                        
                        arc_fv = self.get_fv(self.feature_map, self.sent_attributes, node, dep_node)
                        arc_score = self.get_score(self.w, arc_fv)
                        
                        arc_attributes = {
                            'fv': arc_fv,
                            'score': arc_score
                        }

                        dependents[dep_node] = arc_attributes
                
                self.graph[node] = dependents
                

        
        if self.graph_type == "gold":

            heads = self.sentence.head
            for node in self.nodes:
                node_idx = self.nodes.index(node)
                head = heads[node_idx]
                if head == '_': # no edge [head -> node]
                    continue
                else: # there is an edge from [head -> node]
                    dep = node

                    # extract features
                    fv = self.get_fv(
                        self.feature_map,
                        self.sent_attributes,
                        head_id=head, dep_id=dep
                    )

                    arc_attributes = {
                            'fv':fv, 
                            'score': 1.0        
                    }

                    if head not in self.graph:
                        self.graph[head] = {
                            dep: arc_attributes
                        }

                    else:
                        self.graph[head][dep] = arc_attributes

                

        

    def reverse_graph(self, graph) -> dict:

        '''
        Reverse the graph so that
            graph[head][dep] == rev_graph[dep][head]
        
        REV_GRAPH:dict where
        keys are dependents
        values are dicts where
            keys are heads
            values are dicts where
                keys are fv, score
                values are list(fv), float(score)
        '''

        rev_graph = {}
        
        for head in graph.keys():
            for dep, arc_attributes in graph[head].items():
                if dep not in rev_graph:
                    rev_graph[dep] = {head: arc_attributes}
                else:
                    rev_graph[dep][head] = arc_attributes
        
        return rev_graph



    def find_max_heads_simple(self, rev_graph:dict) -> dict:

        '''
        For each dependent, find candidate head with max score

        MAX_HEADS is a dict where
        keys are dependents
        values are the head with max score
        '''

        max_heads = {}

        for dep, candidate_heads in rev_graph.items():
        
            tmp_dict = {}
            for head, arc_attributes in candidate_heads.items():
                head = head
                score = arc_attributes['score']
                tmp_dict[head] = score

            max_head = max(
                tmp_dict,
                key=lambda key: tmp_dict[key]
                )
            
            max_heads[dep] = max_head # no score, look up score from original graph
            #max_heads[dep] = (max_head, candidate_heads[max_head]) # with score
        
        return max_heads



    def find_cycle_on_max_heads(self, graph:dict) -> list | None:
        for dep in graph.keys():
            
            path = []
            current = dep

            while current not in path:
                
                if current not in graph.keys():
                    break #dead end

                path.append(current)
                current = graph[current]

            else:
                cycle = path[path.index(current):]

                # Return first cycle if found else None
                # Reverse because input graph is reversed
                return list(reversed(cycle))



    def get_fv(self, feature_map:dict, sent_attributes:list, head_id:str, dep_id:str):

        id_list = sent_attributes[0]
        form_list = sent_attributes[1]
        lemma_list = sent_attributes[2]
        pos_list = sent_attributes[3]
        head_list = sent_attributes[4]
        
        dform = form_list[int(dep_id)]
        dlemma = lemma_list[int(dep_id)]
        dpos = pos_list[int(dep_id)]

        # get attributes of the head
        hform, hlemma, hpos = F.get_attributes(
            head_id, form_list, lemma_list, pos_list
        )

        if hform == "ROOT":
            hform = hlemma = hpos = "_ROOT_"

        # direction of arc, distance between head and dep,
        # list of tokens (ids) in between head and dep
        direction, distance, between = F.get_direction_distance_between(
            head_id, dep_id
        )

        # get attributes for left/right tokens
        neighbours_attributes = F.get_neighbours_attributes(
            head_id, dep_id, form_list, lemma_list, pos_list
        )
        
        # unpack
        hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos = neighbours_attributes[:6]
        dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos = neighbours_attributes[6:]


        features_one_arc = F.get_features(
            form_list, lemma_list, pos_list,
            hform, hlemma, hpos, 
            dform, dlemma, dpos, 
            direction, distance, between,
            hP1form, hP1lemma, hP1pos, hM1form, hM1lemma, hM1pos,
            dP1form, dP1lemma, dP1pos, dM1form, dM1lemma, dM1pos
        ) #list


        fv_dense = F.features_to_vector(
            feature_map,
            features_one_arc
        )

        
        return fv_dense



    def get_score(self, weight_vector:list, fv:list, fv_type="dense") -> float:

        # calculate score for one arc
        arc_score = float()

        if fv_type == "dense":
            for feature in fv:
                feat_idx = int(feature)
                arc_score += weight_vector[feat_idx]
        
        elif fv_type == "sparse":
            assert len(weight_vector) == len(fv)

            arc_score += np.dot(fv, weight_vector)

        return arc_score






if __name__ == "__main__":

    test_reader = Read(file_name="wsj_dev.conll06.blind",
                        language="english",
                        mode="dev")
    
    train_reader = Read(file_name="wsj_train.first-1k.conll06",
                        language="english",
                        mode="train") 

    test_sent1 = test_reader.all_sentences[23] #Sentence object
    test_sent2 = test_reader.all_sentences[54] #longer

    train_data = train_reader.all_sentences[:2]

    F = Features()
    feature_map = F.create_feature_map(train_data=train_data)


    random.seed(7)
    tmp_w = [random.uniform(0,20) for _ in len(feature_map)]


    gold_ob = Graph(sentence=train_data[1],
                    feature_map=feature_map,
                    weight_vector=tmp_w,
                    graph_type="gold")

    gold_graph = gold_ob.graph

    #pprint.pprint(gold_graph)
    #pprint.pprint(gold_ob.reverse_graph(gold_graph))


    fully_connected_ob = Graph(sentence=train_data[1],
                                feature_map=feature_map,
                                weight_vector=tmp_w,
                                graph_type="fully_connected")

    fully_connected_graph = fully_connected_ob.graph
    rev_graph = fully_connected_ob.reverse_graph(graph=fully_connected_graph)

    pprint.pprint(fully_connected_graph)
    #pprint.pprint(rev_graph)

    max_heads = fully_connected_ob.find_max_heads_simple(rev_graph=rev_graph)

    cycle = fully_connected_ob.find_cycle_on_max_heads(graph=max_heads)

    #pprint.pprint(max_heads)
    #pprint.pprint(cycle) #list