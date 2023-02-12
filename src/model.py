# structured perceptron

import pprint
import collections
import random

from data import Read
from features import Features
from graph_v2 import Graph
from cle_v2 import Decoder



class StructuredPerceptron:
    def __init__(self, weight_vector=None, 
                gold_graph=None, fully_connected_graph=None,
                feature_map=None,
                mode="test",
                lr=0.2) -> None:

        global D
        D = Decoder()

        self.w = weight_vector

        self.gold_graph = gold_graph
        self.fc_graph = fully_connected_graph

        self.feature_map = feature_map
        self.mode = mode
        self.lr = lr

        #if self.w is None:
        #    self.w = [1 for i in range(len(self.feature_map))] # tmp



    def train(self):
        
        # predict
        pred_graph = D.CLE(graph=self.fc_graph)

        #if not D.is_spanning_tree(test_graph=pred_graph, og_graph=self.fc_graph):
        #    print('NOT a spanning tree')
    

        uas_sent, correct_arcs, total_arcs = self.calculate_UAS_sent(
            pred_graph=pred_graph, gold_graph=self.gold_graph
        )


        if uas_sent == 1.0: # all arcs are correct -> return
            #print(f"uas_sent: 100%") # complete match

            return self.w, uas_sent, correct_arcs, total_arcs


        else: # need to update weight vector

            # sum up features
            fsum_gold = self.get_features_sum(self.gold_graph)
            fsum_pred = self.get_features_sum(pred_graph)

            
            # update weight vector
            combined_features = {**fsum_pred, **fsum_gold}.keys()

            for f in combined_features:
                f_idx = int(f)

                if f in fsum_gold and f in fsum_pred:
                    self.w[f_idx] += self.lr * (fsum_gold[f] - fsum_pred[f])

                if f in fsum_gold and f not in fsum_pred:
                    self.w[f_idx] += self.lr * fsum_gold[f]

                if f in fsum_pred and f not in fsum_gold:
                    self.w[f_idx] += self.lr * (0 - fsum_pred[f])
            
            return self.w, uas_sent, correct_arcs, total_arcs



    def test(self):

        # predict
        pred_graph = D.CLE(graph=self.fc_graph)

        #if not D.is_spanning_tree(test_graph=pred_graph, og_graph=self.fc_graph):
        #    print('NOT a spanning tree')
        #    pred_graph = {}


        uas_sent, correct_arcs, total_arcs = self.calculate_UAS_sent(
            pred_graph=pred_graph, gold_graph=self.gold_graph
        )

        
        return pred_graph, uas_sent, correct_arcs, total_arcs



    def calculate_UAS_sent(self, pred_graph, gold_graph):

        correct_arcs = total_arcs = 0

        rev_pred_graph = Graph.reverse_graph(self, graph=pred_graph)
        rev_gold_graph = Graph.reverse_graph(self, graph=gold_graph)

        for dep in rev_pred_graph:
            assert len(rev_pred_graph[dep].keys()) == 1

            # retrive one and only key == assigned head (only one per token)
            pred_head = list(rev_pred_graph[dep].keys())[0]
            gold_head = list(rev_gold_graph[dep].keys())[0]
            
            if pred_head == gold_head:
                correct_arcs += 1
            
            total_arcs += 1
        
        uas_sent = correct_arcs / total_arcs

        return uas_sent, correct_arcs, total_arcs

            

    def get_features_sum(self, graph):

        features_sum = collections.defaultdict(lambda: 0)

        for head, dep_dict in graph.items():
            for dep, arc_attributes in dep_dict.items():
                for feature in arc_attributes['fv']:
                    features_sum[feature] += 1
        
        return features_sum






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

    features_ob = Features()
    feature_map = features_ob.create_feature_map(train_data)

    random.seed(7)
    tmp_w = [random.uniform(0,20) for _ in len(feature_map)]

    gold_ob = Graph(sentence=train_data[1],
                    feature_map=feature_map,
                    weight_vector=tmp_w,
                    graph_type="gold")

    gold_graph = gold_ob.graph

    #pprint.pprint(gold_graph)

    fully_connected_ob = Graph(sentence=train_data[1],
                                feature_map=feature_map,
                                weight_vector=tmp_w,
                                graph_type="fully_connected")

    fully_connected_graph = fully_connected_ob.graph

    model = StructuredPerceptron(weight_vector=None, 
                                gold_graph=gold_graph, 
                                fully_connected_graph=fully_connected_graph,
                                feature_map=feature_map,
                                mode="train")

    print(model.test())