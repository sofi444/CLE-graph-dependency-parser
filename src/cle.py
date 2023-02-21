
import pprint
import collections
import random

import networkx as nx

from data import Read
from graphs import Graph
from features import Features



class Decoder:
    def __init__(self) -> None:
        pass



    def nx_CLE(self, graph:dict) -> dict:

        '''
        Chu Liu Edmonds algorithm
        NetworkX implementation

        Returns a maximum spanning tree (dict)
        '''

        nx_graph = nx.DiGraph(graph)

        cle = nx.algorithms.tree.branchings.Edmonds(nx_graph)

        nx_mst = cle.find_optimum(
            attr='score', 
            kind='max', 
            style='aborescence', 
            preserve_attrs=True
        )

        mst_dict = nx.to_dict_of_dicts(nx_mst)

        return mst_dict



    def CLE(self, graph:dict) -> dict:
        
        '''
        Chu Liu Edmonds algorithm
        Returns a maximum spanning tree (dict) given the scores 
        of edges in the input graph (fully connencted).

        GRAPH:dict where
        keys are heads
        values are dicts where
            keys are dependents
            values are dicts where
                keys are fv, score
                values are list(fv), float(score)
        '''

        all_nodes = graph.keys()

        # Reverse graph
        rev_graph = Graph.reverse_graph(self, graph=graph)

        # Pick head with max score for each dependent
        max_heads = Graph.find_max_heads(self, rev_graph=rev_graph)
        
        # Check if graph has a cycle
        cycle = Graph.find_cycle(self, graph=max_heads) # list

        if cycle is None:

            return_graph = self.get_return_graph(max_heads=max_heads, 
                                            lookup_graph=graph, 
                                            all_nodes=all_nodes)

            return return_graph
        
        else:
            ''' Contract cycle and recompute scores '''
            # Make new graph with conctracted cycle

            new_graph = dict()

            # Add all nodes that are not involved in the cycle
            for head, dep_dict in graph.items():
                if head not in cycle:
                    new_graph[head] = {}

                    # Add edges that are not involved in the cycle
                    for dep, arc_attributes in dep_dict.items():
                        if dep not in cycle:
                            new_graph[head][dep] = {
                                'fv': arc_attributes['fv'], 
                                'score': arc_attributes['score']
                            }


            # Add contracted node Vc
            vc = str(max([int(i) for i in all_nodes])+1)
            new_graph[vc] = {}

            cycle_score = self.get_cycle_score(graph, cycle)

            # Keys: out-cycle nodes, values: the in-cycle node with max score
            tracker_outward = {}
            tracker_inward = {}
            
            # For each node not in cycle
            for node in all_nodes:
                if node not in cycle:

                    ''' Arcs leaving Vc '''

                    if node != '0': # Avoid KeyError (no incoming edges to ROOT)
                    
                        # Find [in-cycle node -> current out-cycle node] with max score
                        max_out, out_score = None, -100.00
                        for cycle_node in cycle:
                            if graph[cycle_node][node]['score'] > out_score:
                                max_out = cycle_node
                                out_score = graph[cycle_node][node]['score']

                        # Add edge [vc -> current out-cycle node]
                        new_graph[vc][node] = {'fv': [], 'score': out_score}
                        
                        # Remember that it orginally was from max_out (in-cycle node)
                        # [max_out -> out-cycle node]
                        tracker_outward[node] = max_out


                    
                    '''Arcs entering Vc'''

                    # bp: breaking point
                    max_bp_score = -1000.00

                    # Iterate through in-cycle nodes
                    for cycle_node in cycle:
                        enter_score = graph[node][cycle_node]['score']
                        # Previous node in the cycle (with incoming edge to current cycle node)
                        prev_cycle_node = self.get_prev_node(cycle_node, cycle)
                        # Score of the edge [prev cycle node -> cycle node]
                        prev_score = graph[prev_cycle_node][cycle_node]['score']
                        
                        # bp: breaking point
                        bp_score = enter_score + cycle_score - prev_score
                        
                        # Find in-cycle node with max bp score for
                        # edge [out-cycle node -> in-cycle node]
                        if bp_score > max_bp_score:
                            max_bp_score = bp_score

                            # Remember which in-cycle node gives max bp score
                            tracker_inward[node] = cycle_node

                    # Add edge [out-cycle node -> in-cycle node with max bp score]
                    new_graph[node][vc] = {'fv': [], 'score': max_bp_score}


            ''' Call CLE (recursively) on new graph '''

            y_graph = self.CLE(new_graph)


            ''' Resolve cycle '''

            resolved_graph = collections.defaultdict(dict)
            
            # Add all nodes that are not involved in the cycle
            for head, dep_dict in y_graph.items():
                if head not in cycle and head != vc:
                    resolved_graph[head] = {}

                    # Add edges that are not involved in the cycle
                    for dep, arc_attributes in dep_dict.items():
                        if dep not in cycle and dep != vc:

                            resolved_graph[head][dep] = {
                                'fv': arc_attributes['fv'], 
                                'score': arc_attributes['score']
                            }


            ''' Arc entering Vc (always one) '''
            # [One out-cycle node -> Vc]
            # After find_max_heads, Vc can only have one head

            node_head_of_vc = str()
            for head, dep_dict in y_graph.items():
                for dep in dep_dict.keys():
                    if dep == vc:
                        node_head_of_vc = head

            node_dep_incycle = tracker_inward[node_head_of_vc]
            # Score is from the original graph 
            score_enter_cycle = graph[node_head_of_vc][node_dep_incycle]['score']
            fv_enter_cycle = graph[node_head_of_vc][node_dep_incycle]['fv']

            # Add edge [out-cycle node head of vc -> in-cycle node with max bp score]
            resolved_graph[node_head_of_vc][node_dep_incycle] = {'fv': fv_enter_cycle,
                                                                'score': score_enter_cycle}

            # Add in cycle edges [prev cycle node -> cycle node]
            # Except if cycle node already has a head (i.e. is a dep of head of Vc)
            for cycle_node in cycle:
                if cycle_node != node_dep_incycle:

                    prev_cycle_node = self.get_prev_node(cycle_node, cycle)
                    score_incycle_edge = graph[prev_cycle_node][cycle_node]['score']
                    fv_incycle_edge = graph[prev_cycle_node][cycle_node]['fv']

                    # prev_cycle_node cannot be in resolved_graph already -> add
                    resolved_graph[prev_cycle_node][cycle_node] = {'fv': fv_incycle_edge,
                                                                'score': score_incycle_edge}


            ''' Arcs leaving Vc (can be any number, incl. 0) '''
            # [Vc -> out-cycle nodeS that have selected Vc as max head]

            for head, dep_dict in y_graph.items():
                if head == vc:
                    for dep in dep_dict.keys():
                        # Lookup where within the cycle edge originally came from
                        og_head_node = tracker_outward[dep] 
                        score_exit_cycle = graph[og_head_node][dep]['score']
                        fv_exit_cycle = graph[og_head_node][dep]['fv']
                        
                        resolved_graph[og_head_node][dep] = {'fv': fv_exit_cycle,
                                                            'score': score_exit_cycle}

            
            # Add to final graph nodes that have no dependents
            # as keys with value == {}
            for node in all_nodes:
                if node not in resolved_graph.keys():
                    resolved_graph[node] = {}


            return resolved_graph



    def get_prev_node(self, cycle_node:str, cycle:list) -> str:

        ''' 
        Returns node that precedes the given node in the cycle
        '''

        node_idx = cycle.index(cycle_node)
        
        if node_idx == 0:
            last_idx = len(cycle)-1
            prev_node = cycle[last_idx]
        else:
            prev_node = cycle[node_idx-1]

        return prev_node



    def get_next_node(self, cycle_node:str, cycle:list) -> str:
        
        ''' 
        Returns node that follows the given node in the cycle
        '''

        node_idx = cycle.index(cycle_node)
        
        if node_idx == len(cycle)-1:
            next_node = cycle[0]
        else:
            next_node = cycle[node_idx+1]

        return next_node



    def get_cycle_score(self, graph:dict, cycle:list) -> int:

        '''
        Returns the score for the cycle 
        (sum of in-cycle edges' scores)
        '''

        cycle_score = float()

        for cycle_node in cycle:
            next_node = self.get_next_node(cycle_node, cycle)
            cycle_score += graph[cycle_node][next_node]['score']

        return cycle_score



    def get_return_graph(self, max_heads:dict, lookup_graph:dict, all_nodes:list) -> dict:

        return_graph = collections.defaultdict(dict)

        # Retrieve scores and feature vectors from original graph
        for dep, max_head in max_heads.items():
            fv = lookup_graph[max_head][dep]['fv']
            score = lookup_graph[max_head][dep]['score']
            return_graph[max_head][dep] = {'fv': fv, 'score': score}

        # Add nodes that don't have dependents as keys of return graph
        # with value == {}
        for node in all_nodes:
            if node not in return_graph.keys():
                return_graph[node] = {}

        return return_graph



    def is_spanning_tree(self, test_graph:dict, og_graph:dict) -> bool:
        
        '''
        Conditions:
        1) # nodes == # nodes in original graph
        2) # edges == # nodes in original graph - 1
        3) No cycle
        '''

        num_edges = 0

        if test_graph.keys() == og_graph.keys():
            for dep_dict in test_graph.values():
                num_edges += len(dep_dict.keys())
            if num_edges == len(og_graph.keys())-1:
                return True
            else:
                return False
        else:
            return False





if __name__ == "__main__":

    test_reader = Read(in_file="wsj_dev.conll06.blind",
                        language="english",
                        mode="dev")
    
    train_reader = Read(in_file="wsj_train.first-1k.conll06",
                        language="english",
                        mode="train") 
    
    train_data = train_reader.all_sentences[:20]

    test_sent1 = test_reader.all_sentences[23] # Sentence object (len 5)
    test_sent2 = test_reader.all_sentences[266] # len 7
    test_sent3 = test_reader.all_sentences[301] # len 8
    test_sent4 = test_reader.all_sentences[70] # len 9
    test_sent5 = test_reader.all_sentences[54] # len 10
    test_sent6 = test_reader.all_sentences[100] # a long sentence

    feat = Features()
    feature_map = feat.create_feature_map(train_data=train_data)

    random.seed(7)
    tmp_w = [random.uniform(0,20) for _ in range(len(feature_map))]


    graph_ob = Graph(sentence=test_sent6,
                    feature_map=feature_map,
                    weight_vector=tmp_w,
                    graph_type="fully_connected")

    graph = graph_ob.graph

    dec = Decoder()


    ''' My CLE '''

    final_graph = dec.CLE(graph=graph)

    print("\nfinal_graph")
    pprint.pprint(final_graph, compact=True, width=120)

    print(dec.is_spanning_tree(test_graph=final_graph, og_graph=graph))


    ''' NetworkX CLE '''

    #final_graph_nx = dec.nx_CLE(graph=graph)

    #print("\nfinal_graph_nx")
    #pprint.pprint(final_graph_nx, compact=True, width=120)

    #print(dec.is_spanning_tree(test_graph=final_graph_nx, og_graph=graph))
