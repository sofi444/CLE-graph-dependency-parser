import pprint
import collections

from itertools import cycle as cy

from data import Read
from graph_v2 import Graph



def get_prev_node(cycle_node:str, cycle:list) -> str:
    node_idx = cycle.index(cycle_node)
    
    if node_idx == 0:
        last_idx = len(cycle)-1
        prev_node = cycle[last_idx]
    else:
        prev_node = cycle[node_idx-1]

    return prev_node



def get_next_node(cycle_node:str, cycle:list) -> str:
    node_idx = cycle.index(cycle_node)
    
    if node_idx == len(cycle)-1:
        next_node = cycle[0]
    else:
        next_node = cycle[node_idx+1]

    return next_node



def get_cycle_score(graph:dict, cycle:list) -> int:
    cycle_score = float()

    for cycle_node in cycle:
        next_node = get_next_node(cycle_node, cycle)
        cycle_score += graph[cycle_node][next_node]['score']

    return cycle_score



def get_return_graph(max_heads:dict, lookup_graph:dict, all_nodes:list) -> dict:
    return_graph = collections.defaultdict(dict)

    # max_heads example:
    # {'1': '9', 
    # '2': '6', 
    # '3': '8', 
    # '4': '7', 
    # '5': '2', --
    # '6': '1', 
    # '7': '2', --
    # '8': '5', ++
    # '9': '5'} ++

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



def is_spanning_tree(test_graph:dict, og_graph:dict) -> bool:
    
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
    


def CLE(graph:dict):

    # Debug
    global recursion_it
    print("\nRecursion iteration number: ", recursion_it)
    recursion_it += 1
    
    '''
    GRAPH:dict where
    keys are heads
    values are dicts where
        keys are dependents
        values are dicts where
            keys are fv, score
            values are list(fv), float(score)
    '''

    all_nodes = graph.keys()

    # Debug
    print("All nodes of graph: ", all_nodes)
    pprint.pprint(graph)

    # Reverse graph
    rev_graph = graph_ob.reverse_graph(graph=graph)

    # Pick head with max score for each dependent
    max_heads = graph_ob.find_max_heads_simple(rev_graph=rev_graph)

    # Check if graph has a cycle
    cycle = graph_ob.find_cycle_on_max_heads(graph=max_heads) # list

    if cycle is None:

        # Debug
        print("No cycle found, calling return_graph()...")

        return_graph = get_return_graph(max_heads=max_heads, 
                                        lookup_graph=graph, 
                                        all_nodes=all_nodes)

        return return_graph
    
    else:
        ''' Contract cycle and recompute scores '''
        # Make new graph with conctracted cycle

        # Debug
        #print("\ngraph")
        #pprint.pprint(graph) # has all possible nodes with scores
        #print("\nmax_heads")
        #pprint.pprint(max_heads)
        #pprint.pprint(f"Cycle found: {cycle}")
        #pprint.pprint(rev_graph)
        #pprint.pprint(max_heads)

        # Graph excluding in-cycle nodes
        new_graph = dict()

        # Add all nodes that are not involved in the cycle
        for head, dep_dict in graph.items():
            if head not in cycle:
                new_graph[head] = {}

                # Add edges that are not involved in the cycle
                for dep, arc_attributes in dep_dict.items():
                    if dep not in cycle:
                        score = arc_attributes['score']
                        new_graph[head][dep] = {'fv': [], 'score': score}


        # Add contracted node Vc
        vc = str(max([int(i) for i in all_nodes])+1)
        new_graph[vc] = {}

        cycle_score = get_cycle_score(graph, cycle)

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
                max_bp_score = float()

                # Iterate through in-cycle nodes
                for cycle_node in cycle:
                    enter_score = graph[node][cycle_node]['score']
                    # Previous node in the cycle (with incoming edge to current cycle node)
                    prev_cycle_node = get_prev_node(cycle_node, cycle)
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


        # Debug
        #print("\nnew_graph")
        #pprint.pprint(new_graph)
        #pprint.pprint(f"tracker_outward: {tracker_outward}")
        #pprint.pprint(f"tracker_inward: {tracker_inward}")


        ''' Call CLE (recursively) on new graph '''

        # Debug
        #print("Calling CLE() again...")

        y_graph = CLE(new_graph)

        # Debug
        print("\ny_graph")
        pprint.pprint(y_graph)
        print("\ngraph")
        pprint.pprint(graph)


        ''' Resolve cycle '''

        resolved_graph = collections.defaultdict(dict)
        
        # Add all nodes that are not involved in the cycle
        for head, dep_dict in y_graph.items():
            if head not in cycle and head != vc:
                resolved_graph[head] = {}

                # Add edges that are not involved in the cycle
                for dep, arc_attributes in dep_dict.items():
                    if dep not in cycle and dep != vc:
                        fv = arc_attributes['fv']
                        score = arc_attributes['score']
                        resolved_graph[head][dep] = {'fv': fv, 'score': score}

        # Debug
        #print("\nDEBUG - added noded/edges not involved in the cycle")
        #print("cycle: ", cycle)
        #print("nodes: ", all_nodes)
        #pprint.pprint(resolved_graph)

        ''' Arc entering Vc (always one) '''
        # [One out-cycle node -> Vc]
        # After find_max_head it can only have one head

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
        # Except if cycle node already has a head
        # (dependent of head of Vc)
        for cycle_node in cycle:
            if cycle_node != node_dep_incycle:

                prev_cycle_node = get_prev_node(cycle_node, cycle)
                score_incycle_edge = graph[prev_cycle_node][cycle_node]['score']
                fv_incycle_edge = graph[prev_cycle_node][cycle_node]['fv']

                # prev_cycle_node cannot be in resolved_graph already -> add
                resolved_graph[prev_cycle_node] = {cycle_node:score_incycle_edge}
                resolved_graph[prev_cycle_node] = {}
                resolved_graph[prev_cycle_node][cycle_node] = {'fv': fv_incycle_edge,
                                                            'score': score_incycle_edge}


        # Debug
        #print("\nDEBUG - added edges entering Vc")
        #print("nodes:", all_nodes)
        #pprint.pprint(resolved_graph)


        ''' Arcs leaving Vc (can be any number, incl. 0) '''
        # [Vc -> out-cycle nodeS that have selected Vc as max head]
        
        # y_graph example
        # {'0': {'Vc': score},
        # 'Vc': {'1': score, 
        #       '2': score}
        # }

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


        # Debug
        #print("\nDEBUG - added arcs leaving Vc")
        #print("resolved_graph")
        #pprint.pprint(resolved_graph)
        

        return resolved_graph
        






if __name__ == "__main__":

    blind_file = "wsj_dev.conll06.blind"
    language = "english"
    mode = "dev"

    reader = Read(blind_file, language, mode)

    test_sent1 = reader.all_sentences[23] # Sentence object (len 5) - works well
    test_sent2 = reader.all_sentences[266] # len 7, no cycle
    test_sent3 = reader.all_sentences[301] # len 8
    test_sent4 = reader.all_sentences[70] # len 9
    test_sent5 = reader.all_sentences[54] # len 10

    #print(len(test_sent4.id)) # Check sentence length
    #print(test_sent4.form)

    # make graph obj outside of CLE call
    graph_ob = Graph(sentence_ob=test_sent5)
    graph = graph_ob.graph

    # Debug
    recursion_it = 1

    final_graph = CLE(graph=graph)

    print("\nfinal_graph")
    pprint.pprint(final_graph)

    print(is_spanning_tree(test_graph=final_graph, og_graph=graph))