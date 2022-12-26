import pprint
import collections

from itertools import cycle as cy

from data import Read
from graph import Graph



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
        cycle_score += graph[cycle_node][next_node]

    return cycle_score



def get_return_graph(max_heads:dict, lookup_graph:dict) -> dict:
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
        score = lookup_graph[max_head][dep]
        return_graph[max_head][dep] = score

    return return_graph
    


def CLE(graph:dict):
    
    '''
    GRAPH:dict where
    keys are heads
    values are dicts where
        keys are dependents
        values are scores
    '''

    ## include graph_ob as argument / global graph_ob ??
    ## dict keys (nodes) as str or int?
    ## direction of arcs within graph + ordered ? (get score)
    ## Scores for edges entering cycle are massive (?)
        ## should be okay, scores for in-cycle edges get summed up
    ## re-reverse max_heads before returning? DONE (get_return_graph)
    ## Return graph: need scores? (get_return_graph)
    ## new_graph: remove cycle nodes from dependents of out-cycle nodes - DONE
    ## add edges not involved in cycle in a function ? (repeated)
    ## Scores in max_heads /y_graph / return_graph ?
    ## return_graph excludes nodes that are not max_head of any other token !!
        ## as keys - add keys anyway with empty dict?
    ## nicer expression for node_head_of_vc

    all_nodes = graph.keys()

    # Reverse graph
    rev_graph = graph_ob.reverse_graph(graph=graph)

    # Pick head with max score for each dependent
    max_heads = graph_ob.find_max_heads_simple(rev_graph=rev_graph)

    # Check if graph has a cycle
    cycle = graph_ob.find_cycle_on_max_heads(graph=max_heads) # list

    if cycle is None:
        return_graph = get_return_graph(max_heads=max_heads, lookup_graph=graph)
        return return_graph
    
    else:
        ''' Contract cycle and recompute scores '''
        # Make new graph with conctracted cycle

        print("\ngraph")
        pprint.pprint(graph) # has all possible nodes with scores
        print("\nmax_heads")
        pprint.pprint(max_heads)
        pprint.pprint(f"cycle: {cycle}")
        #pprint.pprint(rev_graph)
        #pprint.pprint(max_heads)

        # Graph excluding in-cycle nodes
        #new_graph = {k:v for k,v in graph.items() if k not in cycle}
        new_graph = dict()

        # Add all nodes that are not involved in the cycle
        for head, dep_dict in graph.items():
            if head not in cycle:
                new_graph[head] = {}

                # Add edges that are not involved in the cycle
                for dep, score in dep_dict.items():
                    if dep not in cycle:
                        new_graph[head][dep] = score


        # Add contracted node Vc
        new_graph['Vc'] = {}
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
                    max_out = max(
                        filter(
                            lambda x: x in cycle, rev_graph[node]
                            ), 
                            key=rev_graph[node].get
                        )
                    
                    score = graph[max_out][node]

                    # Add edge [vc -> current out-cycle node]
                    new_graph['Vc'][node] = score
                    
                    # Remember that it orginally was from max_out (in-cycle node)
                    # [max_out -> out-cycle node]
                    tracker_outward[node] = max_out

                '''Arcs entering Vc'''

                # bp: breaking point
                max_bp_score = float()

                # Iterate through in-cycle nodes
                for cycle_node in cycle:
                    enter_score = graph[node][cycle_node]
                    # Previous node in the cycle (with incoming edge to current cycle node)
                    prev_cycle_node = get_prev_node(cycle_node, cycle)
                    # Score of the edge [prev cycle node -> cycle node]
                    prev_score = graph[prev_cycle_node][cycle_node]
                    
                    # bp: breaking point
                    bp_score = enter_score + cycle_score - prev_score
                    
                    # Find in-cycle node with max bp score for
                    # edge [out-cycle node -> in-cycle node]
                    if bp_score > max_bp_score:
                        max_bp_score = bp_score

                        # Remember which in-cycle node gives max bp score
                        tracker_inward[node] = cycle_node

                # Add edge [out-cycle node -> in-cycle node with max bp score]
                new_graph[node]['Vc'] = max_bp_score


        print("\nnew_graph")
        pprint.pprint(new_graph)
        pprint.pprint(f"tracker_outward: {tracker_outward}")
        pprint.pprint(f"tracker_inward: {tracker_inward}")


        ''' Call CLE (recursively) on new graph '''

        y_graph = CLE(new_graph)
        print("\ny_graph")
        pprint.pprint(y_graph)


        ''' Resolve cycle '''

        resolved_graph = dict()
        
        # Add all nodes that are not involved in the cycle
        for head, dep_dict in graph.items():
            if head not in cycle:
                resolved_graph[head] = {}

                # Add edges that are not involved in the cycle
                for dep, score in dep_dict.items():
                    if dep not in cycle:
                        resolved_graph[head][dep] = score

        ''' Arc entering Vc (always one) '''
        # [One out-cycle node -> Vc]
        # After find_max_head it can only have one head

        #node_head_of_vc = [k for k,v in y_graph.items() if v == 'Vc'][0]
        node_head_of_vc = str()
        for head, dep_dict in y_graph.items():
            for dep in dep_dict.keys():
                if dep == 'Vc':
                    node_head_of_vc = head


        node_dep_incycle = tracker_inward[node_head_of_vc]
        # Score is from the original graph 
        score_enter_cycle = graph[node_head_of_vc][node_dep_incycle]

        # Add edge [out-cycle node head of vc -> in-cycle node with max bp score]
        resolved_graph[node_head_of_vc][node_dep_incycle] = score_enter_cycle
        
        # Add in cycle edges [prev cycle node -> cycle node]
        # Except if cycle node already has a head
        # (dependent of head of Vc)
        for cycle_node in cycle:
            if cycle_node != node_dep_incycle:

                prev_cycle_node = get_prev_node(cycle_node, cycle)
                score_incycle_edge = graph[prev_cycle_node][cycle_node]

                # prev_cycle_node cannot be in resolved_graph already -> add
                resolved_graph[prev_cycle_node] = {cycle_node:score_incycle_edge}


        ''' Arcs leaving Vc (can be any number, incl. 0) '''
        # [Vc -> out-cycle nodeS that have selected Vc as max head]
        
        # y_graph example
        # {'0': {'Vc': score},
        # 'Vc': {'1': score, 
        #       '2': score}
        # }

        for head, dep_dict in y_graph.items():
            if head == 'Vc':
                for dep, score in dep_dict.items():
                    # Lookup where within the cycle edge originally came from
                    og_head_node = tracker_outward[dep] 
                    score_exit_cycle = graph[og_head_node][dep]
                    resolved_graph[og_head_node][dep] = score

       
        print("\nresolved_graph")
        pprint.pprint(resolved_graph)
        

        return resolved_graph









if __name__ == "__main__":

    blind_file = "wsj_dev.conll06.blind"
    language = "english"
    mode = "dev"

    reader = Read(blind_file, language, mode)

    test_sent1 = reader.all_sentences[23] #Sentence object (len 5)
    test_sent2 = reader.all_sentences[266] # len 7, no cycle
    test_sent3 = reader.all_sentences[301] # len 8, runs without errors - double check
    test_sent4 = reader.all_sentences[70] # len 9
    test_sent5 = reader.all_sentences[54] # len 10, error

    #print(len(test_sent3.id)) # Check sentence length

    # make graph obj outside of CLE call
    graph_ob = Graph(sentence_ob=test_sent3)
    graph = graph_ob.graph

    # rev_graph() call inside CLE call
    pprint.pprint(CLE(graph=graph))

    #max_heads = graph_ob.find_max_heads_simple(rev_graph=rev_graph)

    #cycle = graph_ob.find_cycle_on_max_heads(graph=max_heads)

    #pprint.pprint(max_heads)
    #pprint.pprint(cycle)