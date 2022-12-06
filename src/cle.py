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



def CLE(graph:dict):
    
    '''
    GRAPH:dict where
    keys are heads
    values are dicts where
        keys are dependents
        values are scores
    '''

    ## include graph_ob as argument / global graph_ob ??
    ## re-reverse max_heads before returning? 
    ## dict keys (nodes) as str or int?
    ## direction of arcs within graph + ordered ? (get score)

    all_nodes = graph.keys()

    # Reverse graph
    rev_graph = graph_ob.reverse_graph(graph=graph)

    # Pick head with max score for each dependent
    max_heads = graph_ob.find_max_heads_simple(rev_graph=rev_graph)

    # Check if graph has a cycle
    cycle = graph_ob.find_cycle_on_max_heads(graph=max_heads) # list

    if cycle is None:
        return max_heads # i.e., MST
    
    else:
        # Contract cycle and recompute scores
        # Make new graph with conctracted cycle

        #pprint.pprint(graph) # has all possible nodes with scores
        pprint.pprint(cycle)
        #pprint.pprint(graph)
        #pprint.pprint(rev_graph)
        pprint.pprint(max_heads)

        # Graph excluding in-cycle nodes
        new_graph = {k:v for k,v in graph.items() if k not in cycle}
        #pprint.pprint(new_graph)

        new_graph['Vc'] = {} # Contracted node
        cycle_score = get_cycle_score(graph, cycle)

        # Keys: out-cycle nodes, values: the in-cycle node with max score
        tracker_outward = {}
        tracker_inward = {}
        
        # For each node not in cycle
        for node in all_nodes:
            if node not in cycle and node != '0':

                ''' Arcs leaving Vc '''

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
                
                # Remember that it orginally was from max_out (in-cycle)
                # [node <- max_out]
                tracker_outward[node] = max_out

                '''Arcs entering Vc'''

                # bp: breaking point
                max_bp_score = float()

                # Iterate through in-cycle nodes
                for cycle_node in cycle:
                    enter_score = graph[node][cycle_node]
                    prev_cycle_node = get_prev_node(cycle_node, cycle)
                    prev_score = graph[prev_cycle_node][cycle_node]
                    
                    # bp: breaking point
                    bp_score = enter_score + cycle_score - prev_score
                    
                    # Find in-cycle node with max bp score for
                    # edge [out-cycle node -> in-cycle node]

                    if bp_score > max_bp_score:
                        max_bp_score = bp_score

                        # Remember which in-cycle node gives max bp score
                        tracker_inward[node] = cycle_node

                # Add only edge [out-cycle node -> in-cycle node with max bp score]
                new_graph[node]['Vc'] = max_bp_score

                ## Scores are massive (?)


        #pprint.pprint(new_graph)
        #pprint.pprint(tracker_outward)
        #pprint.pprint(tracker_inward)
            
        

        # Arcs entering vc (keys of rev_graph if key in cycle)





        # Call CLE (recursively) on new graph
        # Resolve cycle










if __name__ == "__main__":

    blind_file = "wsj_dev.conll06.blind"
    language = "english"
    mode = "dev"

    reader = Read(blind_file, language, mode)

    test_sent1 = reader.all_sentences[23] #Sentence object
    test_sent2 = reader.all_sentences[54] #longer

    # make graph obj outside of CLE call
    graph_ob = Graph(sentence_ob=test_sent2)
    graph = graph_ob.graph

    # rev_graph() call inside CLE call
    pprint.pprint(CLE(graph=graph))

    #max_heads = graph_ob.find_max_heads_simple(rev_graph=rev_graph)

    #cycle = graph_ob.find_cycle_on_max_heads(graph=max_heads)

    #pprint.pprint(max_heads)
    #pprint.pprint(cycle)