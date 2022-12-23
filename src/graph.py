import sys
import collections
import pprint
import random


from data import Read


blind_file = "wsj_dev.conll06.blind"
language = "english"
mode = "dev"

random.seed(7) 



class Graph:
    def __init__(self, sentence_ob) -> None:
        self.sentence_ob = sentence_ob
        self.nodes = self.sentence_ob.id
        self.graph = {}

        '''
        GRAPH:dict where
        keys are heads
        values are dicts where
            keys are dependents
            values are scores
        '''

        for node in self.nodes:
            dependents = {
                
                dep_node: random.uniform(0, 100) # score is random float between 0 and 100
                for dep_node in self.nodes 
                if not (dep_node == '0' or dep_node == node) # impossible edges
                
                }
            
            self.graph[node] = dependents
        


    def reverse_graph(self, graph) -> dict:

        '''
        Reverse the graph so that
            graph[head][dep] == rev_graph[dep][head]
        
        REV_GRAPH:dict where
        keys are dependents
        values are dicts where
            keys are heads
            values are scores
        '''

        rev_graph = {}
        
        for head in graph.keys():
            for dep, score in graph[head].items():
                if dep not in rev_graph:
                    rev_graph[dep] = {head:score}
                else:
                    rev_graph[dep][head] = score
        
        return rev_graph



    def find_max_heads(self, rev_graph:dict) -> dict:

        '''
        For each dependent, find candidate head with max score

        MAX_HEADS is a dict where
        keys are dependents
        values are tuples where
            tuple[0] is the head with max score
            tuple[1] is the score

        ///

        ROOT CAN ONLY BE THE HEAD OF ONE TOKEN
        -> PREVENT MULTIPLE DEPS HAVING ROOT AS THEIR HEAD

        If ROOT has already been assigned and the current dep has ROOT 
        as its max head, it will take as head the second max token instead

        !! separate function for this?
        !! how does CLE deal with this?
        !! if two tokens have as their max head ROOT,
            which one gets the priority for keeping ROOT as its head?
        '''

        max_heads = {}
        root_assigned = False

        for dep, candidate_heads in rev_graph.items():
            
            sorted_heads = sorted(

                candidate_heads.items(),
                key=(lambda i: i[1])

                )
            
            max_heads[dep] = sorted_heads[-1] # tuple where (max_head, score)

            if max_heads[dep][0] == '0' and root_assigned == False:
                root_assigned = True
                continue

            if root_assigned == True and max_heads[dep][0] == '0':
                max_heads[dep] = sorted_heads[-2] # tuple where (second max head, score)
        
        return max_heads



    def find_max_heads_simple(self, rev_graph:dict) -> dict:

        '''
        For each dependent, find candidate head with max score

        MAX_HEADS is a dict where
        keys are dependents
        values are the head with max score
        '''

        max_heads = {}

        for dep, candidate_heads in rev_graph.items():
            
            max_head = max(
                candidate_heads, 
                key=lambda key: candidate_heads[key]
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




if __name__ == "__main__":

    reader = Read(blind_file, language, mode)

    test_sent1 = reader.all_sentences[23] #Sentence object
    test_sent2 = reader.all_sentences[54] #longer

    graph_ob = Graph(sentence_ob=test_sent1)

    graph = graph_ob.graph
    rev_graph = graph_ob.reverse_graph(graph=graph)

    #pprint.pprint(graph)
    #pprint.pprint(rev_graph)

    max_heads = graph_ob.find_max_heads_simple(rev_graph=rev_graph)

    cycle = graph_ob.find_cycle_on_max_heads(graph=max_heads)

    pprint.pprint(max_heads)
    pprint.pprint(cycle)
    pprint.pprint(type(cycle))