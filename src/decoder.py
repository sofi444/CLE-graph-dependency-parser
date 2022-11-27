import sys
import collections
import pprint
import random


from data import Read


blind_file = "wsj_dev.conll06.blind"
language = "english"
mode = "dev"

random.seed(1) 



class Graph:
    def __init__(self, sentence_ob) -> None:
        self.sentence_ob = sentence_ob
        self.nodes = self.sentence_ob.id
        self.graph = {}

        '''
        GRAPH:dict
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
        


    def reverse_graph(self) -> dict:

        '''
        Reverse the graph so that
            graph[head][dep] == rev_graph[dep][head]
        
        REV_GRAPH is a dict where
        keys are dependents
        values are dicts where
            keys are heads
            values are scores
        '''
        
        rev_graph = {}
        
        for head in self.graph.keys():
            for dep, score in self.graph[head].items():
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



    def reverse_max_heads(self, max_heads:dict):
        
        '''
        Reverse MAX_HEADS dict so that
        
        REV_MAX_HEADS is a dict where
        keys are head nodes
        values are lists of dependents of that head node
        * [] if node is not the head of any other node
        '''

        rev_max_heads = collections.defaultdict(list)
        for dep, max_head in max_heads.items():
            rev_max_heads[max_head[0]].append(dep)
        
        return rev_max_heads



    def has_cycle(self, graph:dict, node:int, visited:list, rec_stack:list):

        '''
        Recurrent function
        '''
        
        visited[node] = True
        rec_stack[node] = True

        for dep in graph[str(node)]:
            if visited[int(dep)] == False:
                if self.has_cycle(graph, int(dep), visited, rec_stack) == True:
                    return True
            elif rec_stack[int(dep)] == True:
                return True
        
        rec_stack[node] = False

        return False



    def find_cycle(self, graph:dict, n_nodes:int) -> list | None:

        '''
        Check if there is a cycle in the graph
        '''

        visited = [False] * (n_nodes)
        rec_stack = [False] * (n_nodes)

        for node in range(n_nodes):
            if visited[node] == False:
                if self.has_cycle(graph, node, visited, rec_stack) == True:
                    return True

        return False





reader = Read(blind_file, language, mode)

test_sent1 = reader.all_sentences[23] #Sentence object
test_sent2 = reader.all_sentences[54]

graph_ob = Graph(sentence_ob=test_sent1)

graph = graph_ob.graph
rev_graph = graph_ob.reverse_graph()

#pprint.pprint(graph)
#pprint.pprint(rev_graph)

max_heads = graph_ob.find_max_heads(rev_graph=rev_graph)
rev_max_heads = graph_ob.reverse_max_heads(max_heads=max_heads)

n_nodes = len(graph_ob.nodes)

cycle = graph_ob.find_cycle(graph=rev_max_heads, n_nodes=n_nodes)

pprint.pprint(max_heads)
pprint.pprint(rev_max_heads)
pprint.pprint(cycle)





'''
len(reader.all_sentences)) == 1083 (sentences)
    list of Sentence objects

Sent 23: len == 5
    sent.form:
    ['ROOT', 'Two', '-', 'Way', 'Street']

Sent 54: len == 10
    sent.form:
    ['ROOT', 'Two', 'share', 'a', 'house', 'almost', 'devoid', 'of', 'furniture', '.']

print(test_sent1.id)
print(test_sent1.form)
print(test_sent1.lemma)
print(test_sent1.pos)
print(test_sent1.xpos)
print(test_sent1.morph)
print(test_sent1.head)
print(test_sent1.rel)
print(test_sent1.empty1)
print(test_sent1.empty2)

['0', '1', '2', '3', '4']
['ROOT', 'Two', '-', 'Way', 'Street']
['ROOT', 'two', '-', 'way', 'street']
['_', 'CD', 'HYPH', 'NNP', 'NNP']
['_', '_', '_', '_', '_']
['_', '_', '_', '_', '_']
['_', '_', '_', '_', '_']
['_', '_', '_', '_', '_']
['_', '_', '_', '_', '_']
['_', '_', '_', '_', '_']
'''