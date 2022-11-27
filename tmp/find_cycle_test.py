def has_cycle(graph:dict, node:int, visited:list, rec_stack:list):
    
    visited[node] = True
    rec_stack[node] = True

    for dep in graph[str(node)]:
        if visited[int(dep)] == False:
            has_cycle_bool, node_in_cycle = has_cycle(graph, int(dep), visited, rec_stack)
            if has_cycle_bool == True:
                node_in_cycle = dep
                return True, node_in_cycle
            else:
                rec_stack[node] = False
        elif rec_stack[int(dep)] == True:
            node_in_cycle = dep
            return True, node_in_cycle
    
    #rec_stack[node] = False

    return False, None



def find_cycle(graph:dict, n_nodes:int):

    visited = [False] * (n_nodes)
    rec_stack = [False] * (n_nodes)
    cycle = []
    cycle_found = None

    for node in range(n_nodes):
        if visited[node] == False:
            has_cycle_bool, node_in_cycle = has_cycle(graph, node, visited, rec_stack)
            if has_cycle_bool == True:
                cycles.append(node_in_cycle)
                cycle_found = True
                #return True
    if cycle_found == True:
        return cycles 
    else:
        return []




def find_cycle_on_max_heads(graph:dict):
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
            return cycle # returns first cycle found

            

    

graph = {'0':['2'],
        '1':[],
        '2':[],
        '3':['1','4'],
        '4':['3']}
 
global cycles
cycles =[]

#find_cycle(graph, 5)

max_heads = {'2':'0',
            '1':'3',
            '3':'4',
            '4':'3'}

print(find_cycle_on_max_heads(graph=max_heads))