from __future__ import annotations
from causallearn.utils.GraphUtils import GraphUtils
import time,json,torch
from json import JSONDecodeError
import warnings
from itertools import combinations, permutations
from functools import reduce
import operator
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy import ndarray
from copy import deepcopy

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from causallearn.utils.causal import llm_redirect,redirect_test,llm_local_edge_check,llm_local_noedge_check,\
    json_test, build_dag_dict,build_causal_graph,find_nodes_without_edges,\
    clear_default_edges, GetFMB, evaluate_models,fix_json_string,ring,clean_json_string,clean_json_string1,llm_decycle,\
    propt_test1,llm_redirect1,llm_local_noedge_check1,llm_local_edge_check1
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

# The path of the fine-tuned model.
model_name = "xxxxxxxx"

def save_graph_as_png(graph, factor_list, output_path, file_suffix='0.001'):
    """
    Convert the graph object and factor list to a pydot graph and save it in PNG format.
    """
    pdy = GraphUtils.to_pydot(graph, labels=factor_list)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, f'122mpc0fewhuaxi51llama270b{file_suffix}03.png')
    pdy.write_png(output_file)

def exp_evidence(y):
    return torch.exp(torch.clamp(torch.tensor(y), -100, 100))


def edl_inference(evidence, num_classes):
    """
    Evidential Deep Learning inference process.

    Parameters:
    - evidence (list or np.ndarray): The evidence vector output by the model, shape (K,).
    - num_classes (int): The number of classes (K).

    Returns:
    - uncertainty (float): The model's uncertainty value.
    """
    # Step 1: Calculate Dirichlet parameters (alpha = evidence + 1)
    evidence = exp_evidence(evidence)
    alpha = np.array(evidence) + 1

    # Step 2: Calculate total evidence (S)
    S = np.sum(alpha)

    # Step 3: Compute uncertainty (u = K / S)
    uncertainty = num_classes / S

    return uncertainty


def calculate_uncertainty_list(evidence_list):
    """
    Calculate uncertainty for a list of evidence vectors.

    Parameters:
    - evidence_list (list of list): A list where each element is an evidence vector.

    Returns:
    - uncertainties (list): A list of uncertainties for each evidence vector.
    """
    # Automatically determine the number of classes (K) based on evidence vector size
    num_classes = len(evidence_list[0]) if evidence_list else 0

    uncertainties = [edl_inference(evidence, num_classes) for evidence in evidence_list]
    uncertainties = reduce(operator.mul, uncertainties, 1)
    return uncertainties



def dfs_find_cycles(graph, node, visited, path, all_cycles):
    """
    Use depth-first search (DFS) to find all cycles in the graph
    :param graph: The graph represented by a numpy matrix
    :param node: The current node
    :param visited: A list of visited nodes
    :param path: The nodes on the current path
    :param all_cycles: A list to store all found cycles
    :return: None
    """
    visited[node] = True
    path.append(node)

    for neighbor in range(len(graph)):
        if graph[neighbor, node] == 1 and graph[node, neighbor] == -1:
            if neighbor not in path:
                dfs_find_cycles(graph, neighbor, visited, path, all_cycles)
            else:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                if len(cycle) > 3:
                    all_cycles.append(cycle)

        elif graph[neighbor, node] == -1 and graph[node, neighbor] == -1:
            if neighbor not in path:
                dfs_find_cycles(graph, neighbor, visited, path, all_cycles)
            else:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                if len(cycle) > 3:
                    all_cycles.append(cycle)

        elif graph[neighbor, node] == 1 and graph[node, neighbor] == 1:
            if neighbor not in path:
                dfs_find_cycles(graph, neighbor, visited, path, all_cycles)
            else:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                if len(cycle) > 3:
                    all_cycles.append(cycle)

    path.pop()



def process_edges(string_list, FACTOR_list, GRAPH, cg_new):
    """
    Process the edge relationships in the list of strings, and update the graph and ndarray GRAPH structure based on "Remove" and "Reverse" operations.
    Parameters:
    string_list: A list containing the strings to be processed.
    FACTOR_list: A list storing node information.
    GRAPH: An adjacency matrix in ndarray format, representing the edges in the graph.
    cg_new: An object containing the graph structure, which will be modified within the function.
    Returns:
    The modified cg_new object and the GRAPH ndarray.
    """
    for raw_string in string_list:
        parts = [part.strip() for part in raw_string.split(":")]

        if len(parts) > 1:
            second_part = parts[1]
            single_edge = [part.strip() for part in second_part.split("and")]
            if len(single_edge) >= 2:
                single_edge_0 = single_edge[0]
                single_edge_1 = single_edge[1]

                if single_edge_0 in FACTOR_list and single_edge_1 in FACTOR_list:
                    id0 = FACTOR_list.index(single_edge_0)
                    id1 = FACTOR_list.index(single_edge_1)
                else:
                    print(f"Edge nodes {single_edge_0} or {single_edge_1} not found in FACTOR_list")
                    continue
                max_key = parts[0]

                if max_key == 'Remove':
                    edge0 = cg_new.G.get_edge(cg_new.G.nodes[id0], cg_new.G.nodes[id1])
                    if edge0 is not None:
                        cg_new.G.remove_edge(edge0)
                    # Update the GRAPH.
                    GRAPH[id0, id1] = 0
                    GRAPH[id1, id0] = 0

                elif max_key == 'Reverse':
                    edge0 = cg_new.G.get_edge(cg_new.G.nodes[id0], cg_new.G.nodes[id1])
                    if edge0 is not None:
                        cg_new.G.remove_edge(edge0)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[id1], cg_new.G.nodes[id0], Endpoint.TAIL, Endpoint.ARROW))
                    # Update the GRAPH.
                    GRAPH[id0, id1] = 1
                    GRAPH[id1, id0] = -1

            else:
                print(f"Invalid single_edge parts in: {raw_string}")

    return cg_new, GRAPH


def find_all_cycles(graph):
    """
    Find all cycles in the graph
    :param graph: Graph represented as a numpy matrix
    :return: List of all cycles
    """
    all_cycles = []
    visited = [False] * len(graph)

    for node in range(len(graph)):
        if not visited[node]:
            dfs_find_cycles(graph, node, visited, [], all_cycles)
    unique_cycles = []
    for cycle in all_cycles:
        if cycle not in unique_cycles:
            unique_cycles.append(cycle)

    return unique_cycles

def pc(data: ndarray,
    alpha=0.05,
    indep_test=fisherz,
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2,
    Label_name: list=[],
    Factor_list: list=[],
    var_name: dict={},
    mvpc: bool = False,
    correction_name: str = 'MV_Crtn_Fisher_Z',
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    if mvpc:  # missing value PC
        if indep_test == fisherz:
            indep_test = mv_fisherz
        return mvpc_alg(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test, correction_name=correction_name, stable=stable,
                        uc_rule=uc_rule, uc_priority=uc_priority, background_knowledge=background_knowledge,
                        verbose=verbose,
                        show_progress=show_progress, **kwargs)
    else:
        a=pc_alg(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule,
                      uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                      show_progress=show_progress, Label_list=Label_name, FACTOR_list=Factor_list, VARI_name=var_name,**kwargs)

        return a

def pc_alg(data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    Label_list: list=[],
    FACTOR_list: list=[],
    VARI_name: dict={},
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
            ["fisherz", "chisq", "gsq", "kci"]
           - "fisherz": Fisher's Z conditional independence test
           - "chisq": Chi-squared conditional independence test
           - "gsq": G-squared conditional independence test
           - "kci": Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """
    # Initialize some parameters.
    max_iterations = 2  # Maximum number of iterations
    iteration = 0  # Current iteration count
    min_data_size = 1000 # Termination condition
    edge_scores_list = []
    Final_cg= 0

    # # Split the dataset.
    train_indices, test_indices = train_test_split(np.arange(len(data)), test_size=0.20, random_state=1024)#42/1024/2024
    data_test=data[test_indices]
    GRAPH_todel =[]
    Statistic_uncertainty=[]
    ConFidence=[]
    while iteration < max_iterations:
        statistic_uncertainty = []
        Confidence ={}
        start = time.time()
        indep_test = CIT(data, chisq, **kwargs)
        print('data.shape:',data.shape,iteration)

        cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,FACTOR_list,
                                                    background_knowledge=background_knowledge, verbose=verbose,
                                                    show_progress=show_progress, node_names=node_names,var_names=VARI_name)
        if background_knowledge is not None:
            orient_by_background_knowledge(cg_1, background_knowledge)

        if uc_rule == 0:
            VERBS = ["provokes", " triggers", "leads to", "induces", "results in", "brings about", "yields",
                     "generates", "initiates", "produces", "stimulates", "instigates", "fosters", "engenders",
                     "promotes", "catalyzes", "gives rise to", "spurs", "sparks","causes"]
            verbs = VERBS[4]  # Adjust the index according to the requirements.
            if uc_priority != -1:
                cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
            else:
                cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
            cg = Meek.meek(cg_2, background_knowledge=background_knowledge)
            cg_new = deepcopy(cg)
            # LLM-assisted second stage of PC
            edge_jsons = build_dag_dict(cg_new.G.get_graph_edges(),FACTOR_list)
            Edge_jsons = deepcopy(edge_jsons)
            EDge_jsons = json.loads(Edge_jsons)
            single_edges = cg.find_fully_directed()


            for single_edge in single_edges:
                existing_score = next((item for item in edge_scores_list if 'single_edge' in item and  item['single_edge'] == single_edge), None)
                if existing_score is not None:
                    Edge_score = existing_score['edge_score']

                    print("Using existing score for edge:", single_edge, Edge_score)
                else:
                    second_prompt = llm_local_edge_check1(single_edge,FACTOR_list,VARI_name,verbs)
                    time.sleep(3)
                    edge_score = json_test(second_prompt)
                    # #for uncertainty
                    # second_prompt1 = llm_local_edge_check(single_edge, FACTOR_list, VARI_name, verbs)
                    # Uncertainty_list=get_top5_logits_for_matches(second_prompt1,model_name)
                    #
                    # UNCERTAINTY=calculate_uncertainty_list(Uncertainty_list)
                    # statistic_uncertainty.append(UNCERTAINTY)

                    print("second_score:", edge_score,single_edge,[FACTOR_list[single_edge[0]],FACTOR_list[single_edge[1]]])
                    edge_score,confidence = clean_json_string1(edge_score)
                    Confidence[single_edge]=[edge_score,confidence]
                    Edge_score=edge_score

                    edge_scores_list.append({
                        "single_edge": single_edge,
                        "edge_score": Edge_score
                    })

                max_key = max(Edge_score, key=Edge_score.get)
                edge0 = cg_new.G.get_edge(cg_new.G.nodes[single_edge[0]], cg_new.G.nodes[single_edge[1]])
                if edge0 is not None:
                    cg_new.G.remove_edge(edge0)

                if max_key=='REMOVE':
                    element_to_remove = FACTOR_list[single_edge[0]]

                    if element_to_remove in EDge_jsons[FACTOR_list[single_edge[1]]]["parents"]:
                        EDge_jsons[FACTOR_list[single_edge[1]]]["parents"].remove(element_to_remove)
                elif max_key=='FLIP':
                    EDge_jsons[FACTOR_list[single_edge[1]]]["parents"].remove(FACTOR_list[single_edge[0]])
                    element_to_add = FACTOR_list[single_edge[1]]
                    if FACTOR_list[single_edge[0]] not in EDge_jsons:
                        EDge_jsons[FACTOR_list[single_edge[0]]] = {"parents": []}
                    if element_to_add not in EDge_jsons[FACTOR_list[single_edge[0]]]["parents"]:
                        EDge_jsons[FACTOR_list[single_edge[0]]]["parents"].append(element_to_add)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[single_edge[1]], cg_new.G.nodes[single_edge[0]], Endpoint.TAIL, Endpoint.ARROW))
                elif max_key=='KEEP':
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[single_edge[0]], cg_new.G.nodes[single_edge[1]], Endpoint.TAIL, Endpoint.ARROW))
                    continue
                else:
                    raise ValueError("Error: No such edge exists.")

            # #for undirected edge
            undirected_edges = cg_new.find_undirected()
            unique_undirected_edges = set()
            for edge in undirected_edges:
                unique_undirected_edges.add(tuple(sorted(edge)))
            unique_undirected_edges = list(unique_undirected_edges)

            print('unique_undirected_edges:',unique_undirected_edges)
            for single_edge in unique_undirected_edges:
                existing_score = next((item for item in edge_scores_list if 'undirected_edges' in item and item['undirected_edges'] == single_edge), None)

                if existing_score is not None:
                    edge_score = existing_score['edge_score']
                    print("Using existing score for edge:", single_edge, edge_score)
                else:
                    second_prompt = llm_redirect1(single_edge,FACTOR_list,VARI_name,verbs)
                    time.sleep(3)
                    edge_score,confidence = redirect_test(second_prompt)
                    Confidence[single_edge] = [edge_score,confidence]


                    # # for uncertainty
                    # second_prompt1 = llm_redirect(single_edge, FACTOR_list, VARI_name, verbs)
                    # Uncertainty_list = get_top5_logits_for_matches(second_prompt1, model_name)
                    # UNCERTAINTY = calculate_uncertainty_list(Uncertainty_list)
                    # statistic_uncertainty.append(UNCERTAINTY)

                    edge_scores_list.append({
                        "undirected_edges": single_edge,
                        "edge_score": edge_score
                    })
                element_to_remove1 = FACTOR_list[single_edge[0]]
                element_to_remove2 = FACTOR_list[single_edge[1]]

                edge1 = cg_new.G.get_edge(cg_new.G.nodes[single_edge[0]], cg_new.G.nodes[single_edge[1]])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)

                if element_to_remove1 or element_to_remove2 in EDge_jsons[FACTOR_list[single_edge[1]]]["undirected_parents"]:
                    EDge_jsons[FACTOR_list[single_edge[1]]]["undirected_parents"].remove(element_to_remove1)
                    EDge_jsons[FACTOR_list[single_edge[0]]]["undirected_parents"].remove(element_to_remove2)
                if edge_score==1:
                    element_to_add = FACTOR_list[single_edge[0]]
                    if FACTOR_list[single_edge[1]] not in EDge_jsons:
                        EDge_jsons[FACTOR_list[single_edge[1]]] = {"parents": []}

                    if element_to_add not in EDge_jsons[FACTOR_list[single_edge[1]]]["parents"]:
                        EDge_jsons[FACTOR_list[single_edge[1]]]["parents"].append(element_to_add)

                    cg_new.G.add_edge(Edge(cg_new.G.nodes[single_edge[0]], cg_new.G.nodes[single_edge[1]], Endpoint.TAIL, Endpoint.ARROW))

                elif edge_score==0:
                    element_to_add = FACTOR_list[single_edge[1]]
                    if FACTOR_list[single_edge[0]] not in EDge_jsons:
                        # If the key does not exist, initialize it.
                        EDge_jsons[FACTOR_list[single_edge[0]]] = {"parents": []}
                    # Add specific elements to the "description" array of "node1".
                    if element_to_add not in EDge_jsons[FACTOR_list[single_edge[0]]]["parents"]:
                        EDge_jsons[FACTOR_list[single_edge[0]]]["parents"].append(element_to_add)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[single_edge[1]], cg_new.G.nodes[single_edge[0]], Endpoint.TAIL, Endpoint.ARROW))
                else:
                    raise ValueError("Error: No such edge exists.")
            # # for noedge
            # new_graph,_ = build_causal_graph(EDge_jsons,FACTOR_list,None)
            # noedge_list = find_nodes_without_edges(new_graph)
            # print('noedge_list:',noedge_list)
            # for single_edge in noedge_list:
            #     noedge_propmt = llm_local_noedge_check(single_edge, FACTOR_list)
            #     time.sleep(5)
            #     edge_score = json_test(noedge_propmt)
            #     try:
            #         edge_Json = json.loads(edge_score)
            #     except JSONDecodeError as e:
            #         edge_score = json_test(noedge_propmt)
            #         edge_score = fix_json_string(edge_score)
            #         edge_Json = json.loads(edge_score)
            #     # noedge_propmt = llm_local_noedge_check(single_edge,FACTOR_list)
            #     # edge_score = json_test(noedge_propmt)
            #     # edge_Json = json.loads(edge_score)
            #     print("second_score:", edge_Json)
            #     max_key = max(edge_Json, key=edge_Json.get)
            #     if max_key==single_edge[0]+" CAUSES "+single_edge[1]:
            #         element_to_add = single_edge[0]
            #         if single_edge[1] not in EDge_jsons:
            #             # If the key does not exist, initialize it.
            #             EDge_jsons[single_edge[1]] = {"parents": []}
            #         if element_to_add not in EDge_jsons[single_edge[1]]["parents"]:
            #             EDge_jsons[single_edge[1]]["parents"].append(element_to_add)
            #         cg_new.G.add_edge(Edge(cg_new.G.nodes[FACTOR_list.index(single_edge[0])], cg_new.G.nodes[FACTOR_list.index(single_edge[1])], Endpoint.TAIL, Endpoint.ARROW))
            #     elif max_key==single_edge[1]+" CAUSES "+single_edge[0]:
            #         element_to_add = single_edge[1]
            #         # FACTOR_list[single_edge[0]]
            #         if single_edge[0] not in EDge_jsons:
            #             EDge_jsons[single_edge[0]] = {"parents": []}
            #         if element_to_add not in EDge_jsons[single_edge[0]]["parents"]:
            #             EDge_jsons[single_edge[0]]["parents"].append(element_to_add)
            #         cg_new.G.add_edge(Edge(cg_new.G.nodes[FACTOR_list.index(single_edge[1])], cg_new.G.nodes[FACTOR_list.index(single_edge[0])], Endpoint.TAIL, Endpoint.ARROW))
            #     elif max_key =="NO DIRECT CAUSALITY":
            #         continue
            #     else:
            #         raise ValueError("Error: No such edge exists.")

            zero_edge_causalgraph = clear_default_edges(cg_new)

            after_graph,Final_causalgraph = build_causal_graph(EDge_jsons,FACTOR_list,zero_edge_causalgraph)
            del zero_edge_causalgraph


            ##Determine if there is a cycle, and then use the LLM to remove the cycle
            all_cycles_np = find_all_cycles(Final_causalgraph.G.graph)
            cycle_list = []
            for cycle in all_cycles_np:
                cycle = [FACTOR_list[node_id] for node_id in cycle]
                second_prompt = llm_decycle(cycle,FACTOR_list,VARI_name)
                time.sleep(3)
                cycle_score = json_test(second_prompt)
                cycle_list.append(cycle_score)
            #Remove cycles, then update the causal graph.
            FINAL_causalgraph, GRAPH= process_edges(cycle_list, FACTOR_list, Final_causalgraph.G.graph, cg_new)
            mblist = GetFMB(GRAPH,FACTOR_list)
            indices = [FACTOR_list.index(x) for x in mblist]
            try:
                new_ratio,new_data= evaluate_models(data,data[:, 10],indices,train_indices, data_test) #10/19 for huaxi ,asia 3, child -2
                print('iteration:',iteration,'new_ratio:',indices,new_ratio,new_data.shape[0])

                data=new_data
                train_indices=np.arange(len(data))
                del new_data

            except ValueError:
                new_ratio =  {
                "Recall": 0,
                "Overall Accuracy": 0,
                "Accuracy for True Label 1": 0,
                "Precision": 0,
                "rfRecall": 0,
                "rfOverall Accuracy": 0,
                "rfAccuracy for True Label 1": 0,
                "rfprecision": 0,
                "mlpRecall": 0,
                "mlpOverall Accuracy": 0,
                "mlpAccuracy for True Label 1": 0,
                "mlpprecision": 0}
                data= data
                train_indices = np.arange(len(data))
                print( 'new_ratio:', new_ratio)



            iteration += 1



            Final_cg = FINAL_causalgraph#Final_causalgraph/cg_new/FINAL_causalgraph
            GRAPH_todel.append(Final_cg)
            # Statistic_uncertainty.append(statistic_uncertainty)
            ConFidence.append(Confidence)


        elif uc_rule == 1:
            if uc_priority != -1:
                cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
            else:
                cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
            cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

        elif uc_rule == 2:
            if uc_priority != -1:
                cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
            else:
                cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
            cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
            cg = Meek.meek(cg_before, background_knowledge=background_knowledge)

        else:
            raise ValueError("uc_rule should be in [0, 1, 2]")
        end = time.time()
        print('iteration:', iteration)
        cg.PC_elapsed = end - start
        # Termination condition
        if data.shape[0] <= min_data_size:
            print("New1 data row count is less than or equal to 100. Terminating...")
            break
    return Final_cg,GRAPH_todel,new_ratio,iteration,ConFidence


def mvpc_alg(
    data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    correction_name: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs,
) -> CausalGraph:
    """
    Perform missing value Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha :  float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : str, name of the test-wise deletion independence test being used
            ["mv_fisherz", "mv_g_sq"]
            - mv_fisherz: Fisher's Z conditional independence test
            - mv_g_sq: G-squared conditional independence test (TODO: under development)
    correction_name : correction_name: name of the missingness correction
            [MV_Crtn_Fisher_Z, MV_Crtn_G_sq, MV_DRW_Fisher_Z, MV_DRW_G_sq]
            - "MV_Crtn_Fisher_Z": Permutation based correction method
            - "MV_Crtn_G_sq": G-squared conditional independence test (TODO: under development)
            - "MV_DRW_Fisher_Z": density ratio weighting based correction method (TODO: under development)
            - "MV_DRW_G_sq": G-squared conditional independence test (TODO: under development)
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge: background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    ## Step 1: detect the direct causes of missingness indicators
    prt_m = get_parent_missingness_pairs(data, alpha, indep_test, stable)
    # print('Finish detecting the parents of missingness indicators.  ')

    ## Step 2:
    ## a) Run PC algorithm with the 1st step skeleton;
    cg_pre = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                  background_knowledge=background_knowledge,
                                                  verbose=verbose, show_progress=show_progress, node_names=node_names)
    if background_knowledge is not None:
        orient_by_background_knowledge(cg_pre, background_knowledge)

    cg_pre.to_nx_skeleton()
    # print('Finish skeleton search with test-wise deletion.')

    ## b) Correction of the extra edges
    cg_corr = skeleton_correction(data, alpha, correction_name, cg_pre, prt_m, stable)
    # print('Finish missingness correction.')

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_corr, background_knowledge)

    ## Step 3: Orient the edges
    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_corr, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_corr, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_corr, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_corr, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_corr, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


#######################################################################################################################
## *********** Functions for Step 1 ***********
def get_parent_missingness_pairs(data: ndarray, alpha: float, indep_test, stable: bool = True) -> Dict[str, list]:
    """
    Detect the parents of missingness indicators
    If a missingness indicator has no parent, it will not be included in the result
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """
    parent_missingness_pairs = {'prt': [], 'm': []}

    ## Get the index of missingness indicators
    missingness_index = get_missingness_index(data)

    ## Get the index of parents of missingness indicators
    # If the missingness indicator has no parent, then it will not be collected in prt_m
    for missingness_i in missingness_index:
        parent_of_missingness_i = detect_parent(missingness_i, data, alpha, indep_test, stable)
        if not isempty(parent_of_missingness_i):
            parent_missingness_pairs['prt'].append(parent_of_missingness_i)
            parent_missingness_pairs['m'].append(missingness_i)
    return parent_missingness_pairs


def isempty(prt_r) -> bool:
    """Test whether the parent of a missingness indicator is empty"""
    return len(prt_r) == 0


def get_missingness_index(data: ndarray) -> List[int]:
    """Detect the parents of missingness indicators
    :param data: data set (numpy ndarray)
    :return:
    missingness_index: list, the index of missingness indicators
    """

    missingness_index = []
    _, ncol = np.shape(data)
    for i in range(ncol):
        if np.isnan(data[:, i]).any():
            missingness_index.append(i)
    return missingness_index


def detect_parent(r: int, data_: ndarray, alpha: float, indep_test, stable: bool = True) -> ndarray:
    """Detect the parents of a missingness indicator
    :param r: the missingness indicator
    :param data_: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    : return:
    prt: parent of the missingness indicator, r
    """
    ## TODO: in the test-wise deletion CI test, if test between a binary and a continuous variable,
    #  there can be the case where the binary variable only take one value after deletion.
    #  It is because the assumption is violated.

    ## *********** Adaptation 0 ***********
    # For avoid changing the original data
    data = data_.copy()
    ## *********** End ***********

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    ## *********** Adaptation 1 ***********
    # data
    ## Replace the variable r with its missingness indicator
    ## If r is not a missingness indicator, return [].
    data[:, r] = np.isnan(data[:, r]).astype(float)  # True is missing; false is not missing
    if sum(data[:, r]) == 0 or sum(data[:, r]) == len(data[:, r]):
        return np.empty(0)
    ## *********** End ***********

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.set_ind_test(CIT(data, indep_test.method))

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:

            ## *********** Adaptation 2 ***********
            # the skeleton search
            ## Only test which variable is the neighbor of r
            if x != r:
                continue
            ## *********** End ***********

            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    ## *********** Adaptation 3 ***********
    ## extract the parent of r from the graph
    cg.to_nx_skeleton()
    cg_skel_adj = nx.to_numpy_array(cg.nx_skel).astype(int)
    prt = get_parent(r, cg_skel_adj)
    ## *********** End ***********

    return prt


def get_parent(r: int, cg_skel_adj: ndarray) -> ndarray:
    """Get the neighbors of missingness indicators which are the parents
    :param r: the missingness indicator index
    :param cg_skel_adj: adjacency matrix of a causal skeleton
    :return:
    prt: list, parents of the missingness indicator r
    """
    num_var = len(cg_skel_adj[0, :])
    indx = np.array([i for i in range(num_var)])
    prt = indx[cg_skel_adj[r, :] == 1]
    return prt


## *********** END ***********
#######################################################################################################################

def skeleton_correction(data: ndarray, alpha: float, test_with_correction_name: str, init_cg: CausalGraph, prt_m: dict,
                        stable: bool = True) -> CausalGraph:
    """Perform skeleton discovery
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param test_with_correction_name: name of the independence test being used
           - "MV_Crtn_Fisher_Z": Fisher's Z conditional independence test
           - "MV_Crtn_G_sq": G-squared conditional independence test
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]

    ## *********** Adaption 1 ***********
    no_of_var = data.shape[1]

    ## Initialize the graph with the result of test-wise deletion skeletion search
    cg = init_cg

    if test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]:
        cg.set_ind_test(CIT(data, "mc_fisherz"))
    # No need of the correlation matrix if using test-wise deletion test
    cg.prt_m = prt_m
    ## *********** Adaption 1 ***********

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    return cg


#######################################################################################################################

# *********** Evaluation util ***********

def get_adjacancy_matrix(g: CausalGraph) -> ndarray:
    return nx.to_numpy_array(g.nx_graph).astype(int)


def matrix_diff(cg1: CausalGraph, cg2: CausalGraph) -> (float, List[Tuple[int, int]]):
    adj1 = get_adjacancy_matrix(cg1)
    adj2 = get_adjacancy_matrix(cg2)
    count = 0
    diff_ls = []
    for i in range(len(adj1[:, ])):
        for j in range(len(adj2[:, ])):
            if adj1[i, j] != adj2[i, j]:
                diff_ls.append((i, j))
                count += 1
    return count / 2, diff_ls




