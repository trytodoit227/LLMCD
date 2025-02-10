import langid, re, os
import openai,torch
from openai import OpenAI
import networkx as nx
import re, json, time
import numpy as np
from causallearn.graph.Edge import Edge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.metrics import precision_score
import re


api_key = "your api_key"
api_base = "https://api.rcouyi.com/v1"
openai.api_key = api_key
openai.api_base = api_base

def propt_test(data, var_name):
    prompt = """If you are an expert in statistics and medicine, now, please do your best to complete the following task.

    # Variable Description\n"""
    for id in data[:2]:
        causal_var = var_name[id]
        prompt += f'''  {id}: {causal_var.description}\n'''
        if data[2] != []:
            for j in data[2]:
                causal_var = var_name[j]
                prompt += f'''  {j}: {causal_var.description}\n'''

    prompt1 = """\n   # Task: Conditional Independence Judgment.

    **Are Variable {data_list0} and Variable {data_list1} independent given the condition {data_list2}?**

    - Based on your observation and analysis of the provided data, combined with your experiential knowledge, score the independence of Variable {data_list0} and Variable {data_list1} under the given condition {data_list2}.
    - The score should reflect the probability that the stated independence holds.
    - This is a closed system, only considering the variables involved in {data_list}.
    - The numerical values following the variables represent the values of the variables. When assessing the independence between two variables, be sure to consider the influence of these variable values.

    # About Output

    In this part, you can think through the problem step by step, but you do not need to output your thought process.
    Hint: Based on your prior knowledge, you need to score whether the conditional independence holds or not, and the score can only be from [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].

    In this section, you need to report the final scoring situation.
    - The score should be chosen from the interval of 0.1-1.0. The higher the value, the greater the likelihood that the conditional independence holds.
    - A score of 0.1 indicates that the probability of conditional independence holding is very low; 1 indicates that conditional independence definitely holds.
    - Only consider the variables involved in {data_list}.
    - If the condition {data_list2} is empty, consider only the independence of Variable {data_list0} and Variable {data_list1}.

    ```
    Your final output should follow **this template: **
    **"___"**
    ```
    """.format(data_list=data, data_list0=data[0], data_list1=data[1], data_list2=data[2])
    Prompt = prompt + prompt1
    return Prompt

def propt_test1(data, var_name):
    prompt = """If you are an expert in statistics and medicine, now, please do your best to complete the following task.

    # Variable Description\n"""
    for id in data[:2]:
        causal_var = var_name[id]
        prompt += f'''  {id}: {causal_var.description}\n'''
        if data[2] != []:
            for j in data[2]:
                causal_var = var_name[j]
                prompt += f'''  {j}: {causal_var.description}\n'''

    prompt1 = """\n   # Task: Conditional Independence Judgment.

    **Are Variable {data_list0} and Variable {data_list1} independent given the condition {data_list2}?**

    - Based on your observation and analysis of the provided data, combined with your experiential knowledge, score the independence of Variable {data_list0} and Variable {data_list1} under the given condition {data_list2}.
    - The score should reflect the probability that the stated independence holds.
    - This is a closed system, only considering the variables involved in {data_list}.
    - The numerical values following the variables represent the values of the variables. When assessing the independence between two variables, be sure to consider the influence of these variable values.

    # About Output

    In this part, you can think through the problem step by step, but you do not need to output your thought process.
    Hint: Based on your prior knowledge, you need to score whether the conditional independence holds or not, and the score can only be from [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].

    In this section, you need to report the final scoring situation and your confidence in this answer.
    - The score should be chosen from the interval of 0.1-1.0. The higher the value, the greater the likelihood that the conditional independence holds.
    - A score of 0.1 indicates that the probability of conditional independence holding is very low; 1 indicates that conditional independence definitely holds.
    - Only consider the variables involved in {data_list}.
    - If the condition {data_list2} is empty, consider only the independence of Variable {data_list0} and Variable {data_list1}.
    - The confidence indicates how likely you think your answer is true.

    ```
    Your final output should follow **this template: **
    **"Answer and Confidence: ___, ___"**
    ```
    """.format(data_list=data, data_list0=data[0], data_list1=data[1], data_list2=data[2])
    Prompt = prompt + prompt1
    return Prompt


def individual_test(prompt):
    retries = 3  # Set the number of retries.
    for i in range(retries):
        try:
            client2 = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
            stream = client2.chat.completions.create(
                model='gpt-4o-mini',# gpt-3.5-turbo-1106,gpt-4o.gpt-4o-mini,o1-preview,deepseek-chat,llama2-70b-4096,llama-3-70b
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                top_p=0.7,
                temperature=0.1
            )
            select_factors = stream.choices[0].message.content
            individual_score = extract_number(select_factors)
            return individual_score
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5)
    print('Retry failed!')
    return None



def llm_redirect(data, FACTOR_list, var_name, verbs):
    prompt = """If you are an expert in statistics and medicine, now, please do your best to complete the following task.
    # Variable Description\n"""
    for id in data:
        causal_var = var_name[FACTOR_list[id]]
        prompt += f'''  {FACTOR_list[id]}: {causal_var.description}\n'''

    prompt1 = """\n   # Task: Redirect the Undirected Edge.
    **Based on your expertise, please redirect the undirected edge between two nodes, node {data_edge0} and node {data_edge1} **
    - These variables are diagnostic indicators for lung cancer patients.
    - This is a closed system, only considering the variables involved in {data_list}.
    - The numerical values following the variables represent the corresponding values of the variables. When determining the causal relationship between two variables, it is essential to consider the influence of the variable values.


    # About Output
    Based on your prior knowledge, you can think through the problem step by step, 
    and then direct the undirected edge between {data_edge0} and {data_edge1}.

    In this section, you need to report the final direction situation.
    - {data_edge0} {verbs} {data_edge1}: "Changing {data_edge0} {verbs} a change in {data_edge1} in a causal manner."
    - {data_edge1} {verbs} {data_edge0}: "Changing {data_edge1} {verbs} a change in {data_edge0} in a causal manner."
    - Do not output your thought process, only output the probability corresponding to each option in JSON format.
    - Maintain the integrity of variable names.
    - The provided answer is reasonable and accurate, correlation between two variables does not imply causality.

    ## Examples:
    Input: please redirect the undirected edge between two nodes, node smoke over 80 and node lung cancer
    0utput: {{"smoke CAUSES lung cancer": 90, "lung cancer CAUSES smoke": 10}}

    Based on the information and examples above, think step by step and answer the following questions.
    Input: please redirect the undirected edge between two nodes, node {data_edge0} and node {data_edge1}
    0utput: 
    """
    prompt1 = prompt1.format(data_list=[FACTOR_list[data[0]], FACTOR_list[data[1]]], data_edge0=FACTOR_list[data[0]],
                             data_edge1=FACTOR_list[data[1]], verbs=verbs)  # yao gai!!!
    Prompt = prompt + prompt1

    # ## Examples:
    # Input: please redirect the undirected edge between two nodes, node Ages over 80 and node lang cancer
    # =>"The value is: 1."
    # Input: please redirect the undirected edge between two nodes, node {data_edge0} and node {data_edge1}
    # =>
    return Prompt

def llm_redirect1(data, FACTOR_list, var_name, verbs):
    prompt = """If you are an expert in statistics and medicine, now, please do your best to complete the following task.

    # Variable Description\n"""
    for id in data:
        causal_var = var_name[FACTOR_list[id]]
        prompt += f'''  {FACTOR_list[id]}: {causal_var.description}\n'''

    prompt1 = """\n   # Task: Redirect the Undirected Edge.
    **Based on your expertise, please redirect the undirected edge between two nodes, node {data_edge0} and node {data_edge1} **
    - These variables are diagnostic indicators for lung cancer patients.
    - This is a closed system, only considering the variables involved in {data_list}.
    - The numerical values following the variables represent the corresponding values of the variables. When determining the causal relationship between two variables, it is essential to consider the influence of the variable values.


    # About Output
    Based on your prior knowledge, you can think through the problem step by step, 
    and then direct the undirected edge between {data_edge0} and {data_edge1}.

    In this section, you need to report the final direction situation and your confidence in this answer.
    - If {data_edge0} {verbs} {data_edge1}, then output value 1.
    - if {data_edge1} {verbs} {data_edge0}, then output value 0.
    - Do not output your thought process, only output the final value 0 or 1.
    - Maintain the integrity of variable names.
    - The provided answer is reasonable and accurate, correlation between two variables does not imply causality.
    - The confidence indicates how likely you think your answer is true.

    ```
    Your final output should follow **this template: **
    **"Answer and Confidence: ___, ___"**
    ```  
    ## Examples:
    Input: please redirect the undirected edge between two nodes, node Ages over 80 and node lang cancer
    0utput: Answer and Confidence: 1, 0.85 

    Based on the information and examples above, think step by step and answer the following questions.
    Input: please redirect the undirected edge between two nodes, node {data_edge0} and node {data_edge1}
    0utput: Answer and Confidence: ___, ___
    """
    prompt1 = prompt1.format(data_list=[FACTOR_list[data[0]], FACTOR_list[data[1]]], data_edge0=FACTOR_list[data[0]],
                             data_edge1=FACTOR_list[data[1]], verbs=verbs)  # yao gai!!!
    Prompt = prompt + prompt1
    # ## Examples:
    # Input: please redirect the undirected edge between two nodes, node Ages over 80 and node lang cancer
    # =>"The value is: 1."
    # Input: please redirect the undirected edge between two nodes, node {data_edge0} and node {data_edge1}
    # =>
    return Prompt


def redirect_test(prompt):
    retries = 3
    for i in range(retries):
        try:
            client2 = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
            stream = client2.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                top_p=0.7,
                temperature=0.1
            )
            select_factors = stream.choices[0].message.content
            individual_score = extract_number1(select_factors)
            return individual_score
        except Exception as e:
            print(f"Connection error: {e}. Retrying...")
            time.sleep(5)
    print('Retry failed!')
    return None


def extract_number(input_string):
    """
    Extract the numerical values from the input string.
    Parameters:
    input_string (str): The input string containing a numerical value.
    Returns:
    float: The extracted numerical value, or None if no numerical value is found.
    """
    match = re.search(r"[-+]?\d*\.\d+|\d+", input_string)
    if match:
        return float(match.group())
    else:
        return None

def extract_number1(input_string):
    """
    Extract two numbers from the input string: one is the number between the colon (:) and the comma (,),
    and the other is the number following the comma.

    Returns:
    float: The extracted numerical value, or None if no numerical value is found.
    """
    match1 = re.search(r'(\d+(\.\d+)?)\s*,', input_string)
    if match1:
        first_number = float(match1.group(1))
    else:
        first_number = None
    match2 = re.search(r',\s*(\d+(\.\d+)?)', input_string)
    if match2:
        second_number = float(match2.group(1))
    else:
        second_number = None

    return first_number, second_number


def json_test(prompt):
    retries = 3
    for i in range(retries):
        try:
            client2 = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
            stream = client2.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                top_p=0.7,
                temperature=0.1
            )
            select_factors = stream.choices[0].message.content
            return select_factors
        except Exception as e:
            print(f"Connection error: {e}. Retrying...")
            time.sleep(5)
    print('Retry failed!')
    return None


def build_dag_dict(edge_list, node_list):
    dag_dict = {}
    # Iterate over the Edge objects
    for edge in edge_list:
        # Extract the endpoints from the edge
        endpoint1 = edge.endpoint1
        endpoint2 = edge.endpoint2

        # Extract node indices from the node names
        parent1_index = int(edge.node1.name[1:]) - 1  # 'X1' -> 1 -> 0 (zero-based index)
        parent2_index = int(edge.node2.name[1:]) - 1  # 'X2' -> 2 -> 1 (zero-based index)

        # Map node indices to corresponding names from node_list
        parent1_name = node_list[parent1_index]
        parent2_name = node_list[parent2_index]

        # Determine the direction of the edge
        if endpoint1.name == 'TAIL' and endpoint2.name == 'ARROW':
            parent = parent1_name
            child = parent2_name
            is_directed = True
            is_bi_directional = False
            is_undirected = False
        elif endpoint1.name == 'ARROW' and endpoint2.name == 'TAIL':
            parent = parent2_name
            child = parent1_name
            is_directed = True
            is_bi_directional = False
            is_undirected = False
        elif endpoint1.name == 'ARROW' and endpoint2.name == 'ARROW':
            parent1 = parent1_name
            parent2 = parent2_name
            child = None
            is_directed = False
            is_bi_directional = True
            is_undirected = False
        else:
            parent1 = parent1_name
            parent2 = parent2_name
            child = None
            is_directed = False
            is_bi_directional = False
            is_undirected = True

        # Update the dictionary for directed edges
        if is_directed:
            if child not in dag_dict:
                dag_dict[child] = {"parents": [], "bi_directional_parents": [], "undirected_parents": []}
            dag_dict[child]["parents"].append(parent)
        # Update the dictionary for bidirectional edges
        elif is_bi_directional:
            if parent1 not in dag_dict:
                dag_dict[parent1] = {"parents": [], "bi_directional_parents": [], "undirected_parents": []}
            if parent2 not in dag_dict:
                dag_dict[parent2] = {"parents": [], "bi_directional_parents": [], "undirected_parents": []}
            dag_dict[parent1]["bi_directional_parents"].append(parent2)
            dag_dict[parent2]["bi_directional_parents"].append(parent1)
        # Update the dictionary for undirected edges
        elif is_undirected:
            if parent1 not in dag_dict:
                dag_dict[parent1] = {"parents": [], "bi_directional_parents": [], "undirected_parents": []}
            if parent2 not in dag_dict:
                dag_dict[parent2] = {"parents": [], "bi_directional_parents": [], "undirected_parents": []}
            dag_dict[parent1]["undirected_parents"].append(parent2)
            dag_dict[parent2]["undirected_parents"].append(parent1)

    # Output the resulting dictionary in JSON format
    return json.dumps(dag_dict, indent=4)

def llm_local_noedge_check(edge_data, FACTOR_list, var_name, verbs):
    prompt = """You are an expert in medical statistics with a sub-specialist interest in causal inference. Your task is to explore the hypothesis space of
    the relationships that might exist between a number of variables relating to lung cancer.

    #Variable Description\n"""
    for id in edge_data:
        causal_var = var_name[FACTOR_list[id]]
        prompt += f'''  {FACTOR_list[id]}: {causal_var.description}\n'''

    prompt1 = """\n   # Task
    Please think through the process step by step to arrive at the correct answer, and use the following options to represent the probabilities of different relationships:
    - NO DIRECT CAUSALITY: "There is no direct causal relationship between the {pair0} and {pair1}."
    - {pair0} {verbs} {pair1}: "Changing {pair0} {verbs} a change in {pair1} in a causal manner."
    - {pair1} {verbs} {pair0}: "Changing {pair1} {verbs} a change in {pair0} in a causal manner."

    #About Output
    In this section, you need to report the final probability distribution.
    - Output the probability corresponding to each option in JSON format. 
    - Do not provide any explanation; simply output the result in standard JSON format, without any extra characters, otherwise it will not be processed.
    - The output must be in JSON format, and it should not trigger a JSONDecodeError when processed with the `json.loads` command.
    - Each option corresponds to a probability value between 0 and 1, with the sum of all probability values equal to 1.
    - Maintain the integrity of variable names.
    - The provided answer is reasonable and accurate, correlation between two variables does not imply causality.

    ## Examples:
    Input: Assume there is no prior causal relationship between the variables Ages over 80 and Ages 70-75
    Output: {{"NO DIRECT CAUSALITY": 98, "Ages over 80 CAUSES Ages 70-75": 1, "Ages 70-75 CAUSES Ages over 80": 1}}
    Input: Assume there is no prior causal relationship between the variables lung cancer and Ages 75-80
    Output: {{"NO DIRECT CAUSALITY": 7, "Ages 75-80 CAUSES lung cancer": 92, "lung cancer CAUSES Ages 75-80": 1}}
    Input: Assume there is no prior causal relationship between the variables {pair0} and {pair1}
    Output: 
    """
    prompt1 = prompt1.format(pair0=edge_data[0], pair1=edge_data[1], verbs=verbs)
    Prompt = prompt + prompt1

    # ## Examples:
    # Input: Assume there is no prior causal relationship between the variables Ages over 80 and Ages 70-75
    # Output: {{"NO DIRECT CAUSALITY": 98, "Ages over 80 CAUSES Ages 70-75": 1, "Ages 70-75 CAUSES Ages over 80": 1}}
    # Input: Assume there is no prior causal relationship between the variables lung cancer and Ages 75-80
    # Output: {{"NO DIRECT CAUSALITY": 7, "Ages 75-80 CAUSES lung cancer": 92, "lung cancer CAUSES Ages 75-80": 1}}
    # Input: Assume there is no prior causal relationship between the variables {pair0} and {pair1}
    # Output:

    return Prompt

def llm_local_noedge_check1(edge_data, FACTOR_list, var_name, verbs):
    prompt = """You are an expert in medical statistics with a sub-specialist interest in causal inference. Your task is to explore the hypothesis space of
    the relationships that might exist between a number of variables relating to lung cancer.

    #Variable Description\n"""
    for id in edge_data:
        causal_var = var_name[FACTOR_list[id]]
        prompt += f'''  {FACTOR_list[id]}: {causal_var.description}\n'''

    prompt1 = """\n   # Task
    Please think through the process step by step to arrive at the correct answer, and use the following options to represent the probabilities of different relationships:
    - NO DIRECT CAUSALITY: "There is no direct causal relationship between the {pair0} and {pair1}."
    - {pair0} {verbs} {pair1}: "Changing {pair0} {verbs} a change in {pair1} in a causal manner."
    - {pair1} {verbs} {pair0}: "Changing {pair1} {verbs} a change in {pair0} in a causal manner."

    #About Output
    In this section, you need to report the final probability distribution and your confidence in this answer.
    - Output the probability corresponding to each option in JSON format. 
    - Do not provide any explanation; simply output the result in standard JSON format, without any extra characters, otherwise it will not be processed.
    - The output must be in JSON format, and it should not trigger a JSONDecodeError when processed with the `json.loads` command.
    - Each option corresponds to a probability value between 0 and 1, with the sum of all probability values equal to 1.
    - Maintain the integrity of variable names.
    - The provided answer is reasonable and accurate, correlation between two variables does not imply causality.
    - The confidence indicates how likely you think your answer is true

    ## Examples:
    Input: Assume there is no prior causal relationship between the variables Ages over 80 and Ages 70-75
    Output: Answer and Confidence: {{"NO DIRECT CAUSALITY": 98, "Ages over 80 CAUSES Ages 70-75": 1, "Ages 70-75 CAUSES Ages over 80": 1}}, 0.76
    Input: Assume there is no prior causal relationship between the variables lung cancer and Ages 75-80
    Output: Answer and Confidence: {{"NO DIRECT CAUSALITY": 7, "Ages 75-80 CAUSES lung cancer": 92, "lung cancer CAUSES Ages 75-80": 1}},0.68
    Input: Assume there is no prior causal relationship between the variables {pair0} and {pair1}
    Output: Answer and Confidence: ___, ___
    """
    prompt1 = prompt1.format(pair0=edge_data[0], pair1=edge_data[1], verbs=verbs)
    Prompt = prompt + prompt1
    # ## Examples:
    # Input: Assume there is no prior causal relationship between the variables Ages over 80 and Ages 70-75
    # Output: {{"NO DIRECT CAUSALITY": 98, "Ages over 80 CAUSES Ages 70-75": 1, "Ages 70-75 CAUSES Ages over 80": 1}}
    # Input: Assume there is no prior causal relationship between the variables lung cancer and Ages 75-80
    # Output: {{"NO DIRECT CAUSALITY": 7, "Ages 75-80 CAUSES lung cancer": 92, "lung cancer CAUSES Ages 75-80": 1}}
    # Input: Assume there is no prior causal relationship between the variables {pair0} and {pair1}
    # Output:

    return Prompt

def llm_local_edge_check(edge_data, FACTOR_list, var_name, verbs):
    prompt = """You are an expert in medical statistics with a sub-specialist interest in causal inference. Your task is to explore the hypothesis space of
    the relationships that might exist between a number of variables relating to lung cancer.

    #Variable Description\n"""
    for id in edge_data:
        causal_var = var_name[FACTOR_list[id]]
        prompt += f'''  {FACTOR_list[id]}: {causal_var.description}\n'''

    prompt1 = """\n   # Task
    Please think through the process step by step to arrive at the correct answer, and use the following options to represent the probabilities of different relationships:
    - KEEP: "{pair0} {verbs} {pair1} is correct and the current direction should be kept."
    - FLIP: "FLIP the direction of this relationship. There is evidence to suggest that it is plausible that changing {pair1} {verbs} a change to {pair0}." 
    - REMOVE: "Remove this relationship completely; that is, no causal relationship exists between {pair0} and {pair1}."

    # About Output
    In this section, you need to report the final probability distribution.
    - The sum of all probabilities should be equal to 1.
    - There is no causal relationship between the variables related to age.
    - Do not provide any explanation; simply output the result in standard JSON format, without any extra characters, such as '```' and 'json', otherwise it will not be processed.
    - Maintain the integrity of variable names.
    - The provided answer is reasonable and accurate, correlation between two variables does not imply causality.

    ## Examples:
    Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 70-75], where lung cancer causes Ages 70-75. Please adjust and refine based on your knowledge and experience.
    Output: {{"KEEP": 1, "FLIP": 96, "REMOVE": 3}}

    Based on the information and examples above, think step by step and answer the following questions.
    Input: Assume there is a pre-existing causal relationship {pair}, where {pair0} causes {pair1}. Please adjust and refine based on your knowledge and experience.
    Output: 
    """
    prompt1 = prompt1.format(pair=[FACTOR_list[edge_data[0]], FACTOR_list[edge_data[1]]],
                             pair0=FACTOR_list[edge_data[0]],
                             pair1=FACTOR_list[edge_data[1]], verbs=verbs)
    Prompt = prompt + prompt1
    # for huaxi
    # Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 70-75], where lung cancer causes Ages 70-75. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 96, "REMOVE": 3}}
    # Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 75-80], where lung cancer causes Ages 75-80. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 98, "REMOVE": 1}}
    # Input: Assume there is a pre-existing causal relationship [Ages 65-70,Ages over 80], where Ages 65-70 causes Ages over 80. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 3, "REMOVE": 96}}
    # Input: Assume there is a pre-existing causal relationship [Daily smoking 30-60 cigare, Annual smoking more than 20-29 packs], where Daily smoking 30-60 cigare causes Annual smoking more than 20-29 packs. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 1, "REMOVE": 98}}

    # for mimic
    # Input: Assume there is a pre-existing causal relationship [smoker0,smoker1], where smoker0 causes smoker1. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 3, "REMOVE": 96}}
    # Input: Assume there is a pre-existing causal relationship [metastatic_solid_tumor 0, metastatic_solid_tumor 1], where metastatic_solid_tumor 0 causes metastatic_solid_tumor 1. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 1, "REMOVE": 98}}
    # Input: Assume there is a pre-existing causal relationship [admission_ages 45.01-62.31, admission_ages 27.65-45.01], where admission_ages 45.01-62.31 causes admission_ages 27.65-45.01. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 3, "REMOVE": 96}}
    # Input: Assume there is a pre-existing causal relationship [smoking 1, smoking 0], where smoking 1 causes smoking 0. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 1, "REMOVE": 98}}
    return Prompt

def llm_local_edge_check1(edge_data, FACTOR_list, var_name, verbs):
    prompt = """You are an expert in medical statistics with a sub-specialist interest in causal inference. Your task is to explore the hypothesis space of
    the relationships that might exist between a number of variables relating to lung cancer.

    #Variable Description\n"""
    for id in edge_data:
        causal_var = var_name[FACTOR_list[id]]
        prompt += f'''  {FACTOR_list[id]}: {causal_var.description}\n'''

    prompt1 = """\n   # Task
    Please think through the process step by step to arrive at the correct answer, and use the following options to represent the probabilities of different relationships:
    - KEEP: "{pair0} {verbs} {pair1} is correct and the current direction should be kept."
    - FLIP: "FLIP the direction of this relationship. There is evidence to suggest that it is plausible that changing {pair1} {verbs} a change to {pair0}." 
    - REMOVE: "Remove this relationship completely; that is, no causal relationship exists between {pair0} and {pair1}."

    # About Output
    In this section, you need to report the final probability distribution and your confidence in this answer.
    - The sum of all probabilities should be equal to 1.
    - There is no causal relationship between the variables related to age.
    - Do not provide any explanation; simply output the result in standard JSON format, without any extra characters, such as '```' and 'json', otherwise it will not be processed.
    - Maintain the integrity of variable names.
    - The provided answer is reasonable and accurate, correlation between two variables does not imply causality.
    - The confidence indicates how likely you think your answer is true

    ## Examples:
    Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 70-75], where lung cancer causes Ages 70-75. Please adjust and refine based on your knowledge and experience.
    Output: Answer and Confidence: {{"KEEP": 1, "FLIP": 96, "REMOVE": 3}}, 0.54

    Based on the information and examples above, think step by step and answer the following questions.
    Input: Assume there is a pre-existing causal relationship {pair}, where {pair0} causes {pair1}. Please adjust and refine based on your knowledge and experience.
    Output: Answer and Confidence: ___, ___
    """
    prompt1 = prompt1.format(pair=[FACTOR_list[edge_data[0]], FACTOR_list[edge_data[1]]],
                             pair0=FACTOR_list[edge_data[0]],
                             pair1=FACTOR_list[edge_data[1]], verbs=verbs)
    Prompt = prompt + prompt1
    # for huaxi
    # Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 70-75], where lung cancer causes Ages 70-75. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 96, "REMOVE": 3}}
    # Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 75-80], where lung cancer causes Ages 75-80. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 98, "REMOVE": 1}}
    # Input: Assume there is a pre-existing causal relationship [Ages 65-70,Ages over 80], where Ages 65-70 causes Ages over 80. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 3, "REMOVE": 96}}
    # Input: Assume there is a pre-existing causal relationship [Daily smoking 30-60 cigare, Annual smoking more than 20-29 packs], where Daily smoking 30-60 cigare causes Annual smoking more than 20-29 packs. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 1, "REMOVE": 98}}

    # for mimic
    # Input: Assume there is a pre-existing causal relationship [smoker0,smoker1], where smoker0 causes smoker1. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 3, "REMOVE": 96}}
    # Input: Assume there is a pre-existing causal relationship [metastatic_solid_tumor 0, metastatic_solid_tumor 1], where metastatic_solid_tumor 0 causes metastatic_solid_tumor 1. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 1, "REMOVE": 98}}
    # Input: Assume there is a pre-existing causal relationship [admission_ages 45.01-62.31, admission_ages 27.65-45.01], where admission_ages 45.01-62.31 causes admission_ages 27.65-45.01. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 3, "REMOVE": 96}}
    # Input: Assume there is a pre-existing causal relationship [smoking 1, smoking 0], where smoking 1 causes smoking 0. Please adjust and refine based on your knowledge and experience.
    # Output: {{"KEEP": 1, "FLIP": 1, "REMOVE": 98}}
    return Prompt


def llm_decycle(cycle_data, FACTOR_list, var_name):
    prompt = """You are an expert in medical statistics with a sub-specialist interest in causal inference. Your task is to explore the hypothesis space of
    the relationships that might exist between a number of variables relating to lung cancer.

    #Variable Description\n"""
    for id in cycle_data:
        causal_var = var_name[id]
        prompt += f'''  {id}: {causal_var.description}\n'''

    prompt1 = """\n   # Task
    These variables  in {pair} represent the medical characteristics of patients, each element is considered a node, and with arrows pointing from left to right, they form a cycle. Given this cycle,
    based on your expertise in lung cancer, please analyze this cycle and propose how to modify the connections to eliminate the cycle.

    # About Output
    Remove edges between variables that do not have causal relationships, or reverse the causal direction between variables to make the causal relationships more reasonable, in order to eliminate cycles. 
    - Do not provide any explanation.
    - Simply output the result in standard template, without any extra characters, such as '```' and 'json', otherwise it will not be processed.

    Your final output template should be **one of the following options:**
    **Remove: {pair0} and {pair1}**
    **Reverse: {pair1} and {pair2}**

    ## Examples:
    Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 70-75], where lung cancer causes Ages 70-75
    Output: {{"KEEP": 1, "FLIP": 96, "REMOVE": 3}}

    Based on the information and examples above, think step by step and answer the following questions.
    Input: Here is a cycle {pair}. Please either remove or reverse an edge to break the cycle.
    Output:
    """
    prompt1 = prompt1.format(pair=cycle_data, pair0=cycle_data[0], pair1=cycle_data[1],
                             pair2=cycle_data[2])
    Prompt = prompt + prompt1
    # ## Examples:
    # Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 70-75], where lung cancer causes Ages 70-75
    # Output: {{"KEEP": 1, "FLIP": 96, "REMOVE": 3}}
    # Input: Assume there is a pre-existing causal relationship [lung cancer,Ages 75-80], where lung cancer causes Ages 75-80
    # Output: {{"KEEP": 1, "FLIP": 98, "REMOVE": 1}}
    # Input: Assume there is a pre-existing causal relationship [Ages 65-70,Ages over 80], where Ages 65-70 causes Ages over 80
    # Output: {{"KEEP": 1, "FLIP": 3, "REMOVE": 96}}
    # Input: Assume there is a pre-existing causal relationship {pair}, where {pair0} causes {pair1}
    # Output:

    return Prompt


def find_nodes_without_edges(cg):
    nodes = list(cg.nodes)
    no_edge_pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]

            if (not cg.has_edge(node1, node2)) and (not cg.has_edge(node2, node1)):
                no_edge_pairs.append((node1, node2))
    return no_edge_pairs


def build_causal_graph(data, FACTOR_list, zero_edge_causalgraph):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    # for node, attributes in data.items():
    #     G.add_node(node, **attributes)
    for node in FACTOR_list:
        G.add_node(node)

    # Helper function to parse the description field
    def parse_description(description):
        if isinstance(description, str):
            # description = description.strip("List of parents of node. Think about what variables of can affect it")
            Description = re.search(r"\[.*?\]", description)
            try:
                # return eval(Description)
                return eval(Description.group())
            except:
                return []
        elif isinstance(description, list):
            return description
        return []

    # Add directed edges based on "parents"
    for node, attributes in data.items():
        parents = attributes.get('parents', {})  # .get('description', [])
        if 'description' in parents:
            description = parents.get('description', [])
            parents = parse_description(description)
        else:
            parents = parse_description(parents)
        # parents = parse_description(parents)
        if not isinstance(parents, list):
            parents = [parents]
        for parent in parents:
            G.add_edge(parent, node)
            if zero_edge_causalgraph is not None:
                edgee = Edge(GraphNode('X{}'.format(FACTOR_list.index(parent) + 1)),
                             GraphNode('X{}'.format(FACTOR_list.index(node) + 1)), Endpoint.TAIL, Endpoint.ARROW)
                zero_edge_causalgraph.G.add_edge(edgee)

    # Add bidirectional edges based on "bi_directional_parents"
    for node, attributes in data.items():
        bi_parents = attributes.get('bi_directional_parents', {})

        if 'description' in bi_parents:
            description = bi_parents.get('description', [])
            bi_parents = parse_description(description)
        else:
            bi_parents = parse_description(bi_parents)

        if not isinstance(bi_parents, list):
            bi_parents = [bi_parents]
        if bi_parents == []:
            continue

        for bi_parent in bi_parents:
            G.add_edge(bi_parent, node)
            G.add_edge(node, bi_parent)
            if zero_edge_causalgraph is not None:
                edgee = Edge(GraphNode('X{}'.format(FACTOR_list.index(bi_parent) + 1)),
                             GraphNode('X{}'.format(FACTOR_list.index(node) + 1)), Endpoint.ARROW, Endpoint.ARROW)
                zero_edge_causalgraph.G.add_edge(edgee)

    # Add undirected edges based on "undirected_parents"
    for node, attributes in data.items():
        undirected_parents = attributes.get('undirected_parents', {})

        if 'description' in undirected_parents:
            description = undirected_parents.get('description', [])
            undirected_parents = parse_description(description)
        else:
            undirected_parents = parse_description(undirected_parents)

        if not isinstance(undirected_parents, list):
            undirected_parents = [undirected_parents]
        if undirected_parents == []:
            continue

        for undirected_parent in undirected_parents:
            G.add_edge(undirected_parent, node)
            G.add_edge(node, undirected_parent)
            if zero_edge_causalgraph is not None:
                edgee = Edge(GraphNode('X{}'.format(FACTOR_list.index(undirected_parent) + 1)),
                             GraphNode('X{}'.format(FACTOR_list.index(node) + 1)), Endpoint.TAIL, Endpoint.TAIL)
                zero_edge_causalgraph.G.add_edge(edgee)
    return G, zero_edge_causalgraph


def clear_default_edges(causal_graph):
    edges = causal_graph.G.get_graph_edges()
    for edge in edges:
        causal_graph.G.remove_edge(edge)
    return causal_graph



def GetFMB(G, node_name, target_node=19):  # for huaxi: 10/19; for mimic: 17,asia 3,child -2
    mbset = []
    d = G.shape[0]
    pa = lambda x: [idx for idx in range(d) if G[idx, x] not in [1, 0]]
    mbset += pa(target_node)
    if target_node in mbset:
        mbset.remove(target_node)

    return np.array(node_name)[list(set(mbset))]

def evaluate_models(data, labels, selected_columns, train_indices, test_data):
    """
    Evaluate the Logistic Regression and MLP models, and compute recall and accuracy.

    Parameters:
    - data: Input data, numpy array
    - labels: True labels, numpy array
    - selected_columns: List of selected column indices used for training

    Returns:
    - Logistic Regression model's recall, overall accuracy, accuracy for true label 1, and accuracy for true label 0
    - MLP model's recall, overall accuracy, accuracy for true label 1, and accuracy for true label 0
    """
    # Select the variables corresponding to specific columns as the target nodes for input.
    X = data[:, selected_columns]
    y = labels

    X_train = X[train_indices]
    X_test = test_data[:, selected_columns]
    y_train = y[train_indices]
    y_test = test_data[:, 19]  # 10/19 for huaxi,mimic-iv 17,asia 3,child -2



    del X

    # # Set class weights for imbalanced data.
    class_weights = {0: 1, 1: 100}
    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=4000, class_weight=class_weights)
    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)
    logistic_recall = recall_score(y_test, logistic_predictions)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    logistic_pre = precision_score(y_test, logistic_predictions)
    logistic_accuracy_1 = accuracy_score(y_test[y_test == 1], logistic_predictions[y_test == 1])


    # SVM
    rf_model = RandomForestClassifier(n_estimators=100,
                                      random_state=42,
                                      class_weight=class_weights,
                                      max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    rf_recall = recall_score(y_test, y_pred)
    rf_accuracy = accuracy_score(y_test, y_pred)
    rf_pre = precision_score(y_test, y_pred)
    rf_accuracy_1 = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])

    # MLP
    # Oversampling using SMOTE. for huaxi data
    # smote = SMOTE(random_state=2024, k_neighbors=5, sampling_strategy=0.3)
    # # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    mlp_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=1024)  # 42/2024
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    mlp_recall = recall_score(y_test, mlp_predictions)
    mlp_accuracy = accuracy_score(y_test, mlp_predictions)
    mlp_pre = precision_score(y_test, mlp_predictions)
    mlp_accuracy_1 = accuracy_score(y_test[y_test == 1], mlp_predictions[y_test == 1])
    # mlp_accuracy_0 = accuracy_score(y_test[y_test == 0], mlp_predictions[y_test == 0])

    # Select samples with incorrect predictions.
    train_predictions = rf_model.predict(X_train)
    # train_predictions = mlp_model.predict(X_train)

    # Using a mask to filter out the samples that were predicted incorrectly.
    mask = train_predictions != y_train

    incorrect_samples = data[train_indices[mask]]
    del data
    return {
               "Recall": f"{logistic_recall:.4f}",
               "Overall Accuracy": f"{logistic_accuracy:.4f}",
               "Accuracy for True Label 1": f"{logistic_accuracy_1:.4f}",
               "Precision": f"{logistic_pre:.4f}",
               "rfRecall": f"{rf_recall:.4f}",
               "rfOverall Accuracy": f"{rf_accuracy:.4f}",
               "rfAccuracy for True Label 1": f"{rf_accuracy_1:.4f}",
               "rfprecision": f"{rf_pre:.4f}",
               "mlpRecall": f"{mlp_recall:.4f}",
               "mlpOverall Accuracy": f"{mlp_accuracy:.4f}",
               "mlpAccuracy for True Label 1": f"{mlp_accuracy_1:.4f}",
               "mlpprecision": f"{mlp_pre:.4f}"
           }, incorrect_samples


def fix_json_string(raw_string):
    raw_string = raw_string.replace("'", '"')
    parts = raw_string.split(',')
    for i in range(len(parts)):
        second_part = parts[i].strip()
        match = re.search(r'^(.*)(:)([^:]*)$', second_part)
        if match:
            before_colon, colon, after_colon = match.groups()
            after_colon = after_colon.replace('"', '')
            if not before_colon.endswith('"'):
                second_part = f'{before_colon}"{colon}{after_colon}'
            else:
                second_part = f'{before_colon}{colon}{after_colon}'

        parts[i] = second_part
    fixed_string = ','.join(parts)

    return fixed_string


def ring(data):
    prompt = '''If you were an expert in causal inference, you would approach the task as follows:

Given a graph {graph} with edge {edges}, identify all the cyclic structures within it and then perform reverse operations on each edge in the cyclic structures to eliminate the cycles. Here's a step-by-step guide on how to achieve this:

1.Identify All Cyclic Structures:

Cycle Detection: Use algorithms to find all cycles within the graph. Common algorithms for cycle detection include Depth-First Search (DFS) based methods or algorithms specifically designed for detecting cycles in directed graphs, such as Tarjan’s or Johnson’s algorithm.
2.Analyze Each Cycle:

For each detected cycle, examine the edges involved in the cycle. A cycle is a path where the starting node and ending node are the same, and it includes at least one other node and edge.
3.Reverse Operations on Edges:

To break the cycles, you need to perform reverse operations on the edges. This typically means reversing the direction of one or more edges within the cycle. The goal is to modify the graph such that there are no longer any cycles.
4.Implement the Changes:

Edge Reversal: Select an edge from each cycle and reverse its direction. Ensure that after reversing, the graph no longer contains cycles. In practice, you might reverse multiple edges if necessary.
Verify Cycle Elimination: After performing edge reversals, reapply the cycle detection algorithm to ensure that all cycles have been removed.
    '''.format(graph=data, edges=data.get_graph_edges())
    return prompt


def clean_json_string(s):
    """
    Remove the leading and trailing ```json\n and \n``` from the string if they exist.

    Parameters:
    s (str): The input string

    Returns:
    str: The processed string
    """
    if '```' in s:
        return s.strip('```').replace('json\n', '').strip('\n')
    else:
        return s

def clean_json_string1(s):
    if '```' in s:
        clean_s = s.strip('```').replace('json\n', '').strip('\n')
    else:
        clean_s = s

    # Extract the dictionary content using regular expressions (e.g., {"KEEP": 90, "FLIP": 5, "REMOVE": 5}).
    dict_match = re.search(r'\{.*\}', clean_s)
    if dict_match:
        dict_str = dict_match.group(0)
        p = json.loads(dict_str)
    else:
        p = None
    after_comma_number = re.search(r'},\s*([\d\.]+)', clean_s)
    if after_comma_number:
        conf = float(after_comma_number.group(1))
    else:
        conf = None

    return p, conf
