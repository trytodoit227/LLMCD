import openai,os
from openai import OpenAI
from efficiency.nlp import Chatbot
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from openai import OpenAI
from efficiency.nlp import Chatbot
from vari_name import *
from metrics import compute_metrics
import time

from datetime import datetime
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import mv_fisherz,fisherz,kci,chisq
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd
import numpy as np

import bnlearn as bn
from utils import translate_to_english,check,causal_discovery,\
    extract_floats,Cut,extract_smoking,extract_water,extract_daixie,\
    extract_yanghuamei,filter_bins_by_proportion_for_all_columns,extract_selected_data,\
    create_adjacency_matrix,normalize_dict_values,plot_uncertainty_distribution,\
    calculate_uncertainty_list

dag = bn.import_DAG(f'./asia.bif')#asia/child
var_names = dag['adjmat'].columns
adj_mat = dag['adjmat'].to_numpy().astype(int)


file_bin = './variable_filtering.txt'#for huaxi

selected_data, selected_indices = extract_selected_data(file_bin)
ids_to_remove = [11,14]#FOR NODE 16 [11,14] FOR NODE 52 [23,28,36] for mimic []
Selected_indices = [item for i, item in enumerate(selected_indices) if i not in ids_to_remove]
print("Selected Data List:", selected_data)
print("Selected Indices:", Selected_indices)

var_name=VAR_NAMES_AND_DESC['huaxi16']#Load the variable descriptions from the dataset.
var_list = []
for var in var_name:
    causal_var = var_name[var]
    var_list.append(causal_var.description)

alpha =0.05

# file_path_WCHSU = './output_mimic.xlsx'  # for mimic
file_path_WCHSU = './wchsu16.xlsx'
df = pd.read_excel(file_path_WCHSU, nrows=100)# Just for extracting variable names.

# # for WCHSU
factor_meaning = df.columns.tolist()
factor_meaning= [s.replace(":", "") for s in factor_meaning]
Factor_list = [factor_meaning[u] for u in Selected_indices]

label_name = df.columns.tolist()
label_name= [s.replace(":", "") for s in label_name]
Label_name = [label_name[j] for j in Selected_indices]

# # # # for asia/child
# file_path_standard = './asia_data_10000.csv'
# df = pd.read_csv(file_path_standard, nrows=100)
# Data = df.iloc[1:, :]
# factor_meaning = df.columns.tolist()
# factor_meaning= [s.replace(":", "") for s in factor_meaning]
# Factor_list = factor_meaning

# label_name = df.columns.tolist()
# label_name= [s.replace(":", "") for s in label_name]
# Label_name = label_name



# # for MIMIC
# Data = df.iloc[1:, :]
# factor_meaning = df.iloc[0].tolist()
# factor_meaning= [s.replace(":", "") for s in factor_meaning]
# Factor_list = factor_meaning

# label_name = df.columns.tolist()
# label_name= [s.replace(":", "") for s in label_name]
# Label_name = label_name


# ##for WCHSU
def load_data(file_path, selected_indices):
    df = pd.read_excel(file_path, nrows=160000)
    Data = df.iloc[0:, selected_indices]
    Data2 = Data.to_numpy(dtype=float)
    # Ablation experiments based on sample data size
    # Randomly select K rows from the data.
    selected_rows = np.random.choice(Data2.shape[0], size=160000, replace=False)
    New_data = Data2[selected_rows, :]
    del Data
    return New_data


if __name__ == "__main__":
    file_path1 = file_path_WCHSU # Path to your Excel file
    Data=load_data(file_path1, Selected_indices)

# # #for mimic
# def load_data(file_path):
#     df = pd.read_excel(file_path)
#     Data = df.iloc[1:, :]
#     Data2 = Data.to_numpy(dtype=float)
#     return Data2
# if __name__ == "__main__":
#     file_path1 = file_path_mimic  # Path to your Excel file
#     Data=load_data(file_path1)


# #for asia/child
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     Data = df.iloc[:, :]
#     Data2 = Data.to_numpy(dtype=float)
#     #Ablation experiments based on sample data size
#     #Randomly select K rows from the data.
#     selected_rows = np.random.choice(Data2.shape[0], size=10000, replace=False)
#     New_data = Data2[selected_rows, :]
#     return New_data
# if __name__ == "__main__":
#     file_path1 = file_path_standard  # Path to your Excel file
#     Data=load_data(file_path1)

## Our LLM-CD model
cg = pc(Data, alpha, chisq, True, 0, 2,Label_name, Factor_list,var_name,mvpc=False)

# Just for Asia and Child
predicted_adj_mat = create_adjacency_matrix(cg[0],len(Factor_list))
metrics = compute_metrics(adj_mat,predicted_adj_mat)
pdy = GraphUtils.to_pydot(cg[0].G, labels=Factor_list)
pdy.write_png(r'bin_1006mpc3few{}03.png'.format('0.001'))





