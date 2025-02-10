import langid,re
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import mv_fisherz,fisherz,kci,chisq
from causallearn.utils.GraphUtils import GraphUtils
import numpy as np
import pandas as pd

api_key = "sk-zQDMqI1Mi4AGnqHQ9cDb0b28Ce4b4891A8AdEe48007e1bDe"
# api_key ='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'
api_base ="https://api.rcouyi.com/v1"  # "https://api.rcouyi.com/v1/chat/completions"
openai.api_key = api_key
openai.api_base = api_base

#将因子中的中文转为英文，便于LLM输出影响因子
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

def translate_to_english(text):
    try:
        content = f"Translate the following Chinese text to English:\n\n{text}"
        client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
        stream = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{"role": "user", "content": content}],
            stream=False, )
        return stream.choices[0].message.content
        # response = openai.chat.completions.create(
        #     model='gpt-3.5-turbo', #"text-davinci-002",
        #     messages=[
        #         {"role": "user", "content": content}
        #     ],
        #     stream=False,
        #     top_p=0.7,
        #     temperature=0.1
        # )
        # # print(response)
        # valid_content = response.choices[0].message.content


        # return valid_content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def check(text):
    language = detect_language(text)
    print("Detected language:", language)

    # Translate if it's Chinese
    if language == 'zh':
        english_translation = translate_to_english(text)
        print("English Translation:", english_translation)
        return english_translation
    elif language == 'en':
        print("Text is already in English:", text)
        return text

def extract_floats(value):
    if isinstance(value, str):  # 检查值是否为字符串
        numbers = re.findall(r'(\d+\.\d+|\d+)', value)  # 正则表达式查找所有数字（整数或浮点数）
        if numbers:  # 如果找到数字
            return float(numbers[0])  # 转换第一个找到的数字为浮点数并返回
    return value  # 如果不是字符串或没有找到数字，返回原始值

# 定义一个函数检查元素是否为字符串
def is_string(x):
    return isinstance(x, str)


# 定义转换函数处理’吸烟‘
def extract_smoking(data):
    if pd.isna(data):  # 检查数据是否为NaN
        return np.nan
    if isinstance(data, str):  # 确保数据是字符串类型
        if '戒烟' in data or data == '无':
            return 0
        elif '吸烟' in data or '支' in data:
            # 尝试提取数字，如果失败返回NaN
            # nums = pd.to_numeric(data.split('吸烟')[-1].rstrip('年'), errors='coerce')
            match = re.search(r'(\d+)支', data)
            if match:
                # 返回数字部分
                return int(match.group(1))
    return np.nan  # 如果数据不是字符串，返回NaN

# 定义转换函数处理’身体总水分‘
def extract_water(value):
    # 首先检查值是否为浮点数和NaN
    if pd.isna(value):
        return 0
    elif isinstance(value, str) and 'L' in value:
        return float(value.replace('L', ''))
    elif isinstance(value, str):
        return float(value)
    return value  # 返回原值，如果以上条件都不满足

# 定义转换函数处理’基础代谢率‘
def extract_daixie(value):
    # 首先检查值是否为浮点数和NaN
    if pd.isna(value):
        return 0
    elif isinstance(value, str) and 'kcal' in value:
        return float(value.replace('kcal', ''))
    elif isinstance(value, str):
        return float(value)
    return value  # 返回原值，如果以上条件都不满足

# 定义转换函数处理’髓过氧化物酶‘

def extract_yanghuamei(value):
    if pd.isna(value) or value == 'nan':
        return 0
    elif isinstance(value, str):
        # 使用正则表达式提取数字
        match = re.search(r'\d+\.?\d*', value)
        return float(match.group()) if match else 0
    else:
        return value

def Cut(data):
    for column in data.columns[16:]:#对部分列的值进行分组
        # 使用q=4将每列数据分为四个等量分箱
        # errors='coerce'在存在非数值型数据时生成NaN
        # duplicates='drop'用于避免因为分位数重复而出错
        # data[column] = pd.qcut(data[column], q=4, duplicates='drop', labels=False)
        # 等宽分箱
        bins = 4  # 指定分箱的数量
        labels = [i for i in range(bins)]  # 为每个分箱指定一个标签
        data[column] = pd.cut(data[column], bins=bins, labels=labels)
    return data

def causal_discovery(data,iter):
    #deal  data str-->value
    data = data.applymap(extract_floats)
    # 映射字典
    mapping = {'有': 1, '无': 0,'偶尔':np.nan, '不吸烟': np.nan,'nan': np.nan,'男': 1, '女': 0, '不吸烟': np.nan,'nan': np.nan}

    # 应用映射
    data = data.replace(mapping)
    data = data.apply(pd.to_numeric, errors='coerce')
    Data = data.to_numpy(dtype=float)

    # #检查是否数据中还含有字符串
    # # 应用这个函数到 DataFrame 的每列
    # string_mask = Data.applymap(is_string)
    #
    # # 检查每列是否包含字符串
    # columns_with_strings = string_mask.any()

    cg = pc(Data,0.05,chisq,True,0,-1,mvpc=True)
    # visualization

    pdy = GraphUtils.to_pydot(cg.G)
    # pdy.write_png(r'D:\wenxian\model\异质图神经网络最新进展\图神经网络\大模型可解释性\大模型和因果推断\数据\simple_test.png')
    pdy.write_png(r'simple_test{}.png'.format(iter))

def filter_bins_by_proportion_for_all_columns(data, start_col, n_bins, threshold):
    """
    对DataFrame进行处理，其中指定列进行等宽分箱和过滤，其他列进行普通的one-hot编码。

    参数:
    - data: 输入的DataFrame。
    - start_col: 开始执行分箱操作的列索引。
    - n_bins: 分箱的数量。
    - threshold: 保留的样本比例阈值。

    返回:
    - result_df: 处理后的DataFrame。
    """
    # 处理0到start_col之前的列，保留原始值作为分箱标签进行one-hot编码
    df_one_hot = pd.DataFrame()
    for column in data.columns[:start_col]:
        # 使用列的原始值作为分箱标签
        df_temp = pd.get_dummies(data[column], prefix=column, dtype=int)
        df_one_hot = pd.concat([df_one_hot, df_temp], axis=1)
    df_one_hot = pd.concat([df_one_hot, data.iloc[:, start_col:start_col+1]], axis=1)

    # 处理start_col之后列到最后的列，执行等宽分箱和过滤
    for column in data.columns[start_col+1:]:
        if data[column].nunique() <= n_bins:
            # 如果唯一值数量小于或等于分箱数，则直接进行one-hot编码
            df_temp = pd.get_dummies(data[column], prefix=column, dtype=int)
        else:
            # 等宽分箱
            # labels = pd.cut(data[column], bins=n_bins, labels=False)
            bins = pd.cut(data[column], bins=n_bins, labels=False, duplicates='drop', retbins=True)

            # 获取分箱的结果和区间
            labels = bins[0]
            intervals = bins[1]
            print('+++---')
            print('column:',column)
            print('intervals:',intervals)
            print('******')

            # 生成one-hot编码
            df_temp = pd.get_dummies(labels, prefix=column, dtype=int)

            # 计算每个one-hot编码列的非零值比例
            col_proportions = df_temp.sum() / len(df_temp)

            # 过滤出比例大于阈值的列
            cols_to_keep = col_proportions[col_proportions > threshold].index

            # 删除比例不足的列
            df_temp = df_temp[cols_to_keep]

        # 合并到主DataFrame
        df_one_hot = pd.concat([df_one_hot, df_temp], axis=1)

    return df_one_hot

def extract_selected_data(file_path, threshold=4):#raw =5
    """
    从文本文件中提取满足条件的数据行，并返回符合条件的数据及其索引。

    Args:
    file_path (str): 文本文件的路径。
    threshold (int): 检查数值的阈值，默认为5。

    Returns:
    tuple: 返回一个包含选中数据列表和相应索引列表的元组。
    """
    selected_data_list = []  # 存储符合条件的元素列表
    selected_indices = []  # 存储符合条件的索引列表
    data_list = []  # 存储所有的元素
    buffer = []  # 用于收集连续的非空行
    line_index = 0  # 初始化索引计数

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # 如果行不为空
                buffer.append(stripped_line.split())  # 添加分割后的行到缓冲区
            else:
                # 当遇到空行时，处理累积的数据
                if len(buffer) >= 2:  # 如果缓冲区有至少两行数据
                    second_element = buffer[1]
                    values = [int(x.replace(':', '')) for x in second_element if x.replace(':', '').isdigit()]
                    if any(value >= threshold for value in values):
                        selected_data_list.append(buffer[:2])  # 将当前满足条件的数据添加到选中列表
                        selected_indices.append(line_index)  # 记录满足条件的数据的索引

                    data_list.append(buffer[:2])  # 将所有的数据添加到总列表
                    line_index += 1  # 索引加一
                buffer = []  # 清空缓冲区以用于下一次数据的收集

        # 文件结束后，如果缓冲区仍有数据并且符合处理要求
        if buffer and len(buffer) >= 2:
            second_element = buffer[1]
            values = [int(x.replace(':', '')) for x in second_element if x.replace(':', '').isdigit()]
            if any(value >= threshold for value in values):
                selected_data_list.append(buffer[:2])
                selected_indices.append(line_index)

            data_list.append(buffer[:2])

    return selected_data_list, selected_indices

def normalize_dict_values(dict_list):
    """
    Normalize the values in a list of dictionaries based on the sum of values in each dictionary.

    Parameters:
    - dict_list (list of dict): A list of dictionaries where each dictionary contains numerical values.

    Returns:
    - normalized_list (list of list): A list of lists where each list contains the normalized values of the corresponding dictionary.
    """
    normalized_list = []

    for d in dict_list:
        # Step 1: Extract values from the dictionary
        values = list(d.values())

        # Step 2: Calculate the sum of the values
        total_sum = sum(values)

        # Step 3: Normalize the values by dividing each value by the sum
        if total_sum != 0:
            normalized_values = [v / total_sum for v in values]
        else:
            normalized_values = [0] * len(values)  # Handle the case where sum is 0 to avoid division by zero

        # Step 4: Append the normalized values as a list
        normalized_list.append(normalized_values)

    return normalized_list

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
    return uncertainties


def plot_uncertainty_distribution(uncertainties, save_path="uncertainty_distribution.pdf"):
    """
    Plot the uncertainty distribution and save as PDF.

    Parameters:
    - uncertainties (list): A list of uncertainty values.
    - save_path (str): The file path to save the PDF.
    """
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(uncertainties, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
    plt.title("Uncertainty Distribution", fontsize=16)
    plt.xlabel("Uncertainty", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(alpha=0.3)

    # Save to PDF
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"Uncertainty distribution saved to {save_path}")


# 使用函数示例

def individual_test(prompt):
    client2 = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
    stream = client2.chat.completions.create(
        model='gpt-4-0613',
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        top_p=0.7,
        temperature=0.1
    )
    select_factors = stream.choices[0].message.content
    return select_factors


import os
from pydot import Dot


def save_graph_as_png(graph, factor_list, output_path, file_suffix='0.001'):
    """
    将图对象和因子列表转换为 pydot 图，并保存为 PNG 格式。

    参数：
    - graph: 网络图对象，通常为 cg[0].G。
    - factor_list: 因子列表，用于节点标签。
    - output_path: 输出文件夹路径。
    - file_suffix: 文件后缀或标识符，默认为 '0.001'。

    返回值：
    - 输出图像的完整路径。
    """
    # 将图对象转换为 pydot 格式
    pdy = GraphUtils.to_pydot(graph, labels=factor_list)

    # 检查输出路径是否存在，不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 生成 PNG 文件名和完整路径
    output_file = os.path.join(output_path, f'bin_1006mpc3few{file_suffix}03.png')

    # 保存图像
    pdy.write_png(output_file)


def create_adjacency_matrix(edge_list, n):
    """
    Creates an adjacency matrix from a list of edges.

    Parameters:
    - edge_list: list of tuples, where each tuple represents an edge (node_id_1, node_id_2)
    - n: int, the number of nodes in the graph

    Returns:
    - adj_matrix: np.array, the adjacency matrix with 1's for edges and 0's elsewhere
    """
    # Initialize the adjacency matrix with zeros
    adj_matrix = np.zeros((n, n))

    # Fill the adjacency matrix based on edges in edge_list
    for edge in edge_list.G.get_graph_edges():
        i= int(edge.node1.name[1:]) - 1
        j = int(edge.node2.name[1:]) - 1
        adj_matrix[i][j] = 1

    return adj_matrix

#deal mimic iv
# prompt = '''Assume you are an expert in the field of lung cancer. Now, do your best.
#
# # Data
#
# {data_list}
#
# # Task: Factor Scoring.
#
# **Is the factor related to lung cancer?**
#
# - Based on your observation and analysis of the given data, score each factor in the {data_list} for its correlation.
# - The scores should reflect the correlation of each factor with the occurrence of lung cancer.
# - The factors do not interfere with each other.
# - When scoring, consider both the factor and the corresponding value comprehensively.
# - In each factor, the colon is immediately followed by the corresponding value.
#
# # About Output
#
# Your output should include the following sections.
#
# **Part 1**: Consideration Process.
#
# In this section, you are free to document your thought process.
# Hint: Based on your prior knowledge, you need to score each factor in the list, and each factor's score can only be [1, 2, 3, 4, 5].
#
# **Part 2**: Final Output.
#
# In this section, you need to report your scoring for each factor.
# - The score for each factor should be chosen from **[1, 2, 3, 4, 5]**.
# - Higher values indicate higher correlation.
# - Each factor in the {data_list} must be scored.
#
# Report the factors using **the following template: **
#
# ```
# **Factor Name:  **
# - 1: [Extremely weakly correlated]
# - 2: [Weak correlation].
# - 3: [Related].
# - 4: [Strongly correlated].
# - 5: [Highly correlated].
# ```
#
# '''.format(data_list=factor_name)
