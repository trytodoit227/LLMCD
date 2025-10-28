<h1 align="center">Causal Discovery through Synergizing Large Language Model
and Data-Driven Reasoning</h1>


## Preparation

### Environment Setup

We mainly use the following key libraries:

```
causal_learn==0.1.4.0
numpy==1.24.4
openai==1.3.7
scikit-learn==1.3.2
torch==2.1.1
python==3.8
```

Interact with LLMs using your own API key.:

```
client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
stream = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{"role": "user", "content": prompt}],
    stream=False,
    top_p=0.7,
    temperature=0.1
)
select_factors = stream.choices[0].message.content
```

### Datasets

The experimental data are located under the DATA folder.
- asia.bif and child.bif store the causal graph information of the benchmark datasets
- The .csv file stores the sample information in the benchmark dataset, that is, the values of the variables.
- The .xlsx file stores the MIMIC data.
-  The processed WCHSU data can be accessed through this [link](https://docs.google.com/spreadsheets/d/1t3LeQlA53QhzpaOVdZ894sU_SWiYYv4n/edit?usp=sharing&ouid=110872687347041304938&rtpof=true&sd=true).

### Files
- `MIMIC_variable_filtering` file contains the correlations between the initial variables of MIMIC and the target variable (lung cancer variable).
- `vari_name.py` file contains the descriptive information of the variables in the dataset.

## Experiments

### Pre-experiment operations

- Installing the causal-learn library.
- Replace the `pc.py` file in the `causallearn/search/ConstraintBased` directory with the `pc.py` file. Place the `causal.py` file under the `causallearn/utils` directory. Replace the `SkeletonDiscovery.py` file in the `causallearn/utils/PCUtils` directory with the `SkeletonDiscovery.py` file.


- Configure the appropriate API.

### Running the LLM-CD Model

```
python deal_data.py
```
## Reference
"Causal Discovery through Synergizing Large Language Model and Data-Driven Reasoning", KDD2025
