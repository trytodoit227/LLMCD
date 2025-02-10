

class CausalVariable:
    def __init__(self, symbol, name, description):
        self.symbol = symbol
        self.name = name
        self.description = description

    def __repr__(self):
        return f"CausalVariable({self.name}, {self.description})"

    def __str__(self):
        return f"{self.name} ({self.description})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


ASIA_VAR_NAMES_AND_DESC = {
    "dysp" : CausalVariable("dysp", "dyspnoea", "whether or not the patient has dyspnoea, also known as shortness of breath"),
    "tub" : CausalVariable("tub", "tuberculosis", "whether or not the patient has tuberculosis"),
    "lung" : CausalVariable("lung", "lung cancer", "whether or not the patient has lung cancer"),
    "bronc" : CausalVariable("bronc", "bronchitis", "whether or not the patient has bronchitis"),
    "either" : CausalVariable("either", "either tuberculosis or lung cancer", "whether or not the patient has either tuberculosis or lung cancer"),
    "smoke" : CausalVariable("smoke", "smoking", "whether or not the patient is a smoker"),
    "asia" : CausalVariable("asia", "recent visit to asia", "whether or not the patient has recently visited asia"),
    "xray" : CausalVariable("xray", "positive chest xray", "whether or not the patient has had a positive chest xray"),
}

WCHSU16_VAR_NAMES_AND_DESC = {
    "Ages 65-70" : CausalVariable("Ages 65-70", "Ages 65-70", "The patient's age is 65-70"),
    "Ages 70-75" : CausalVariable("Ages 70-75", "Ages 70-75", "The patient's age is 70-75"),
    "Ages 75-80" : CausalVariable("Ages 75-80", "Ages 75-80", "The patient's age is 75-80"),
    "Ages over 80" : CausalVariable("Ages over 80", "Ages over 80", "The patient's age is over 80 years"),
    "History of malignant tumors" : CausalVariable("History of malignant tumors", "History of malignant tumors", "The patient has a history of malignant tumors."),
    "With chronic obstructive pulmonary disease" : CausalVariable("With chronic obstructive pulmonary disease", "With chronic obstructive pulmonary disease", "The patient has chronic obstructive pulmonary disease."),
    "Smoker" : CausalVariable("Smoker", "Smoker", "The patient has a smoking habit."),
    "Annual smoking 20-29 packs" : CausalVariable("Annual smoking 20-29 packs", "Annual smoking 20-29 packs", "The patient smokes 20-29 packs per year."),
    "Annual smoking more than 30 packs": CausalVariable("Annual smoking more than 30 packs", "Annual smoking more than 30 packs", "The patient smoke more than 30 packs per year."),
    "Not quit smoking": CausalVariable("Not quit smoking", "Not quit smoking", "The patient has not quit smoking"),
    "lung cancer": CausalVariable("lung cancer", "lung cancer", "The patient has lung cancer."),
    "Non-small cell lung cancer antigen 0-44.6": CausalVariable("Non-small cell lung cancer antigen 0-44.6", "Non-small cell lung cancer antigen 0-44.6", "The patient's Non-small cell lung cancer antigen level is 0-44.6."),
    "Carcinoembryonic antigen 0-188.1675": CausalVariable("Carcinoembryonic antigen 0-188.1675", "Carcinoembryonic antigen 0-188.1675", "The patient's carcinoembryonic antigen level is 0-188.1675."),
    "Melanoma Antigen A1 Immunoglobulin MAGE 0-7.375": CausalVariable("Melanoma Antigen A1 Immunoglobulin MAGE: 0-7.375", "Melanoma Antigen A1 Immunoglobulin MAGE: 0-7.375", "The patient's melanoma antigen A1 immunoglobulin (MAGE) level is 0-7.375."),
    "Daily smoking 30-60 cigare": CausalVariable("Daily smoking: 30-60 cigare", "Daily smoking: 30-60 cigare", "The patient smokes 30-60 cigarettes per day."),
    "Serum CA199 0-409.65": CausalVariable("Serum CA199 0-409.65", "Serum CA199 0-409.65", "The patient's serum CA199 is 0-409.65."),
}

WCHSU51_VAR_NAMES_AND_DESC = {
    "Male": CausalVariable("Male", "Male", "The patient's gender is male."),
    "Ages 60-65": CausalVariable("Ages 60-65", "Ages 60-65", "The patient's age is between 60 and 65."),
    "Ages 65-70" : CausalVariable("Ages 65-70", "Ages 65-70", "The patient's age is between 65 and 70."),
    "Ages 70-75" : CausalVariable("Ages 70-75", "Ages 70-75", "The patient's age is between 70 and 75."),
    "Ages 75-80" : CausalVariable("Ages 75-80", "Ages 75-80", "The patient's age is between 75 and 80."),
    "Ages over 80" : CausalVariable("Ages over 80", "Ages over 80", "The patient's age is over 80 years."),
    "Neutrophil-to-lymphocyte ratio ≥3.86": CausalVariable("Neutrophil-to-lymphocyte ratio ≥3.86", "Neutrophil-to-lymphocyte ratio ≥3.86","The patient's Neutrophil-to-Lymphocyte Ratio is greater than or equal to 3.86."),
    "Platelet-to-lymphocyte ratio ≥191": CausalVariable("Platelet-to-lymphocyte ratio ≥191", "Platelet-to-lymphocyte ratio ≥191", "The patient's platelet-to-lymphocyte ratio is greater than or equal to 191."),
    "Family history of lung cancer": CausalVariable("Family history of lung cancer", "Family history of lung cancer", "The patient's family members have cancer."),
    "History of malignant tumors" : CausalVariable("History of malignant tumors", "History of malignant tumors", "The patient has a history of malignant tumors."),
    "With chronic obstructive pulmonary disease" : CausalVariable("With chronic obstructive pulmonary disease", "With chronic obstructive pulmonary disease", "The patient has chronic obstructive pulmonary disease."),
    "Pulmonary fibrosis": CausalVariable("Pulmonary fibrosis", "Pulmonary fibrosis", "The patient has pulmonary fibrosis."),
    "With chronic lung disease": CausalVariable("With chronic lung disease", "With chronic lung disease", "The patient has chronic lung disease."),
    "Smoker" : CausalVariable("Smoker", "Smoker", "The patient has a smoking habit."),
    "Annual smoking 10-19 packs" : CausalVariable("Annual smoking 10-19 packs", "Annual smoking 10-19 packs", "The patient smokes 10-19 packs per year."),
    "Annual smoking 20-29 packs" : CausalVariable("Annual smoking 20-29 packs", "Annual smoking 20-29 packs", "The patient smokes 20-29 packs per year."),
    "Annual smoking more than 30 packs": CausalVariable("Annual smoking more than 30 packs", "Annual smoking more than 30 packs", "The patient smoke more than 30 packs per year."),
    "Not quit smoking": CausalVariable("Not quit smoking", "Not quit smoking", "The patient has not quit smoking."),
    "Quit smoking for less than 15 years": CausalVariable("Quit smoking for less than 15 yearsg", "Quit smoking for less than 15 yearsg", "The patient has quit smoking for no more than 15 years."),
    "lung cancer": CausalVariable("lung cancer", "lung cancer", "The patient has lung cancer."),
    "With pulmonary nodules": CausalVariable("With pulmonary nodules", "With pulmonary nodules", "The patient has pulmonary nodules."),
    "Albumin 0-15.8": CausalVariable("Albumin 0-15.8", "Albumin 0-15.8", "The patient's Albumin level is 0-15.8."),
    "Monocyte percentage 10.5-15.75": CausalVariable("Monocyte percentage 10.5-15.75", "Monocyte percentage 10.5-15.75", "The patient's monocyte percentage is 10.5-15.75."),
    "Non-small cell lung cancer antigen 0-44.6": CausalVariable("Non-small cell lung cancer antigen 0-44.6", "Non-small cell lung cancer antigen 0-44.6", "The patient's Non-small cell lung cancer antigen level is 0-44.6"),
    "Carcinoembryonic antigen 0-188.1675": CausalVariable("Carcinoembryonic antigen 0-188.1675", "Carcinoembryonic antigen 0-188.1675", "The patient's carcinoembryonic antigen level is 0-188.1675."),
    "RBC distribution width CV 14-21": CausalVariable("RBC distribution width CV 14-21", "RBC distribution width CV 14-21", "The coefficient of variation of the patient's red blood cell distribution width is 14-21."),
    "Serum CA125 0-516.75": CausalVariable("Serum CA125 0-516.75", "Serum CA125 0-516.75", "The patient's serum CA125 level is 0-516.75."),
    "Melanoma Antigen A1 Immunoglobulin MAGE 0-7.375": CausalVariable("Melanoma Antigen A1 Immunoglobulin MAGE 0-7.375", "Melanoma Antigen A1 Immunoglobulin MAGE 0-7.375", "The patient's Melanoma Antigen A1 Immunoglobulin MAGE is 0-7.375."),
    "Platelet Count 294.75-582.5": CausalVariable("Platelet Count 294.75-582.5", "Platelet Count 294.75-582.5", "The patient's platelet count is 294.75-582.5."),
    "Lactate Dehydrogenase 0-881.25": CausalVariable("Lactate Dehydrogenase 0-881.25", "Lactate Dehydrogenase 0-881.25", "The patient's lactate dehydrogenase level is 0-881.25."),
    "Weight 75-112.5": CausalVariable("Weight 75-112.5", "Weight 75-112.5", "The patient's weight is 75-112.5 kg."),
    "Waist-to-hip ratio 0.875-1.5": CausalVariable("Waist-to-hip ratio 0.875-1.5", "Waist-to-hip ratio 0.875-1.5", "The patient's waist-to-hip ratio is 0.875-1.5."),
    "Daily smoking 0-30 cigarettes": CausalVariable("Daily smoking 0-30 cigarettes", "Daily smoking 0-30 cigarettes", "The patient’s daily smoking amount is 0-30 cigarettes."),
    "Daily smoking 30-60 cigare": CausalVariable("Daily smoking 30-60 cigare", "Daily smoking 30-60 cigare", "The patient’s daily smoking amount is 30-60 cigarettes."),
    "Serum CA199 0-409.65": CausalVariable("Serum CA199 0-409.65", "Serum CA199 0-409.65", "The patient's serum CA199 is 0-409.65."),
    "Waist circumference 74-111": CausalVariable("Waist circumference 74-111", "Waist circumference 74-111", "The patient's waist circumference is 74-111."),
    "Serum CA153 23.7625-47.525": CausalVariable("Serum CA153 23.7625-47.525", "Serum CA153 23.7625-47.525", "The patient's serum CA153 level is 23.7625-47.525."),
    "Body fat 0-15.25kg": CausalVariable("Body fat 0-15.25kg", "Body fat 0-15.25kg", "The patient's body fat is 0-15.25 kg."),
    "Body fat 15.25-30.5kg": CausalVariable("Body fat 15.25-30.5kg", "Body fat 15.25-30.5kg", "The patient's body fat is 15.25-30.5 kg."),
    "CD8 count 0-826.25": CausalVariable("CD8 count 0-826.25", "CD8 count 0-826.25", "The absolute number of all CD8 cells in the patient is 0-826.25, including cytotoxic T cells and regulatory T cells."),
    "CD8 subgroups 0-19.6": CausalVariable("CD8 subgroups 0-19.6", "CD8 subgroups 0-19.6", "The number of CD8 subgroups in the patient is 0-19.6."),
    "CD8 subgroups 19.6-39.2": CausalVariable("CD8 subgroups 19.6-39.2", "CD8 subgroups 19.6-39.2", "The patient's body fat is 19.6-39.2."),
    "Absolute CD3 count 0-1183.25": CausalVariable("Absolute CD3 count 0-1183.2", "Absolute CD3 count 0-1183.2", "The total number of T cells in the patient is 0-1183.2, including CD4 and CD8 cells."),
    "Absolute CD3 count 1183.25-2366.5": CausalVariable("Absolute CD3 count 1183.25-2366.5", "Absolute CD3 count 1183.25-2366.5", "The total number of T cells in the patient is 1183.25-2366.5, including CD4 and CD8 cells."),
    "CD3 count 0-1273.75": CausalVariable("CD3 count 0-1273.75", "CD3 count 0-1273.75", "The patient's relative quantity of T cells is 0-1273.75."),
    "CD3 count 1273.75-2547.5": CausalVariable("CD3 count 1273.75-2547.5", "CD3 count 1273.75-2547.5", "The patient's relative quantity of T cells is 1273.75-2547.5."),
    "Absolute CD4 count 0-673.5": CausalVariable("Absolute CD4 count 0-673.5", "Absolute CD4 count 0-673.5", "The patient's count of a specific type of white blood cell, CD4, in the immune system is 0-673.5."),
    "Absolute CD4 count 673.5-1347": CausalVariable("Absolute CD4 count 673.5-1347", "Absolute CD4 count 673.5-1347", "The patient's count of a specific type of white blood cell, CD4, in the immune system is 673.5-1347."),
    "CD4/CD8 ratio 0-2.9325": CausalVariable("CD4/CD8 ratio 0-2.9325", "CD4/CD8 ratio 0-2.9325", "The patient's ratio of helper cells to suppressor cells is 0-2.9325."),
    "CD4 count 0-688": CausalVariable("CD4 count 0-688", "CD4 count 0-688", "The patient's CD4 count is 0-688."),
    "CD4 count 688-1376": CausalVariable("CD4 count 688-1376", "CD4 count 688-1376", "The patient's CD4 count is 688-1376."),
}


MIMIC_VAR_NAMES_AND_DESC = {
    "admission_age 27.65-45.01" : CausalVariable("admission_age 27.65-45.01", "admission_age 27.65-45.01", "The patient's age is 27.65-45.01"),
    "admission_age 45.01-62.31" : CausalVariable("admission_age 45.01-62.31", "admission_age 45.01-62.31", "The patient's age is 45.01-62.31"),
    "admission_age 62.31-79.60" : CausalVariable("admission_age 62.31-79.60", "admission_age 62.31-79.60", "The patient's age is 62.31-79.60"),
    "admission_age 79.60-96.89" : CausalVariable("admission_age 79.60-96.89", "admission_age 79.60-96.89", "The patient's age is 79.60-96.89"),
    "smoker 0" : CausalVariable("smoker 0", "smoker 0", "The patient does not smoke."),
    "smoker 1" : CausalVariable("smoker 1", "smoker 1", "The patient has a smoking habit."),
    "smoking 0" : CausalVariable("smoking 0", "smoking 0", "The patient is currently a non-smoker."),
    "smoking 1" : CausalVariable("smoking 1", "smoking 1", "The patient currently has a smoking habit."),
    "chronic_pulmonary_disease 0": CausalVariable("chronic_pulmonary_disease 0", "chronic_pulmonary_disease 0", "The patient does not have chronic pulmonary disease."),
    "chronic_pulmonary_disease 1": CausalVariable("chronic_pulmonary_disease 1", "chronic_pulmonary_disease 1", "The patient has chronic pulmonary disease."),
    "malignant_cancer 1": CausalVariable("malignant_cancer 1", "malignant_cancer 1", "The patient has malignant cancer."),
    "metastatic_solid_tumor 0": CausalVariable("metastatic_solid_tumor 0", "metastatic_solid_tumor 0", "The patient does not have metastatic solid tumor."),
    "metastatic_solid_tumor 1": CausalVariable("metastatic_solid_tumor 1", "metastatic_solid_tumor 1", "The patient has metastatic solid tumor."),
    "charlson 1.98-6.5": CausalVariable("charlson 1.98-6.5", "charlson 1.98-6.5", "The Charlson Comorbidity Index is a metric used to assess the severity of comorbidities in patients, particularly when evaluating the prognosis and risks of hospitalized patients. A patient's Charlson score ranges from 1.98 to 6.5."),
    "charlson 6.5-11": CausalVariable("charlson 6.5-11", "charlson 6.5-11", "The Charlson Comorbidity Index is a metric used to assess the severity of comorbidities in patients, particularly when evaluating the prognosis and risks of hospitalized patients. A patient's Charlson score ranges from 6.5 to 11."),
    "charlson 11-15.5": CausalVariable("charlson 11-15.5", "charlson 11-15.5", "The Charlson Comorbidity Index is a metric used to assess the severity of comorbidities in patients, particularly when evaluating the prognosis and risks of hospitalized patients. A patient's Charlson score ranges from 11 to 15.5."),
    "charlson 15.5-20": CausalVariable("charlson 15.5-20", "charlson 15.5-20", "The Charlson Comorbidity Index is a metric used to assess the severity of comorbidities in patients, particularly when evaluating the prognosis and risks of hospitalized patients. A patient's Charlson score ranges from 15.5 to 20."),
    "lung cancer": CausalVariable("lung cancer", "lung cancer", "The patient has lung cancer."),
}



CHILD_VAR_NAMES_AND_DESC = {
    "DuctFlow" : CausalVariable("DuctFlow", "duct flow", "blood flow across the ductus arteriosus"),
    "HypDistrib" : CausalVariable("HypDistrib", "hypoxia distribution", "low oxygen areas equally distributed around the body"),
    "CardiacMixing" : CausalVariable("CardiacMixing", "cardiac mixing", "mixing of oxygenated and deoxygenated blood"),
    "HypoxiaInO2" : CausalVariable("HypoxiaInO2", "hypoxia when breathing oxygen", "hypoxia when breathing oxygen"),
    "LungParench" : CausalVariable("LungParench", "lung parenchyma", "the state of the blood vessels in the lungs"),
    "CO2" : CausalVariable("CO2", "CO2 level", "level of CO2 in the body"),
    "ChestXray" : CausalVariable("ChestXray", "chest xray", "having a chest x-ray"),
    "LungFlow" : CausalVariable("LungFlow", "lung flow", "low blood flow in the lungs"),
    "Grunting" : CausalVariable("Grunting", "grunting", "grunting in infants"),
    "Sick" : CausalVariable("Sick", "sick", "presence of an illness"),
    "LVH" : CausalVariable("LVH", "left ventricular hypertrophy", "having left ventricular hypertrophy"),
    "LVHreport" : CausalVariable("LVHreport", "left ventricular hypertrophy report", "report of having left ventricular hypertrophy"),
    "LowerBodyO2" : CausalVariable("LowerBodyO2", "lower body oxygen level", "level of oxygen in the lower body"),
    "RUQO2" : CausalVariable("RUQO2", "right upper quadriceps oxygen level", "level of oxygen in the right upper quadriceps muscule"),
    "CO2Report" : CausalVariable("CO2Report", "CO2 report", "a document reporting high level of CO2 levels in blood"),
    "XrayReport" : CausalVariable("XrayReport", "xray report", "lung excessively filled with blood"),
    "BirthAsphyxia" : CausalVariable("BirthAsphyxia", "birth asphyxia", "lack of oxygen to the blood during the infant's birth"),
    "Disease" : CausalVariable("Disease", "disease", "infant methemoglobinemia"),
    "GruntingReport" : CausalVariable("GruntingReport", "grunting report", "report of infant grunting"),
    "Age" : CausalVariable("Age", "age", "age of infant at disease presentation"),
}




VAR_NAMES_AND_DESC = {
    "asia" : ASIA_VAR_NAMES_AND_DESC,
    "huaxi16" : WCHSU16_VAR_NAMES_AND_DESC,
    "huaxi51" : WCHSU51_VAR_NAMES_AND_DESC,
    "mimic" : MIMIC_VAR_NAMES_AND_DESC,
    "child" : CHILD_VAR_NAMES_AND_DESC,
}


