# Radiomics-Analysis-and-ML-for-Prognostic-Biomarkers-in-IPF
This repository contains the code for my master’s thesis on predicting Forced Vital Capacity (FVC) one year after baseline for Idiopathic Pulmonary Fibrosis (IPF) patients using clinical features, radiomic features from HRCT, and a combined radiomic + clinical (RC) setting.
The workflow includes image preprocessing, automated segmentation, radiomics extraction, feature selection pipelines, and regression models (Elastic Net, Partial Least Squares, Random Forest) with basic interpretability.

# Dataset (OSIC Pulmonary Fibrosis)

## Source
This project uses the **OSIC Pulmonary Fibrosis dataset** (Open Source Imaging Consortium).
The dataset is **not included in this repository** size restrictions.

## Contents
The dataset includes:
- **HRCT scans** (anonymized) for IPF patients
- **Clinical metadata** collected over time (longitudinal measurements such as FVC

## Pipeline Overview

![Pipeline Overview](https://github.com/YashodharPansuriya/Radiomics-Analysis-and-ML-for-Prognostic-Biomarkers-in-IPF/blob/main/Images/IPF_Pipeline.PNG)


## Setup

Clone the repository and install the dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv .venv
source .venv/bin/activate   

pip install -r requirements.txt


## Repository Overview (Step-by-step Structure)

This repository is organized according to the workflow used in the thesis.  
Each folder represents one major step of the pipeline, from Patients cohort selection to model training and result generation.

### Folder descriptions

- **`Patients_Data_Exploration/`**  
  Contains notebooks/scripts for selecting the patient cohort from the OSIC dataset that is relevant to the thesis objective.

- **`Segmentation/`** *(if you have this folder)*  
  Code for automated lung segmentation (mask generation) used before radiomics extraction.  
  **Output:** segmentation masks saved in `data/processed/masks/`.

- **`Exploratory_Data_Analysis/`**  
  Notebooks for exploring clinical variables, checking missing values, distributions, and correlations.  
  **Output:** EDA plots saved in `results/figures/`.

- **`Radiomics_Extraction/`**  
  Scripts/notebooks to extract radiomic features from segmented HRCT scans (e.g., using PyRadiomics).  
  **Output:** radiomics feature table saved as `data/processed/radiomics_features.csv`.

- **`Feature_Selection/`** *(if you have this folder)*  
  Implements different feature selection pipelines (six pipelines + full feature set) for the combined RC scenario.  
  **Output:** selected feature sets saved in `data/processed/featuresets/`.

- **`Model_Training/`**  
  Training and evaluation code for regression models:
  - Elastic Net (EN)
  - Partial Least Squares (PLS)
  - Random Forest (RF)  
  **Output:** trained models and evaluation results saved in `models/` and `results/tables/`.

- **`Results/`** *(or `results/`)*  
  Final thesis outputs:
  - `results/figures/` → all figures used in the thesis  
  - `results/tables/` → all tables used in the thesis

---

## Recommended execution order (to reproduce results)

Run the folders in this order:

1. `Image_Preprocessing/`
2. `Segmentation/` (if applicable)
3. `Radiomics_Extraction/`
4. `Exploratory_Data_Analysis/`
5. `Feature_Selection/`
6. `Model_Training/`
7. `Results/` (figure/table generation)

> Note: The dataset is not included in this repository. Please follow `data/README.md` to download and place the OSIC dataset in the correct structure.

