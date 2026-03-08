# Risk and Protective Factors for Hepatitis B Vaccination in Under-5s in Cambodia

This repository contains the analytical code for my final-year undergraduate research project investigating 
the socioeconomic, demographic, and behavioural factors associated with Hepatitis B vaccination uptake 
among children under 5 in Cambodia.

## Data
This project uses the 2021-22 Cambodia Demographic and Health Survey (DHS), accessed via the DHS Program 
(https://dhsprogram.com). Raw data is not included in this repository in line with DHS data use agreements.

## Methods
- Binary logistic regression for vaccination at birth (yes/no)
- Ordinal logistic regression for full vaccination completion
- Survey-weighted analysis using DHS sample weights
- Analysis conducted in Python using rpy2 to interface with R's survey package

## Structure
- `analysis notebooks/` – Jupyter notebooks for each variable group (Maternal, Regional, etc)
- `src/` – Reusable Python functions including model fitting and data cleaning utilities
- `data/` – Notebook used to clean raw data (raw data not included)
- `output/` – Generated figures and model outputs

## Requirements
Reproduce the environment using conda:
```bash
conda env create -f environment.yml
conda activate Vaccination_Cambodia
```
