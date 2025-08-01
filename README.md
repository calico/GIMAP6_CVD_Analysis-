# GIMAP6_CVD_Analysis

Code and analysis for the association of GIMAP6 variants with cardiovascular disease in the All of Us cohort, as described in Chen, X. et al., 2025.

## Project Overview
This repository contains the code used to perform the statistical analyses and generate the figures for the human cohort data presented in Figure 6 of our manuscript. The analysis was conducted within the secure All of Us Researcher Workbench. Due to patient privacy restrictions, the individual-level raw data cannot be shared. However, this code is provided to ensure full transparency of our methodology.

## System Requirements
* Python 3.9
* All required packages are listed in the `requirements.txt` file.

## "All of Us" Data Requirements
* **Dataset:** All of Us Controlled Tier Dataset v8 [C2024Q3R5]
* **Phenotypes:** The specific SNOMED and ICD concept IDs used to define Coronary Artery Disease (CAD), Heart Failure (HF), and Hypercholesterolemia (HChol) are provided in `data/phenotype_concept_ids.csv`.
* **Variants:** The primary variant of interest is GIMAP6 p.Val65Ile (chr7:150628405 C>T). All other variants used for comparison are listed in the analysis scripts.

## Instructions for Use
To replicate this analysis, an approved researcher on the All of Us Researcher Workbench should:
1. Create a new workspace using the specified dataset version.
2. Upload the scripts from the `/scripts` folder.
3. Run the scripts in the following order:
   - `01_define_cohort.py`: To query the database and create the initial analysis cohort.
   - `02_run_associations.py`: To perform the logistic regression and other statistical tests.
   - `03_generate_figures.py`: To generate the final plots.

## Expected Output
The scripts will generate the figures found in Figure 6 and the summary-level data tables located in the `/results` folder.
