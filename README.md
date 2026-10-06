# GIMAP6_CVD_Analysis

Code for the analysis of GIMAP6 variants and cardiovascular disease in the All of Us cohort, as described in Chen, X. et al., 2025.

## Project overview

This repository contains the code used to run the statistical analyses and generate the figures for the human cohort data in Figure 6 of the manuscript. The primary variant of interest is GIMAP6 p.Val65Ile (chr7:150628405 C>T, `7-150628405-C-T`). The scripts compare carriers and non-carriers across cardiovascular phenotypes, with positive and negative control diseases and control variants.

The analysis ran inside the secure **[All of Us Researcher Workbench](https://workbench.researchallofus.org/)**. Participant privacy rules mean the individual-level data cannot be shared, so this code is published for methodological transparency. Running it requires approved access to the All of Us Controlled Tier.

## Requirements

- An All of Us Researcher Workbench workspace on the **Controlled Tier Dataset v8 (C2024Q3R5)**
- Python 3.9 or later
- The packages in `requirements.txt`, all of which are preinstalled in the Workbench Jupyter environment. `hail` is needed only for the genotype extraction script.

Phenotypes are defined by OMOP concept IDs (SNOMED/ICD-derived), declared at the top of each script (for example `CAD_CONCEPTS` and `HF_CONCEPTS`).

## Scripts

| Script                                                                    | What it does                                                                                                                                                                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GIMAP6 V65I homozygous genotype extraction_py`                           | Hail: extracts genotypes at chr7:150628405 from a workspace genomic extraction, to identify homozygous carriers.                                                                                  |
| `run_prevalence_analysis.py`                                              | Disease prevalence in carriers vs non-carriers of GIMAP6 V65I, TTR V122I and GIMAP6 Q237R across primary, positive-control and negative-control diseases (Fisher's exact test).                   |
| `Carrier frequency across age strata_Analysis.py`                         | Selects random missense control variants, then compares how carrier frequency changes across age strata for the test and control variants.                                                        |
| `Age-stratified carrier frequency analysis`                               | Age-stratified carrier frequency of the test variants against a distribution of 100 control variants, with multiple-testing correction.                                                           |
| `Adjusted logistic regression for cardiovascular comorbidity_analysis.py` | Logistic regression of cardiovascular outcomes on carrier status, adjusted for age, sex, smoking, BMI category, diabetes, hypertension, hyperlipidemia and genetic ancestry principal components. |
| `V65I case-control study_.py`                                             | Case-control comparison of a defined case group against a control cohort from the same race group, with variant carriers excluded.                                                                |
| `Permutation Testing.py`                                                  | Permutation test (10,000 permutations) for the count of major CVD events in the case group against random groups.                                                                                 |
| `Scripts`                                                                 | An identical copy of `run_prevalence_analysis.py`, kept so that existing links continue to work.                                                                                                  |

## Usage

In a Jupyter notebook in your Workbench workspace, upload the scripts and run them one at a time, for example:

```bash
python run_prevalence_analysis.py
```

The scripts were developed as notebook cells, so running them cell by cell in a notebook works just as well. They use the environment variables that the Workbench sets:

- `WORKSPACE_CDR`: the CDR BigQuery dataset. Several scripts default it to `fc-aou-cdr-prod-ct.C2024Q3R5`.
- `GOOGLE_CLOUD_PROJECT`: the billing project for BigQuery.

Two scripts need inputs that are not in this repository:

- **Genotype extraction** reads `GIMAP6_VCF_PATH`, the VCFs produced by a genomic extraction in your own workspace:

  ```bash
  export GIMAP6_VCF_PATH="$WORKSPACE_BUCKET/genomic-extractions/<extraction-id>/vcfs/*.vcf.gz"
  ```

- **Case-control** and **permutation testing** need a `case_person_ids` list. The original list was removed to protect participant privacy, so define your own case group and fill it in at the top of each script.

Each script prints summary tables and draws its figures inline. Follow the **[All of Us Data and Statistics Dissemination Policy](https://www.researchallofus.org/data-tools/data-access/)** before exporting any output from the Workbench: no participant-level data, and no counts below 20.

## Contributing

This repository is a snapshot of the code behind the published analysis and is not under active development. Issues and questions are welcome. Pull requests that change the analysis will not be merged, so that the code continues to match the paper.

## License

MIT. See **[LICENSE.md](LICENSE.md)**.
