# =======================================================================
# WARNING: SENSITIVE DATA REMOVED
# The original list of 'case_person_ids' has been removed from this script
# to protect participant privacy and comply with data use agreements.
# To run this script, you must replace the placeholder with your own
# defined list of case person_ids.
# =======================================================================

import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, fisher_exact

# Import the specific exception class we need to catch
from pandas_gbq.gbq import GenericGBQException

print("--- Script Started: Running with CORRECTED method for variant exclusion. ---")

### -----------------------------------------------------------------------
### PART 1 & 2: CREATE CASE AND CONTROL COHORTS
### -----------------------------------------------------------------------
print("\nPART 1 & 2: Creating case and control cohorts...")

# IMPORTANT: PASTE YOUR LIST OF CASE PERSON IDS HERE
# The original list has been removed for privacy.
case_person_ids = []

case_race_ids = [8516, 2100000001, 2000000008]
case_race_sql_string = ",".join(map(str, case_race_ids))
control_race_id = 8516
min_ehr_years = 5

# --- Case Cohort Definition ---
case_cohort_sql = f"""
WITH ehr_summary AS (
    SELECT person_id, MIN(visit_start_date) as first_visit_date, DATE_DIFF(MAX(visit_start_date), MIN(visit_start_date), DAY) / 365.25 AS ehr_years
    FROM `{os.environ["WORKSPACE_CDR"]}.visit_occurrence` GROUP BY person_id
)
SELECT p.person_id, p.sex_at_birth_concept_id, p.race_concept_id, 
       DATE_DIFF(summary.first_visit_date, CAST(p.birth_datetime AS DATE), YEAR) AS age_at_first_visit, 
       summary.ehr_years
FROM `{os.environ["WORKSPACE_CDR"]}.person` AS p
JOIN `{os.environ["WORKSPACE_CDR"]}.cb_search_person` AS cbsp ON p.person_id = cbsp.person_id
JOIN ehr_summary AS summary ON p.person_id = summary.person_id
WHERE p.person_id IN ({",".join(map(str, case_person_ids))})
  AND p.race_concept_id IN ({case_race_sql_string})
  AND cbsp.has_ehr_data = 1
  AND cbsp.has_whole_genome_variant = 1
  AND summary.ehr_years >= {min_ehr_years} 
"""
case_cohort_df = pd.read_gbq(case_cohort_sql, progress_bar_type=None)
print(f"Found {len(case_cohort_df)} cases that meet the initial criteria.")

# --- Control Cohort Definition (Initial Pull) ---
control_cohort_sql = f"""
SELECT p.person_id, p.sex_at_birth_concept_id, p.race_concept_id,
        DATE_DIFF(MIN(v.visit_start_date), CAST(p.birth_datetime AS DATE), YEAR) AS age_at_first_visit,
       DATE_DIFF(MAX(v.visit_start_date), MIN(v.visit_start_date), DAY) / 365.25 AS ehr_years
FROM `{os.environ["WORKSPACE_CDR"]}.person` AS p
JOIN `{os.environ["WORKSPACE_CDR"]}.cb_search_person` AS cbsp ON p.person_id = cbsp.person_id
JOIN `{os.environ["WORKSPACE_CDR"]}.visit_occurrence` AS v ON p.person_id = v.person_id
WHERE p.race_concept_id = {control_race_id}
  AND cbsp.has_ehr_data = 1
  AND cbsp.has_whole_genome_variant = 1
  AND p.person_id NOT IN ({",".join(map(str, case_person_ids))})
GROUP BY p.person_id, p.sex_at_birth_concept_id, p.race_concept_id, p.birth_datetime
HAVING DATE_DIFF(MAX(v.visit_start_date), MIN(v.visit_start_date), DAY) / 365.25 >= {min_ehr_years}
LIMIT 70000
"""
control_cohort_df = pd.read_gbq(control_cohort_sql, progress_bar_type=None)
print(f"Found {len(control_cohort_df)} potential controls before variant exclusion.")

# --- CORRECTED LOGIC: Exclude Variant Carriers using cb_variant_to_person ---
variant_to_exclude = "7-150628405-C-T"
try:
    # This query now uses the method you showed me, which should be correct.
    variant_carriers_sql = f"""
    SELECT DISTINCT person_id_unnested AS person_id
    FROM `{os.environ["WORKSPACE_CDR"]}.cb_variant_to_person`
    CROSS JOIN UNNEST(person_ids) AS person_id_unnested
    WHERE vid = '{variant_to_exclude}'
    """
    variant_carriers_df = pd.read_gbq(variant_carriers_sql, progress_bar_type=None)

    # If successful, filter the controls.
    if not variant_carriers_df.empty:
        ids_to_exclude = variant_carriers_df["person_id"].tolist()
        initial_control_count = len(control_cohort_df)
        control_cohort_df = control_cohort_df[
            ~control_cohort_df["person_id"].isin(ids_to_exclude)
        ]
        final_control_count = len(control_cohort_df)
        print(
            f"Successfully excluded {initial_control_count - final_control_count} variant carriers from the control pool."
        )
    else:
        print(
            f"No carriers found for variant '{variant_to_exclude}'. No controls were excluded."
        )
except GenericGBQException as e:
    # This will catch an error if the 'cb_variant_to_person' table itself is not found.
    print("\n" + "=" * 80)
    print(f"!!! WARNING: Querying for variant carriers failed.")
    print(
        f"!!! The analysis will continue WITHOUT excluding variant carriers from the control group."
    )
    print(f"--- Error Details: {e}")
    print("=" * 80 + "\n")

print(
    f"Final pool of {len(control_cohort_df)} potential controls is ready for matching."
)

### -----------------------------------------------------------------------
### PART 3: MATCHING
### -----------------------------------------------------------------------
print("\nPART 3: Performing upgraded matching with flexible windows...")
AGE_WINDOW, EHR_YEARS_WINDOW, CONTROLS_PER_CASE = 5, 5, 100

control_pool = control_cohort_df.copy()
matched_controls_list, all_case_ids = [], []

for i, case in case_cohort_df.iterrows():
    case_id, case_sex, case_age, case_ehr_years = (
        case["person_id"],
        case["sex_at_birth_concept_id"],
        case["age_at_first_visit"],
        case["ehr_years"],
    )

    possible_matches = control_pool[
        (control_pool["sex_at_birth_concept_id"] == case_sex)
        & (
            control_pool["age_at_first_visit"].between(
                case_age - AGE_WINDOW, case_age + AGE_WINDOW
            )
        )
        & (
            control_pool["ehr_years"].between(
                case_ehr_years - EHR_YEARS_WINDOW, case_ehr_years + EHR_YEARS_WINDOW
            )
        )
    ]

    if len(possible_matches) >= CONTROLS_PER_CASE:
        matched_controls = possible_matches.sample(n=CONTROLS_PER_CASE, random_state=1)
        matched_controls_list.append(matched_controls)
        all_case_ids.append(case_id)
        control_pool.drop(matched_controls.index, inplace=True)

final_control_df = (
    pd.concat(matched_controls_list) if matched_controls_list else pd.DataFrame()
)
final_case_df = case_cohort_df[case_cohort_df["person_id"].isin(all_case_ids)].copy()

print(
    f"Matching complete. Final Case Cohort size: {len(final_case_df)}, Final Control Cohort size: {len(final_control_df)}"
)

### -----------------------------------------------------------------------
### PART 4: ANALYSIS
### -----------------------------------------------------------------------
print("\nPART 4: Calculating final results and generating tables...")

cvd_phenotype_map = {
    "Atherosclerosis & Infarction": [764123, 37309626],
    "Stroke & Cerebrovascular Disease": [443454, 373503, 4288310],
    "Heart Failure & Enlarged Heart": [316139, 314658, 4229440, 40481042],
    "Arrhythmias & ECG Abnormalities": [44784217, 313217, 4154290, 437579],
    "Angina Pectoris (Ischemia Sign)": [321318],
}

background_phenotype_map = {
    "Hypercholesterolemia": [4029305, 35207060, 37200312, 44834564, 437827, 4031945],
    "Diabetes": [201820, 201826],
    "Hypertension": [316866, 320128, 42709887, 4028741, 312648],
    "CKD": [46271022, 198124],
}

all_cvd_concepts = list(
    set(item for sublist in cvd_phenotype_map.values() for item in sublist)
)
all_background_concepts = list(
    set(item for sublist in background_phenotype_map.values() for item in sublist)
)
all_concepts_to_query = list(set(all_cvd_concepts + all_background_concepts))
all_person_ids_to_check = pd.concat(
    [final_case_df["person_id"], final_control_df["person_id"]]
).tolist()

if all_person_ids_to_check:
    comprehensive_sql = f"""
    SELECT DISTINCT person_id, condition_concept_id 
    FROM `{os.environ["WORKSPACE_CDR"]}.condition_occurrence` 
    WHERE person_id IN ({",".join(map(str, all_person_ids_to_check))})
      AND condition_concept_id IN ({",".join(map(str, all_concepts_to_query))})
    """
    all_conditions_df = pd.read_gbq(comprehensive_sql, progress_bar_type=None)

    print("\n\n--- Table 1: Baseline Demographics of Matched Cohorts ---")
    table1_data = []
    total_cases, total_controls = len(final_case_df), len(final_control_df)
    FEMALE_CONCEPT_ID = 45878463
    case_ages, control_ages = (
        final_case_df["age_at_first_visit"],
        final_control_df["age_at_first_visit"],
    )
    _, age_p_value = ttest_ind(case_ages, control_ages, equal_var=False)
    table1_data.append(
        {
            "Characteristic": "Age (Mean ± SD)",
            f"Cases (N={total_cases})": f"{case_ages.mean():.1f} ± {case_ages.std():.1f}",
            f"Controls (N={total_controls})": f"{control_ages.mean():.1f} ± {control_ages.std():.1f}",
            "P-Value": f"{age_p_value:.3f}",
        }
    )
    cases_female = (final_case_df["sex_at_birth_concept_id"] == FEMALE_CONCEPT_ID).sum()
    controls_female = (
        final_control_df["sex_at_birth_concept_id"] == FEMALE_CONCEPT_ID
    ).sum()
    _, sex_p_value = fisher_exact(
        [
            [cases_female, total_cases - cases_female],
            [controls_female, total_controls - controls_female],
        ]
    )
    table1_data.append(
        {
            "Characteristic": "Sex (% Female)",
            f"Cases (N={total_cases})": f"{cases_female} ({cases_female / total_cases * 100:.1f}%)",
            f"Controls (N={total_controls})": f"{controls_female} ({controls_female / total_controls * 100:.1f}%)",
            "P-Value": f"{sex_p_value:.3f}",
        }
    )
    table1_df = pd.DataFrame(table1_data).set_index("Characteristic")
    print(table1_df)

    def calculate_or_ci_p(cases_with, total_cases, controls_with, total_controls):
        contingency_table = [
            [cases_with, total_cases - cases_with],
            [controls_with, total_controls - controls_with],
        ]
        odds_ratio, p_value = fisher_exact(contingency_table)
        a, b, c, d = (
            cases_with + 0.5,
            (total_cases - cases_with) + 0.5,
            controls_with + 0.5,
            (total_controls - controls_with) + 0.5,
        )
        se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci_lower = np.exp(np.log(odds_ratio) - 1.96 * se_log_or)
        ci_upper = np.exp(np.log(odds_ratio) + 1.96 * se_log_or)
        return odds_ratio, ci_lower, ci_upper, p_value

    def generate_results_table(phenotype_map):
        results_data = []
        for phenotype, concept_ids in phenotype_map.items():
            person_ids_with_pheno = set(
                all_conditions_df[
                    all_conditions_df["condition_concept_id"].isin(concept_ids)
                ]["person_id"]
            )
            cases_with = final_case_df["person_id"].isin(person_ids_with_pheno).sum()
            controls_with = (
                final_control_df["person_id"].isin(person_ids_with_pheno).sum()
            )
            or_val, ci_low, ci_high, p_val = calculate_or_ci_p(
                cases_with, total_cases, controls_with, total_controls
            )
            results_data.append(
                {
                    "Phenotype": phenotype,
                    "Cases": f"{cases_with}/{total_cases}",
                    "Controls": f"{controls_with}/{total_controls}",
                    "Odds Ratio": f"{or_val:.2f}",
                    "95% CI": f"({ci_low:.2f} - {ci_high:.2f})",
                    "P-Value": f"{p_val:.3f}" if p_val >= 0.001 else "<0.001",
                }
            )
        return pd.DataFrame(results_data).set_index("Phenotype")

    print("\n\n--- Table 2: Odds Ratios for Updated CVD Phenotypes ---")
    cvd_results_df = generate_results_table(cvd_phenotype_map)

    # MODIFICATION: Add "Heart Failure & Enlarged Heart" to the Major CVD Events calculation.
    major_cvd_list = list(
        set(
            cvd_phenotype_map["Atherosclerosis & Infarction"]
            + cvd_phenotype_map["Stroke & Cerebrovascular Disease"]
            + cvd_phenotype_map["Heart Failure & Enlarged Heart"]
        )
    )
    major_ids = set(
        all_conditions_df[
            all_conditions_df["condition_concept_id"].isin(major_cvd_list)
        ]["person_id"]
    )
    major_cases = final_case_df["person_id"].isin(major_ids).sum()
    major_controls = final_control_df["person_id"].isin(major_ids).sum()
    or_val, ci_low, ci_high, p_val = calculate_or_ci_p(
        major_cases, total_cases, major_controls, total_controls
    )

    # MODIFICATION: Update the label for the summary row to be more descriptive.
    cvd_results_df.loc["Major CVD Events (Infarction/Stroke/HF)"] = [
        f"{major_cases}/{total_cases}",
        f"{major_controls}/{total_controls}",
        f"{or_val:.2f}",
        f"({ci_low:.2f} - {ci_high:.2f})",
        f"{p_val:.3f}" if p_val >= 0.001 else "<0.001",
    ]
    print(cvd_results_df)

    print("\n\n--- Table 3: Odds Ratios for Background Conditions ---")
    background_results_df = generate_results_table(background_phenotype_map)
    print(background_results_df)
else:
    print(
        "\nMatching resulted in zero participants. Cannot proceed with final analysis."
    )
