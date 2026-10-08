### =======================================================================
# WARNING: SENSITIVE DATA REMOVED
# The original list of 'case_person_ids' has been removed from this script
# to protect participant privacy. To run this script, you must replace
# the placeholder with your own list of case person_ids.
### =======================================================================

# Step 1: Import all necessary libraries
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, fisher_exact
from pandas_gbq.gbq import GenericGBQException
import matplotlib.pyplot as plt
import seaborn as sns

print(
    "--- Script Started: Running with ROBUST Permutation Testing for Significance. ---"
)

### -----------------------------------------------------------------------
### PART 1 & 2: CREATE CASE AND CONTROL COHORTS (Unchanged)
### -----------------------------------------------------------------------
print("\nPART 1 & 2: Creating case and control cohorts...")
# This section is the same as Version 9, it prepares our initial data

# <-- IMPORTANT: PASTE YOUR LIST OF CASE PERSON IDS HERE. The original list has been removed for privacy.
case_person_ids = []

case_race_ids = [8516, 2100000001, 2000000008]
case_race_sql_string = ",".join(map(str, case_race_ids))
control_race_id = 8516
min_ehr_years = 5

case_cohort_sql = f"""
WITH ehr_summary AS (
    SELECT person_id, MIN(visit_start_date) as first_visit_date, DATE_DIFF(MAX(visit_start_date), MIN(visit_start_date), DAY) / 365.25 AS ehr_years
    FROM `{os.environ["WORKSPACE_CDR"]}.visit_occurrence` GROUP BY person_id
)
SELECT p.person_id, p.sex_at_birth_concept_id, p.race_concept_id, DATE_DIFF(summary.first_visit_date, CAST(p.birth_datetime AS DATE), YEAR) AS age_at_first_visit, summary.ehr_years
FROM `{os.environ["WORKSPACE_CDR"]}.person` AS p
JOIN `{os.environ["WORKSPACE_CDR"]}.cb_search_person` AS cbsp ON p.person_id = cbsp.person_id
JOIN ehr_summary AS summary ON p.person_id = summary.person_id
WHERE p.person_id IN ({",".join(map(str, case_person_ids))})
    AND p.race_concept_id IN ({case_race_sql_string})
    AND cbsp.has_ehr_data = 1 AND cbsp.has_whole_genome_variant = 1 AND summary.ehr_years >= {min_ehr_years}
"""
case_cohort_df = pd.read_gbq(case_cohort_sql, progress_bar_type=None)
print(f"Found {len(case_cohort_df)} cases that meet the initial criteria.")

control_cohort_sql = f"""
SELECT p.person_id, p.sex_at_birth_concept_id, p.race_concept_id,
        DATE_DIFF(MIN(v.visit_start_date), CAST(p.birth_datetime AS DATE), YEAR) AS age_at_first_visit,
        DATE_DIFF(MAX(v.visit_start_date), MIN(v.visit_start_date), DAY) / 365.25 AS ehr_years
FROM `{os.environ["WORKSPACE_CDR"]}.person` AS p
JOIN `{os.environ["WORKSPACE_CDR"]}.cb_search_person` AS cbsp ON p.person_id = cbsp.person_id
JOIN `{os.environ["WORKSPACE_CDR"]}.visit_occurrence` AS v ON p.person_id = v.person_id
WHERE p.race_concept_id = {control_race_id} AND cbsp.has_ehr_data = 1 AND cbsp.has_whole_genome_variant = 1
    AND p.person_id NOT IN ({",".join(map(str, case_person_ids))})
GROUP BY p.person_id, p.sex_at_birth_concept_id, p.race_concept_id, p.birth_datetime
HAVING DATE_DIFF(MAX(v.visit_start_date), MIN(v.visit_start_date), DAY) / 365.25 >= {min_ehr_years}
LIMIT 70000
"""
control_cohort_df = pd.read_gbq(control_cohort_sql, progress_bar_type=None)

variant_to_exclude = "7-150628405-C-T"
try:
    variant_carriers_sql = f"""
    SELECT DISTINCT person_id_unnested AS person_id
    FROM `{os.environ["WORKSPACE_CDR"]}.cb_variant_to_person`
    CROSS JOIN UNNEST(person_ids) AS person_id_unnested
    WHERE vid = '{variant_to_exclude}'
    """
    variant_carriers_df = pd.read_gbq(variant_carriers_sql, progress_bar_type=None)
    if not variant_carriers_df.empty:
        ids_to_exclude = variant_carriers_df["person_id"].tolist()
        control_cohort_df = control_cohort_df[
            ~control_cohort_df["person_id"].isin(ids_to_exclude)
        ]
        print(f"Successfully excluded variant carriers from the control pool.")
except GenericGBQException as e:
    print(
        f"!!! WARNING: Could not exclude variant carriers. The analysis will continue WITHOUT this exclusion."
    )

# Fetch condition data for all participants
all_person_ids_to_check = pd.concat(
    [case_cohort_df["person_id"], control_cohort_df["person_id"]]
).tolist()

cvd_phenotype_map = {
    "Atherosclerosis & Infarction": [764123, 37309626],
    "Stroke & Cerebrovascular Disease": [443454, 373503, 4288310],
    "Heart Failure & Enlarged Heart": [316139, 314658, 4229440, 40481042],
}
major_cvd_concept_ids = list(
    set(item for sublist in cvd_phenotype_map.values() for item in sublist)
)
# Define the concept IDs for "Heart Failure & Enlarged Heart" specifically
heart_failure_concept_ids = cvd_phenotype_map["Heart Failure & Enlarged Heart"]

comprehensive_sql = f"""
SELECT DISTINCT person_id, condition_concept_id FROM `{os.environ["WORKSPACE_CDR"]}.condition_occurrence`
WHERE person_id IN ({",".join(map(str, all_person_ids_to_check))})
    AND condition_concept_id IN ({",".join(map(str, major_cvd_concept_ids))})
"""
all_conditions_df = pd.read_gbq(comprehensive_sql, progress_bar_type=None)
person_ids_with_major_cvd = set(all_conditions_df["person_id"])

# Create a set of person IDs with 'Heart Failure & Enlarged Heart'
person_ids_with_heart_failure = set(
    all_conditions_df[
        all_conditions_df["condition_concept_id"].isin(heart_failure_concept_ids)
    ]["person_id"]
)

### -----------------------------------------------------------------------
### PART 3: ROBUST PERMUTATION TESTING (for Major CVD)
### -----------------------------------------------------------------------
print("\nPART 3: Beginning Robust Permutation Test for Major CVD...")
N_PERMUTATIONS = 10000  # Number of permutations for the p-value calculation
AGE_WINDOW, EHR_YEARS_WINDOW = 5, 5

# --- Step 3a: Create the high-quality pool of all possible matched controls ---
possible_matches_pool_indices = set()
for i, case in case_cohort_df.iterrows():
    case_sex, case_age, case_ehr_years = (
        case["sex_at_birth_concept_id"],
        case["age_at_first_visit"],
        case["ehr_years"],
    )
    matches = control_cohort_df[
        (control_cohort_df["sex_at_birth_concept_id"] == case_sex)
        & (
            control_cohort_df["age_at_first_visit"].between(
                case_age - AGE_WINDOW, case_age + AGE_WINDOW
            )
        )
        & (
            control_cohort_df["ehr_years"].between(
                case_ehr_years - EHR_YEARS_WINDOW, case_ehr_years + EHR_YEARS_WINDOW
            )
        )
    ]
    possible_matches_pool_indices.update(matches.index)

matched_control_pool_df = control_cohort_df.loc[list(possible_matches_pool_indices)]

# --- Step 3b: Create the MASTER POOL and calculate the TRUE observed statistic ---
# Combine the cases and all their potential matches into one DataFrame
master_pool_df = pd.concat([case_cohort_df, matched_control_pool_df])
print(
    f"Created a Master Pool of {len(master_pool_df)} total individuals (cases + matched controls)."
)

# The true statistic is the number of events in our actual case group. This is fixed.
num_cases = len(case_cohort_df)

# -------------------------------------------------------------------------
# Permutation Test for Major CVD
# -------------------------------------------------------------------------
observed_major_cvd_count = (
    case_cohort_df["person_id"].isin(person_ids_with_major_cvd).sum()
)
print(
    f"\nOBSERVED STATISTIC (Major CVD): The true case group has {observed_major_cvd_count} individuals with a Major CVD Event."
)
print(f"Running {N_PERMUTATIONS} permutations for Major CVD Events...")
random_major_cvd_counts = []
for i in range(N_PERMUTATIONS):
    random_group = master_pool_df.sample(n=num_cases, replace=False)
    random_count = random_group["person_id"].isin(person_ids_with_major_cvd).sum()
    random_major_cvd_counts.append(random_count)

# Calculate descriptive statistics for Major CVD permutations
major_cvd_perm_mean = np.mean(random_major_cvd_counts)
major_cvd_perm_median = np.median(random_major_cvd_counts)
major_cvd_perm_min = np.min(random_major_cvd_counts)
major_cvd_perm_max = np.max(random_major_cvd_counts)
major_cvd_perm_std = np.std(random_major_cvd_counts)

# Analyze results and calculate p-value for Major CVD
count_extreme_values_major_cvd = np.sum(
    np.array(random_major_cvd_counts) >= observed_major_cvd_count
)
empirical_p_value_major_cvd = count_extreme_values_major_cvd / N_PERMUTATIONS

print("\n--- PERMUTATION TEST RESULTS (ROBUST METHOD) for Major CVD ---")
print(f"Observed event count in the true case group: {observed_major_cvd_count}")
print(
    f"Times a random group had >= {observed_major_cvd_count} events: {count_extreme_values_major_cvd} out of {N_PERMUTATIONS}"
)
print(f"Empirical P-Value: {empirical_p_value_major_cvd:.4f}")
print(f"Average individuals with Major CVD per random group: {major_cvd_perm_mean:.2f}")
print(f"Median individuals with Major CVD per random group: {major_cvd_perm_median}")
print(f"Min individuals with Major CVD in a sample: {major_cvd_perm_min}")
print(f"Max individuals with Major CVD in a sample: {major_cvd_perm_max}")
print(f"Standard deviation of Major CVD counts: {major_cvd_perm_std:.2f}")

print("\n--- Summary of Raw Data for Major CVD Event Permutations ---")
major_cvd_counts_summary = (
    pd.Series(random_major_cvd_counts).value_counts().sort_index()
)
print(major_cvd_counts_summary.to_string())

# Plot for Major CVD Permutation Test
print("\nGenerating results plot for Major CVD Events Permutation Test...")
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
sns.histplot(
    data=random_major_cvd_counts, ax=ax, discrete=True, stat="density", color="#5DADE2"
)
ax.axvline(
    x=observed_major_cvd_count,
    color="#C0392B",
    linestyle="--",
    linewidth=2,
    label=f"Observed Event Count ({observed_major_cvd_count})",
)
ax.legend()
ax.set_title(
    f"Distribution of Major CVD Event Counts in Random Groups ({N_PERMUTATIONS} Permutations)",
    fontsize=14,
    weight="bold",
)
ax.set_xlabel(
    f"Number of People with Major CVD Event (in a random group of {num_cases})",
    fontsize=12,
)
ax.set_ylabel("Density (Proportion of Permutations)", fontsize=12)
plt.show()

### -----------------------------------------------------------------------
### NEW PART: ROBUST PERMUTATION TESTING (for Heart Failure & Enlarged Heart)
### -----------------------------------------------------------------------
print(
    "\nNEW PART: Beginning Robust Permutation Test for Heart Failure & Enlarged Heart..."
)

# -------------------------------------------------------------------------
# Permutation Test for Heart Failure & Enlarged Heart
# -------------------------------------------------------------------------
observed_heart_failure_count = (
    case_cohort_df["person_id"].isin(person_ids_with_heart_failure).sum()
)
print(
    f"\nOBSERVED STATISTIC (Heart Failure & Enlarged Heart): The true case group has {observed_heart_failure_count} individuals with Heart Failure & Enlarged Heart."
)
N_HEART_FAILURE_PERMUTATIONS = 10000  # Using 10000 simulations as requested
print(
    f"Running {N_HEART_FAILURE_PERMUTATIONS} permutations for Heart Failure & Enlarged Heart..."
)
random_heart_failure_counts = []
for i in range(N_HEART_FAILURE_PERMUTATIONS):
    random_group_hf = master_pool_df.sample(n=num_cases, replace=False)
    hf_count = random_group_hf["person_id"].isin(person_ids_with_heart_failure).sum()
    random_heart_failure_counts.append(hf_count)

# Calculate descriptive statistics for Heart Failure permutations
heart_failure_perm_mean = np.mean(random_heart_failure_counts)
heart_failure_perm_median = np.median(random_heart_failure_counts)
heart_failure_perm_min = np.min(random_heart_failure_counts)
heart_failure_perm_max = np.max(random_heart_failure_counts)
heart_failure_perm_std = np.std(random_heart_failure_counts)

# Analyze results and calculate p-value for Heart Failure & Enlarged Heart
count_extreme_values_hf = np.sum(
    np.array(random_heart_failure_counts) >= observed_heart_failure_count
)
empirical_p_value_hf = count_extreme_values_hf / N_HEART_FAILURE_PERMUTATIONS

print(
    f"\n--- PERMUTATION TEST RESULTS (ROBUST METHOD) for Heart Failure & Enlarged Heart ---"
)
print(f"Observed event count in the true case group: {observed_heart_failure_count}")
print(
    f"Times a random group had >= {observed_heart_failure_count} events: {count_extreme_values_hf} out of {N_HEART_FAILURE_PERMUTATIONS}"
)
print(f"Empirical P-Value: {empirical_p_value_hf:.4f}")
print(
    f"Average individuals with Heart Failure & Enlarged Heart per random group: {heart_failure_perm_mean:.2f}"
)
print(
    f"Median individuals with Heart Failure & Enlarged Heart per random group: {heart_failure_perm_median}"
)
print(
    f"Min individuals with Heart Failure & Enlarged Heart in a sample: {heart_failure_perm_min}"
)
print(
    f"Max individuals with Heart Failure & Enlarged Heart in a sample: {heart_failure_perm_max}"
)
print(
    f"Standard deviation of Heart Failure & Enlarged Heart counts: {heart_failure_perm_std:.2f}"
)

print("\n--- Summary of Raw Data for 'Heart Failure & Enlarged Heart' Permutations ---")
heart_failure_counts_summary = (
    pd.Series(random_heart_failure_counts).value_counts().sort_index()
)
print(heart_failure_counts_summary.to_string())

# Plot for Heart Failure & Enlarged Heart Quantification
print("\nGenerating permutation plot for Heart Failure & Enlarged Heart...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
sns.histplot(
    data=random_heart_failure_counts,
    ax=ax,
    discrete=True,
    stat="density",
    color="#E67E22",
)
ax.axvline(
    x=observed_heart_failure_count,
    color="#C0392B",
    linestyle="--",
    linewidth=2,
    label=f"Observed Event Count ({observed_heart_failure_count})",
)
ax.legend()
ax.set_title(
    f"Distribution of Heart Failure & Enlarged Heart Counts in Random Groups ({N_HEART_FAILURE_PERMUTATIONS} Permutations)",
    fontsize=14,
    weight="bold",
)
ax.set_xlabel(
    f"Number of People with Heart Failure & Enlarged Heart (in a random group of {num_cases})",
    fontsize=12,
)
ax.set_ylabel("Density (Proportion of Permutations)", fontsize=12)
plt.show()

### -----------------------------------------------------------------------
### FINAL SUMMARY TABLE
### -----------------------------------------------------------------------
print("\n--- Summary of Key Results ---")
summary_data = {
    "Metric": [
        "Major CVD Empirical P-Value",
        "Average Major CVD Count (from 10000 Permutations)",
        "Heart Failure & Enlarged Heart Empirical P-Value",
        "Average Heart Failure & Enlarged Heart Count (from 10000 Permutations)",
    ],
    "Value": [
        f"{empirical_p_value_major_cvd:.4f}",
        f"{major_cvd_perm_mean:.2f}",
        f"{empirical_p_value_hf:.4f}",
        f"{heart_failure_perm_mean:.2f}",
    ],
}
results_summary_df = pd.DataFrame(summary_data)
print(results_summary_df.to_string(index=False))

print("\n--- Script Finished. ---")
