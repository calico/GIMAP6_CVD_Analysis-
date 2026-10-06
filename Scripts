import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy.stats import fisher_exact

# Suppress FutureWarning from seaborn if it's still appearing
import warnings

warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")
warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.",
)

# =========================
# Variant definitions
# =========================
VARIANTS = {
    "GIMAP6_V65I": ["7-150628405-C-T"],
    "TTR_V122I": ["18-31598655-G-A"],
    "GIMAP6_Q237R": ["7-150627888-T-C"],  # New control variant
}
# Note: LDLR pathogenic is handled separately in create_query and will remain.

# =========================
# Disease concept IDs
# =========================
# Define common sets of concept IDs for clarity and reusability
CAD_CONCEPTS = [
    40481919,
    40479625,
    312934,
    315558,
    36712955,
    37109252,
    312934,
    4124834,
    321882,
    4187067,
    317576,
    764123,
    316995,
    44782712,
    4134723,
]
HF_CONCEPTS = [316139]
ISCHEMIC_HEART_DISEASE_CONCEPTS = [
    4185932,  # Ischemic heart disease
    321318,  # Angina pectoris
    319844,  # Acute ischemic heart disease
    315286,  # Chronic ischemic heart disease
]
HYPERCHOLESTEROLEMIA_CONCEPTS = [4029305, 35207060, 37200312, 44834564]
DIABETES_CONCEPTS = [201820, 201826]
HYPERTENSION_CONCEPTS = [316866]
CKD_CONCEPTS = [46271022, 198124]
ASTHMA_CONCEPTS = [317009]
DVT_CONCEPTS = [4133004]
VASCULITIS_CONCEPTS = [4137275]
CANCER_CONCEPTS = [443392]
CHEST_PAIN_CONCEPTS = [77670]
ECG_ABNORMAL_CONCEPTS = [320536]
CARDIOMEGALY_CONCEPTS = [314658]
CEREBROVASCULAR_DISEASE_CONCEPTS = [381591]

DISEASES = {
    # Primary Diseases
    "Coronary atherosclerosis": CAD_CONCEPTS,
    "Heart failure overall": HF_CONCEPTS,
    "CAD or HF": {
        "OR_concepts": CAD_CONCEPTS + HF_CONCEPTS
    },  # New: OR condition for combined concepts
    "CAD and HF": {
        "AND_concepts": [CAD_CONCEPTS, HF_CONCEPTS]
    },  # Existing: AND condition for combined concepts
    "Ischemic heart disease (Overall)": ISCHEMIC_HEART_DISEASE_CONCEPTS,
    "Hypercholesterolemia": HYPERCHOLESTEROLEMIA_CONCEPTS,
    # Positive Control Diseases (known associations or high prevalence)
    "Diabetes mellitus (Overall)": DIABETES_CONCEPTS,
    "Hypertensive disorder": HYPERTENSION_CONCEPTS,
    "Chronic kidney disease": CKD_CONCEPTS,
    # Negative Control Diseases (expected no association with studied variants/CV)
    "Asthma": ASTHMA_CONCEPTS,
    "Deep venous thrombosis": DVT_CONCEPTS,
    "Vasculitis": VASCULITIS_CONCEPTS,
    # Other relevant conditions (for general context/validation)
    "Malignant neoplastic disease": CANCER_CONCEPTS,
    "Chest pain": CHEST_PAIN_CONCEPTS,
    "Electrocardiogram abnormal": ECG_ABNORMAL_CONCEPTS,
    "Cardiomegaly": CARDIOMEGALY_CONCEPTS,
    "Cerebrovascular disease": CEREBROVASCULAR_DISEASE_CONCEPTS,
}


def create_query(workspace_cdr, disease_definition):
    """
    Constructs a BigQuery SQL query to retrieve patient data including demographics,
    disease status, genetic variant carrier status, and last visit datetime.
    This function is updated to handle single-list disease concepts,
    combined (OR'd) disease definitions, and combined (AND'd) disease definitions.

    Args:
        workspace_cdr (str): The BigQuery dataset path for the workspace CDR.
        disease_definition (list or dict): A list of concept IDs for single diseases (OR logic),
                                           or a dictionary for combined diseases (e.g., {"OR_concepts": [...]} or {"AND_concepts": [[...], [...]]}).
    Returns:
        str: The SQL query string.
    """
    disease_cases_sql = ""

    if isinstance(disease_definition, list):
        # Standard single disease case (OR logic for all concept IDs in the list)
        concept_id_list = ", ".join(map(str, disease_definition))
        disease_cases_sql = f"""
          SELECT DISTINCT person_id
          FROM `{workspace_cdr}.cb_search_all_events`
          WHERE concept_id IN ({concept_id_list})
        """
    elif isinstance(disease_definition, dict):
        if "OR_concepts" in disease_definition:
            # Combined disease case using OR logic (UNION ALL)
            concept_id_list = ", ".join(map(str, disease_definition["OR_concepts"]))
            disease_cases_sql = f"""
                SELECT DISTINCT person_id
                FROM `{workspace_cdr}.cb_search_all_events`
                WHERE concept_id IN ({concept_id_list})
            """
        elif "AND_concepts" in disease_definition:
            # Combined disease case using AND logic (INNER JOIN)
            # Each item in "AND_concepts" list must be a list of concept IDs for one condition
            join_clauses = []
            for i, concepts_for_condition in enumerate(
                disease_definition["AND_concepts"]
            ):
                concept_list_str = ", ".join(map(str, concepts_for_condition))
                alias = f"t{i}"  # Alias for each sub-table

                # Each subquery forms a CTE or sub-select for joining
                join_clauses.append(
                    f"""
                    SELECT DISTINCT person_id
                    FROM `{workspace_cdr}.cb_search_all_events`
                    WHERE concept_id IN ({concept_list_str})
                """
                )

            # Construct the final SQL for AND logic using INNER JOINs
            if not join_clauses:
                # If no conditions provided, return an empty result set by ensuring no rows match
                disease_cases_sql = (
                    "SELECT DISTINCT person_id FROM `{workspace_cdr}.person` WHERE 1=0"
                )
            elif len(join_clauses) == 1:
                # If only one condition, no join needed
                disease_cases_sql = join_clauses[0]
            else:
                # Build a series of INNER JOINs
                # Start with the first subquery, aliased as t0
                base_query = f"SELECT t0.person_id FROM ({join_clauses[0]}) AS t0"

                # Add subsequent INNER JOINs
                for i in range(1, len(join_clauses)):
                    base_query += f"\nINNER JOIN ({join_clauses[i]}) AS t{i} ON t0.person_id = t{i}.person_id"

                disease_cases_sql = base_query

        else:
            raise ValueError(
                "Invalid dictionary format for disease_definition. Must contain 'OR_concepts' or 'AND_concepts' key."
            )
    else:
        raise ValueError(
            "disease_definition must be a list of concept IDs or a dictionary for combined diseases."
        )

    # All defined VARIANTS are included now, including the control variant
    all_variant_queries = []
    for label, vids in VARIANTS.items():
        vid_list = ", ".join([f"'{vid}'" for vid in vids])
        all_variant_queries.append(
            f"""
            SELECT person_id, '{label}' AS variant_label
            FROM `{workspace_cdr}.cb_variant_to_person`
            CROSS JOIN UNNEST(person_ids) AS person_id
            WHERE vid IN ({vid_list})
        """
        )

    # LDLR pathogenic variant query (remains unchanged)
    ldlr_sql = f"""
        SELECT person_id, 'LDLR_pathogenic' AS variant_label
        FROM `{workspace_cdr}.cb_variant_to_person` AS vtp
        CROSS JOIN UNNEST(vtp.person_ids) AS person_id
        WHERE vtp.vid IN (
            SELECT DISTINCT va.vid
            FROM `{workspace_cdr}.cb_variant_attribute` va
            JOIN `{workspace_cdr}.cb_variant_attribute_genes` vag
              ON va.vid = vag.vid
            WHERE UPPER(vag.gene_symbol) = 'LDLR'
              AND (LOWER(va.clinical_significance_string) LIKE '%pathogenic%'
                   OR LOWER(va.clinical_significance_string) LIKE '%likely pathogenic%')
        )
    """
    union_sql = "\nUNION ALL\n".join(
        all_variant_queries + [ldlr_sql]
    )  # Union with all VARIANTS and LDLR

    return f"""
    WITH population AS (
      SELECT
        p.person_id,
        g1.concept_name AS gender,
        g2.concept_name AS race,
        p.birth_datetime,
        MAX(vo.visit_end_datetime) AS last_visit_datetime
      FROM `{workspace_cdr}.person` p
      JOIN `{workspace_cdr}.cb_search_person` cbsp ON p.person_id = cbsp.person_id
      LEFT JOIN `{workspace_cdr}.concept` g1 ON p.gender_concept_id = g1.concept_id
      LEFT JOIN `{workspace_cdr}.concept` g2 ON p.race_concept_id = g2.concept_id
      LEFT JOIN `{workspace_cdr}.visit_occurrence` vo ON p.person_id = vo.person_id
      WHERE p.race_concept_id = 8516 -- Filters for race: 8516 is 'Black or African American'
        AND cbsp.has_ehr_data = 1
        AND cbsp.has_whole_genome_variant = 1
      GROUP BY p.person_id, g1.concept_name, g2.concept_name, p.birth_datetime
    ),
    Disease_cases AS (
      {disease_cases_sql}
    ),
    variant_carriers AS (
      {union_sql}
    )
    SELECT
      pop.person_id, pop.gender, pop.race, pop.birth_datetime, pop.last_visit_datetime,
      CASE WHEN d.person_id IS NOT NULL THEN 1 ELSE 0 END AS has_Disease,
      vc.variant_label
    FROM population pop
    LEFT JOIN Disease_cases d           ON pop.person_id = d.person_id
    LEFT JOIN variant_carriers vc ON pop.person_id = vc.person_id
    """


def calculate_age(df):
    """
    Calculates age for each individual based on birth_datetime and last_visit_datetime.
    Filters for ages between 18 and 100.

    Args:
        df (pd.DataFrame): DataFrame containing 'person_id', 'birth_datetime', and 'last_visit_datetime'.
    Returns:
        pd.DataFrame: DataFrame with 'age' column added, filtered for ages between 18 and 100.
    """
    df["birth_datetime"] = pd.to_datetime(df["birth_datetime"])
    df["last_visit_datetime"] = pd.to_datetime(df["last_visit_datetime"])

    df = df.dropna(subset=["last_visit_datetime"]).copy()

    df["age"] = df["last_visit_datetime"].dt.year - df["birth_datetime"].dt.year

    df["age"] = np.where(
        (df["last_visit_datetime"].dt.month < df["birth_datetime"].dt.month)
        | (
            (df["last_visit_datetime"].dt.month == df["birth_datetime"].dt.month)
            & (df["last_visit_datetime"].dt.day < df["birth_datetime"].dt.day)
        ),
        df["age"] - 1,
        df["age"],
    )

    return df[df["age"].between(18, 100)].copy()


def compute_overall_prevalence(df):
    """
    Calculates the overall prevalence of the disease in the entire filtered cohort.
    Args:
        df (pd.DataFrame): The DataFrame after all initial filtering (race, EHR, WGS, age 18-100).
                            Needs 'person_id' and 'has_Disease'.
    Returns:
        float: The overall disease prevalence as a percentage.
    """
    total_individuals = df["person_id"].nunique()
    total_disease_cases = df[df["has_Disease"] == 1]["person_id"].nunique()

    if total_individuals == 0:
        return 0.0
    else:
        return (total_disease_cases / total_individuals) * 100


def compute_cumulative_prevalence(df, age_cutoffs=None):
    """
    Computes the cumulative prevalence of the disease at different age cutoffs
    for each *defined variant carrier group*, the *entire cohort*, and *gender-specific cohorts*.
    Includes variant-specific cumulative prevalence broken down by gender, and gender-specific non-carriers.

    Args:
        df (pd.DataFrame): Processed DataFrame with 'age', 'has_Disease', 'gender', and 'variant_label'.
                            'variant_label' will be None for non-carriers.
        age_cutoffs (list): A list of age thresholds for cumulative prevalence calculation.
    Returns:
        pd.DataFrame: A DataFrame with cumulative prevalence at specified age cutoffs
                      for various groups, including gender-specific variant carriers and non-carriers.
    """
    if age_cutoffs is None:
        age_cutoffs = list(range(35, 86, 5))  # Defaulting to 85 as the max
        if 85 not in age_cutoffs:
            age_cutoffs.append(85)
            age_cutoffs = sorted(list(set(age_cutoffs)))

    results = []
    temp_df = df.copy()
    temp_df["variant_label"] = temp_df["variant_label"].fillna("Non-carrier")

    # Order of variant labels for display
    ordered_variant_labels_main = list(VARIANTS.keys()) + ["LDLR_pathogenic"]
    ordered_variant_labels_all = ordered_variant_labels_main + [
        "Non-carrier"
    ]  # Include non-carrier for overall

    # 1. Cumulative prevalence for each overall variant carrier group (including 'Non-carrier')
    for variant in ordered_variant_labels_all:
        carrier_df = temp_df[temp_df["variant_label"] == variant].copy()
        if carrier_df.empty:
            continue
        for ac in age_cutoffs:
            sub_carriers = carrier_df[carrier_df["age"] < ac]
            if sub_carriers.empty:
                continue
            total_carriers_at_cutoff = sub_carriers["person_id"].nunique()
            cases_among_carriers = sub_carriers[sub_carriers["has_Disease"] == 1][
                "person_id"
            ].nunique()
            prevalence = (
                (cases_among_carriers / total_carriers_at_cutoff) * 100
                if total_carriers_at_cutoff > 0
                else 0.0
            )
            results.append(
                {
                    "variant_label": variant,
                    "total": total_carriers_at_cutoff,
                    "cases": cases_among_carriers,
                    "age_cutoff": ac,
                    "prevalence": prevalence,
                    "group_type": "Variant Overall",
                }
            )

    # 2. Cumulative prevalence for the entire cohort (overall)
    for ac in age_cutoffs:
        sub_cohort = df[df["age"] < ac].copy()
        if sub_cohort.empty:
            continue
        total_cohort_at_cutoff = sub_cohort["person_id"].nunique()
        cases_among_cohort = sub_cohort[sub_cohort["has_Disease"] == 1][
            "person_id"
        ].nunique()
        prevalence = (
            (cases_among_cohort / total_cohort_at_cutoff) * 100
            if total_cohort_at_cutoff > 0
            else 0.0
        )
        results.append(
            {
                "variant_label": "Overall Cohort Cumulative",
                "total": total_cohort_at_cutoff,
                "cases": cases_among_cohort,
                "age_cutoff": ac,
                "prevalence": prevalence,
                "group_type": "Cohort Overall",
            }
        )

    # 3. Cumulative prevalence for Male and Female overall cohorts
    for gender_label in ["Male", "Female"]:
        gender_df = df[df["gender"] == gender_label].copy()
        if gender_df.empty:
            continue
        for ac in age_cutoffs:
            sub_gender_cohort = gender_df[gender_df["age"] < ac].copy()
            if sub_gender_cohort.empty:
                continue
            total_gender_at_cutoff = sub_gender_cohort["person_id"].nunique()
            cases_among_gender = sub_gender_cohort[
                sub_gender_cohort["has_Disease"] == 1
            ]["person_id"].nunique()
            prevalence = (
                (cases_among_gender / total_gender_at_cutoff) * 100
                if total_gender_at_cutoff > 0
                else 0.0
            )
            results.append(
                {
                    "variant_label": f"{gender_label} Cohort Cumulative",
                    "total": total_gender_at_cutoff,
                    "cases": cases_among_gender,
                    "age_cutoff": ac,
                    "prevalence": prevalence,
                    "group_type": f"{gender_label} Cohort",
                }
            )

    # 4. Cumulative prevalence for specific variants broken down by gender, INCLUDING gender-specific non-carriers
    variants_for_gender_breakdown = ordered_variant_labels_main + ["Non-carrier"]
    for variant in variants_for_gender_breakdown:
        for gender_label in ["Male", "Female"]:
            gender_variant_df = temp_df[
                (temp_df["variant_label"] == variant)
                & (temp_df["gender"] == gender_label)
            ].copy()
            if gender_variant_df.empty:
                continue
            for ac in age_cutoffs:
                sub_gender_variant = gender_variant_df[gender_variant_df["age"] < ac]
                if sub_gender_variant.empty:
                    continue
                total_gender_variant_at_cutoff = sub_gender_variant[
                    "person_id"
                ].nunique()
                cases_among_gender_variant = sub_gender_variant[
                    sub_gender_variant["has_Disease"] == 1
                ]["person_id"].nunique()
                prevalence = (
                    (cases_among_gender_variant / total_gender_variant_at_cutoff) * 100
                    if total_gender_variant_at_cutoff > 0
                    else 0.0
                )

                # Adjust label for gender-specific non-carrier
                label_prefix = variant
                if variant == "Non-carrier":
                    label_prefix = "Non-carrier"  # Keep it consistent

                results.append(
                    {
                        "variant_label": f"{label_prefix} {gender_label} Cumulative",
                        "total": total_gender_variant_at_cutoff,
                        "cases": cases_among_gender_variant,
                        "age_cutoff": ac,
                        "prevalence": prevalence,
                        "group_type": f"{label_prefix} {gender_label}",
                    }
                )

    if not results:
        return pd.DataFrame(
            columns=[
                "variant_label",
                "total",
                "cases",
                "age_cutoff",
                "prevalence",
                "group_type",
            ]
        )

    cdf = pd.DataFrame(results)

    # Sort the DataFrame for better readability, putting variant cumulative first
    sort_order = (
        ordered_variant_labels_main
        + ["Non-carrier"]
        + [
            f"{v} Male Cumulative"
            for v in ordered_variant_labels_main + ["Non-carrier"]
        ]
        + [
            f"{v} Female Cumulative"
            for v in ordered_variant_labels_main + ["Non-carrier"]
        ]
        + [
            "Overall Cohort Cumulative",
            "Male Cohort Cumulative",
            "Female Cohort Cumulative",
        ]
    )

    # Create a categorical type for sorting
    cdf["variant_label"] = pd.Categorical(
        cdf["variant_label"], categories=sort_order, ordered=True
    )
    cdf = cdf.sort_values(by=["variant_label", "age_cutoff"]).reset_index(drop=True)

    return cdf


def calculate_or_p_values_at_age_cutoffs(df, age_cutoffs=None):
    """
    Calculates Odds Ratios and p-values for each variant vs. non-carriers
    at specified age cutoffs, for overall, male, and female cohorts.

    Args:
        df (pd.DataFrame): Processed DataFrame with 'age', 'has_Disease', 'gender', and 'variant_label'.
        age_cutoffs (list): A list of age thresholds for calculation.
    Returns:
        pd.DataFrame: A DataFrame with ORs, 95% CIs, and p-values.
    """
    if age_cutoffs is None:
        age_cutoffs = list(range(35, 86, 5))  # Defaulting to 85 as the max
        if 85 not in age_cutoffs:
            age_cutoffs.append(85)
            age_cutoffs = sorted(list(set(age_cutoffs)))

    results = []
    temp_df = df.copy()
    temp_df["variant_label"] = temp_df["variant_label"].fillna("Non-carrier")

    target_variants = list(VARIANTS.keys()) + [
        "LDLR_pathogenic"
    ]  # Variants to compare against non-carriers
    groups = {"Overall": None, "Male": "Male", "Female": "Female"}

    for group_name, gender_filter in groups.items():
        for variant in target_variants:
            for ac in age_cutoffs:
                sub_df = temp_df[temp_df["age"] < ac].copy()
                if gender_filter:
                    sub_df = sub_df[sub_df["gender"] == gender_filter].copy()

                if sub_df.empty:
                    results.append(
                        {
                            "Group": group_name,
                            "Variant": variant,
                            "Age Cutoff": ac,
                            "OR": np.nan,
                            "Lower CI": np.nan,
                            "Upper CI": np.nan,
                            "P-value": np.nan,
                            "Target Cases": 0,
                            "Target Non-Cases": 0,
                            "Non-carrier Cases": 0,
                            "Non-carrier Non-Cases": 0,
                        }
                    )
                    continue

                # Counts for target variant
                target_carrier_df = sub_df[sub_df["variant_label"] == variant]
                target_cases = target_carrier_df[
                    target_carrier_df["has_Disease"] == 1
                ].shape[0]
                target_non_cases = target_carrier_df[
                    target_carrier_df["has_Disease"] == 0
                ].shape[0]

                # Counts for non-carriers within the specific gender/age sub_df
                non_carrier_df = sub_df[sub_df["variant_label"] == "Non-carrier"]
                non_carrier_cases = non_carrier_df[
                    non_carrier_df["has_Disease"] == 1
                ].shape[0]
                non_carrier_non_cases = non_carrier_df[
                    non_carrier_df["has_Disease"] == 0
                ].shape[0]

                table = [
                    [target_cases, target_non_cases],
                    [non_carrier_cases, non_carrier_non_cases],
                ]

                or_val, p_val, lower_ci, upper_ci = np.nan, np.nan, np.nan, np.nan

                # Check if Fisher's exact test is possible
                if all(sum(row) > 0 for row in table) and all(
                    sum(table[i][j] for i in range(2)) > 0 for j in range(2)
                ):
                    try:
                        oddsratio, p_value = fisher_exact(
                            table, alternative="two-sided"
                        )
                        or_val = oddsratio
                        p_val = p_value

                        # Calculate CI only if OR is valid and not zero counts in cells.
                        # Adding more robust checks for CI calculation to avoid log(0) or division by zero.
                        if (
                            oddsratio is not None
                            and oddsratio > 0
                            and target_cases > 0
                            and target_non_cases > 0
                            and non_carrier_cases > 0
                            and non_carrier_non_cases > 0
                        ):
                            se_log_or = np.sqrt(
                                1 / target_cases
                                + 1 / target_non_cases
                                + 1 / non_carrier_cases
                                + 1 / non_carrier_non_cases
                            )
                            log_or = np.log(oddsratio)
                            lower_ci = np.exp(log_or - 1.96 * se_log_or)
                            upper_ci = np.exp(log_or + 1.96 * se_log_or)
                    except ValueError:
                        # Happens if table has a zero margin, e.g., all cases/controls are in one group
                        pass  # OR and CI will remain NaN

                results.append(
                    {
                        "Group": group_name,
                        "Variant": variant,
                        "Age Cutoff": ac,
                        "OR": or_val,
                        "Lower CI": lower_ci,
                        "Upper CI": upper_ci,
                        "P-value": p_val,
                        "Target Cases": target_cases,
                        "Target Non-Cases": target_non_cases,
                        "Non-carrier Cases": non_carrier_cases,
                        "Non-carrier Non-Cases": non_carrier_non_cases,
                    }
                )

    if not results:
        return pd.DataFrame()  # Return empty DataFrame if no results

    or_df = pd.DataFrame(results)

    # Order variants for display
    ordered_variants_for_or = list(VARIANTS.keys()) + ["LDLR_pathogenic"]
    or_df["Variant"] = pd.Categorical(
        or_df["Variant"], categories=ordered_variants_for_or, ordered=True
    )

    # Sort by Variant, then Group, then Age Cutoff
    or_df = or_df.sort_values(by=["Variant", "Group", "Age Cutoff"]).reset_index(
        drop=True
    )

    return or_df


# --- Plotting Functions with Standardized Colors and Y-axis ---
# Define a consistent color palette for all plots
VARIANT_COLORS = {
    "GIMAP6_V65I": "red",
    "LDLR_pathogenic": "orange",
    "TTR_V122I": "blue",
    "GIMAP6_Q237R": "purple",  # New control variant color
    "Non-carrier": "black",  # Overall non-carrier
    "Overall Cohort Cumulative": "green",
    "Male Cohort Cumulative": "darkgreen",  # Distinct but related to overall
    "Female Cohort Cumulative": "teal",  # Distinct but related to overall
    # Gender-specific variant colors (can be darker shades or same as overall variant color)
    "GIMAP6_V65I Male Cumulative": "firebrick",  # Darker red
    "GIMAP6_V65I Female Cumulative": "lightcoral",  # Lighter red
    "LDLR_pathogenic Male Cumulative": "darkorange",  # Darker orange
    "LDLR_pathogenic Female Cumulative": "sandybrown",  # Lighter orange
    "TTR_V122I Male Cumulative": "darkblue",  # Darker blue
    "TTR_V122I Female Cumulative": "cornflowerblue",  # Lighter blue
    "GIMAP6_Q237R Male Cumulative": "darkviolet",  # Darker purple
    "GIMAP6_Q237R Female Cumulative": "plum",  # Lighter purple
    # Gender-specific non-carrier colors
    "Non-carrier Male Cumulative": "dimgray",  # Darker gray for male non-carrier
    "Non-carrier Female Cumulative": "lightgray",  # Lighter gray for female non-carrier
}

# Define a consistent line style for all plots
LINE_STYLES = {
    "GIMAP6_V65I": "-",
    "LDLR_pathogenic": "-",
    "TTR_V122I": "-",
    "GIMAP6_Q237R": "-",
    "Non-carrier": "--",
    "Overall Cohort Cumulative": ":",
    "Male Cohort Cumulative": "-.",
    "Female Cohort Cumulative": "-.",
    "GIMAP6_V65I Male Cumulative": "-",
    "GIMAP6_V65I Female Cumulative": "-",
    "LDLR_pathogenic Male Cumulative": "-",
    "LDLR_pathogenic Female Cumulative": "-",
    "TTR_V122I Male Cumulative": "-",
    "TTR_V122I Female Cumulative": "-",
    "GIMAP6_Q237R Male Cumulative": "-",
    "GIMAP6_Q237R Female Cumulative": "-",
    "Non-carrier Male Cumulative": "--",
    "Non-carrier Female Cumulative": "--",
}

# Define markers
MARKERS = {
    "GIMAP6_V65I": "o",
    "LDLR_pathogenic": "D",
    "TTR_V122I": "^",
    "GIMAP6_Q237R": "s",
    "Non-carrier": "P",
    "Overall Cohort Cumulative": "X",
    "Male Cohort Cumulative": "*",
    "Female Cohort Cumulative": "p",
    "GIMAP6_V65I Male Cumulative": "o",
    "GIMAP6_V65I Female Cumulative": "o",
    "LDLR_pathogenic Male Cumulative": "D",
    "LDLR_pathogenic Female Cumulative": "D",
    "TTR_V122I Male Cumulative": "^",
    "TTR_V122I Female Cumulative": "^",
    "GIMAP6_Q237R Male Cumulative": "s",
    "GIMAP6_Q237R Female Cumulative": "s",
    "Non-carrier Male Cumulative": "P",
    "Non-carrier Female Cumulative": "P",
}


def plot_overall_focused_cumulative_prevalence(
    cdf, disease, overall_prevalence_value, ax, y_max
):
    """
    Generates a line plot focusing on cumulative disease prevalence for
    the overall cohort and overall GIMAP6_V65I, TTR_V122I, LDLR_pathogenic, and GIMAP6_Q237R carriers on a given Axes.
    """
    focused_labels = [
        "GIMAP6_V65I",
        "TTR_V122I",
        "LDLR_pathogenic",
        "GIMAP6_Q237R",  # New control variant
        "Non-carrier",  # Overall non-carrier for context
        "Overall Cohort Cumulative",  # Overall cohort at the end
    ]

    plot_data_focused = cdf[cdf["variant_label"].isin(focused_labels)].copy()
    plot_data_focused["variant_label"] = pd.Categorical(
        plot_data_focused["variant_label"], categories=focused_labels, ordered=True
    )
    plot_data_focused = plot_data_focused.sort_values("variant_label")

    for label in focused_labels:
        data_subset = plot_data_focused[plot_data_focused["variant_label"] == label]
        if not data_subset.empty:
            sns.lineplot(
                data=data_subset,
                x="age_cutoff",
                y="prevalence",
                color=VARIANT_COLORS.get(label, "gray"),
                linestyle=LINE_STYLES.get(label, "-"),
                marker=MARKERS.get(label, "o"),
                linewidth=2.5,
                markersize=7,  # Increased line and marker size
                label=label,
                ax=ax,
            )

    ax.axhline(
        y=overall_prevalence_value,
        color="red",
        linestyle="--",
        linewidth=2.0,  # Increased line width
        label=f"Overall Cohort Prevalence (Static: {overall_prevalence_value:.2f}%)",
    )

    ax.axvline(
        x=60,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Early-Onset Threshold (Age 60)",
    )

    ax.set_title(f"Overall: {disease}", fontsize=15)  # Larger title
    ax.set_xlabel("Age Cutoff", fontsize=13)  # Larger label
    ax.set_ylabel("Cumulative Prevalence (%)", fontsize=13)  # Larger label
    ax.tick_params(axis="x", labelsize=11)  # Larger tick labels
    ax.tick_params(axis="y", labelsize=11)  # Larger tick labels
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(
        title="Group", loc="upper left", fontsize=10, title_fontsize=12
    )  # Larger legend text and title
    ax.set_xticks(sorted(plot_data_focused["age_cutoff"].unique()))
    ax.set_ylim(0, y_max)  # Set consistent Y-axis limit


def plot_male_focused_cumulative_prevalence(
    cdf, disease, overall_prevalence_value, ax, y_max
):
    """
    Generates a line plot focusing on cumulative disease prevalence for
    the Male cohort and Male GIMAP6_V65I, TTR_V122I, LDLR_pathogenic, GIMAP6_Q237R and Non-carrier Male Cumulative.
    """
    focused_labels = [
        "GIMAP6_V65I Male Cumulative",
        "TTR_V122I Male Cumulative",
        "LDLR_pathogenic Male Cumulative",
        "GIMAP6_Q237R Male Cumulative",  # New control variant for male
        "Non-carrier Male Cumulative",  # Gender-specific non-carrier
        "Male Cohort Cumulative",  # Overall male cohort at the end
    ]

    plot_data_focused = cdf[cdf["variant_label"].isin(focused_labels)].copy()
    plot_data_focused["variant_label"] = pd.Categorical(
        plot_data_focused["variant_label"], categories=focused_labels, ordered=True
    )
    plot_data_focused = plot_data_focused.sort_values("variant_label")

    for label in focused_labels:
        data_subset = plot_data_focused[plot_data_focused["variant_label"] == label]
        if not data_subset.empty:
            sns.lineplot(
                data=data_subset,
                x="age_cutoff",
                y="prevalence",
                color=VARIANT_COLORS.get(label, "gray"),
                linestyle=LINE_STYLES.get(label, "-"),
                marker=MARKERS.get(label, "o"),
                linewidth=2.5,
                markersize=7,
                label=label,
                ax=ax,
            )

    ax.axhline(
        y=overall_prevalence_value,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Overall Cohort Prevalence (Static: {overall_prevalence_value:.2f}%)",
    )

    ax.axvline(
        x=55,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Male Early-Onset Threshold (Age 55)",
    )

    ax.set_title(f"Male: {disease}", fontsize=15)
    ax.set_xlabel("Age Cutoff", fontsize=13)
    ax.set_ylabel("Cumulative Prevalence (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Group", loc="upper left", fontsize=10, title_fontsize=12)
    ax.set_xticks(sorted(plot_data_focused["age_cutoff"].unique()))
    ax.set_ylim(0, y_max)


def plot_female_focused_cumulative_prevalence(
    cdf, disease, overall_prevalence_value, ax, y_max
):
    """
    Generates a line plot focusing on cumulative disease prevalence for
    the Female cohort and Female GIMAP6_V65I, TTR_V122I, LDLR_pathogenic, GIMAP6_Q237R and Non-carrier Female Cumulative.
    """
    focused_labels = [
        "GIMAP6_V65I Female Cumulative",
        "TTR_V122I Female Cumulative",
        "LDLR_pathogenic Female Cumulative",
        "GIMAP6_Q237R Female Cumulative",  # New control variant for female
        "Non-carrier Female Cumulative",  # Gender-specific non-carrier
        "Female Cohort Cumulative",
    ]

    plot_data_focused = cdf[cdf["variant_label"].isin(focused_labels)].copy()
    plot_data_focused["variant_label"] = pd.Categorical(
        plot_data_focused["variant_label"], categories=focused_labels, ordered=True
    )
    plot_data_focused = plot_data_focused.sort_values("variant_label")

    for label in focused_labels:
        data_subset = plot_data_focused[plot_data_focused["variant_label"] == label]
        if not data_subset.empty:
            sns.lineplot(
                data=data_subset,
                x="age_cutoff",
                y="prevalence",
                color=VARIANT_COLORS.get(label, "gray"),
                linestyle=LINE_STYLES.get(label, "-"),
                marker=MARKERS.get(label, "o"),
                linewidth=2.5,
                markersize=7,
                label=label,
                ax=ax,
            )

    ax.axhline(
        y=overall_prevalence_value,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Overall Cohort Prevalence (Static: {overall_prevalence_value:.2f}%)",
    )

    ax.axvline(
        x=65,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Female Early-Onset Threshold (Age 65)",
    )

    ax.set_title(f"Female: {disease}", fontsize=15)
    ax.set_xlabel("Age Cutoff", fontsize=13)
    ax.set_ylabel("Cumulative Prevalence (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Group", loc="upper left", fontsize=10, title_fontsize=12)
    ax.set_xticks(sorted(plot_data_focused["age_cutoff"].unique()))
    ax.set_ylim(0, y_max)


def plot_combined_focused_prevalence(cdf, disease, overall_prevalence_value):
    """
    Generates a combined figure with horizontal subplots for overall, male, and female
    focused cumulative prevalence, with a consistent Y-axis scale.
    """
    # Determine the maximum prevalence across all relevant groups for consistent Y-axis
    all_relevant_labels_for_ymax = (
        list(VARIANTS.keys())
        + [
            "LDLR_pathogenic",
            "Non-carrier",  # Overall non-carrier
            "Overall Cohort Cumulative",
            "Male Cohort Cumulative",
            "Female Cohort Cumulative",
        ]
        + [f"{v} Male Cumulative" for v in list(VARIANTS.keys()) + ["LDLR_pathogenic"]]
        + [
            f"{v} Female Cumulative"
            for v in list(VARIANTS.keys()) + ["LDLR_pathogenic"]
        ]
        + ["Non-carrier Male Cumulative", "Non-carrier Female Cumulative"]
    )  # Gender-specific non-carriers

    # Filter cdf to only include these relevant labels for y_max calculation
    filtered_cdf_for_ymax = cdf[
        cdf["variant_label"].isin(all_relevant_labels_for_ymax)
    ].copy()

    if not filtered_cdf_for_ymax.empty:
        y_max = filtered_cdf_for_ymax["prevalence"].max() * 1.10  # Add 10% buffer
        if (
            y_max < 10
        ):  # Ensure minimum Y-axis is at least 10% if prevalence is very low
            y_max = 10
    else:
        y_max = 10  # Default to 10 if no data

    fig, axes = plt.subplots(
        1, 3, figsize=(26, 8)
    )  # Adjusted figure size for more space

    # Pass y_max to each plotting function
    plot_overall_focused_cumulative_prevalence(
        cdf, disease, overall_prevalence_value, axes[0], y_max
    )
    plot_male_focused_cumulative_prevalence(
        cdf, disease, overall_prevalence_value, axes[1], y_max
    )
    plot_female_focused_cumulative_prevalence(
        cdf, disease, overall_prevalence_value, axes[2], y_max
    )

    fig.suptitle(
        f"Cumulative Prevalence by Age for {disease}", fontsize=22
    )  # Larger main title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def run_analysis_per_disease(workspace_cdr):
    """
    Orchestrates the analysis workflow, including overall, gender-specific, and age-banded
    odds ratio for GIMAP6_V65I, plus cumulative prevalence and plotting.

    Args:
        workspace_cdr (str): The BigQuery dataset path for the workspace CDR.
    """
    # Set the maximum age cutoff for analysis to 85
    age_cutoffs_for_analysis = list(range(35, 86, 5))  # Go up to and including 85
    if 85 not in age_cutoffs_for_analysis:  # Ensure 85 is always included
        age_cutoffs_for_analysis.append(85)
        age_cutoffs_for_analysis = sorted(list(set(age_cutoffs_for_analysis)))

    # Use an ordered list of diseases to ensure "CAD or HF" and "CAD and HF" appear where desired
    ordered_diseases = [
        "Coronary atherosclerosis",
        "Heart failure overall",
        "CAD or HF",  # New "OR" condition
        "CAD and HF",  # Existing "AND" condition
        "Ischemic heart disease (Overall)",
        "Hypercholesterolemia",
        "Diabetes mellitus (Overall)",
        "Hypertensive disorder",
        "Chronic kidney disease",
        "Asthma",
        "Deep venous thrombosis",
        "Vasculitis",
        "Malignant neoplastic disease",
        "Chest pain",
        "Electrocardiogram abnormal",
        "Cardiomegaly",
        "Cerebrovascular disease",
    ]

    for disease_name in ordered_diseases:
        disease_def = DISEASES[disease_name]  # Get the definition (list or dict)

        print(f"\n=== Running Analysis for {disease_name} ===")
        # Print the exact definition being used
        if isinstance(disease_def, list):
            print(f"Disease Concept IDs being used (OR logic): {disease_def}")
        elif isinstance(disease_def, dict):
            if "OR_concepts" in disease_def:
                print(
                    f"Disease Concept IDs being used (Combined OR logic): {disease_def['OR_concepts']}"
                )
            elif "AND_concepts" in disease_def:
                print(
                    f"Disease Concept IDs being used (Combined AND logic): {disease_def['AND_concepts']}"
                )

        print(
            "Fetching data from BigQuery with Race (African American), EHR, WGS filters, and last visit date..."
        )
        df_raw = pd.read_gbq(
            create_query(workspace_cdr, disease_def),
            dialect="standard",
            use_bqstorage_api=True,
        )

        if df_raw.empty:
            print(
                f"No data found for {disease_name} with initial filters. Skipping analysis."
            )
            continue

        print(
            f"\n--- Debugging: Distinct Individuals by Disease Condition (from raw query) ---"
        )
        # For debug query, we need to adapt for combined disease types.
        # It's difficult to show specific concept counts for combined "AND" conditions directly in debug here
        # without duplicating complex SQL, but we can confirm the total cases identified by the main query.
        if isinstance(disease_def, list):
            debug_concept_list_str = ", ".join(map(str, disease_def))
            debug_disease_event_query = f"""
            SELECT c.concept_name, COUNT(DISTINCT ce.person_id) AS distinct_people_with_event
            FROM `{workspace_cdr}.cb_search_all_events` ce
            JOIN `{workspace_cdr}.concept` c ON ce.concept_id = c.concept_id
            JOIN (
                SELECT p.person_id
                FROM `{workspace_cdr}.person` p
                JOIN `{workspace_cdr}.cb_search_person` cbsp ON p.person_id = cbsp.person_id
                WHERE p.race_concept_id = 8516 -- African American
                  AND cbsp.has_ehr_data = 1
                  AND cbsp.has_whole_genome_variant = 1
            ) AS filtered_pop ON ce.person_id = filtered_pop.person_id
            WHERE ce.concept_id IN ({debug_concept_list_str})
            GROUP BY c.concept_name
            ORDER BY distinct_people_with_event DESC
            """
        elif isinstance(disease_def, dict) and "OR_concepts" in disease_def:
            debug_concept_list_str = ", ".join(map(str, disease_def["OR_concepts"]))
            debug_disease_event_query = f"""
            SELECT c.concept_name, COUNT(DISTINCT ce.person_id) AS distinct_people_with_event
            FROM `{workspace_cdr}.cb_search_all_events` ce
            JOIN `{workspace_cdr}.concept` c ON ce.concept_id = c.concept_id
            JOIN (
                SELECT p.person_id
                FROM `{workspace_cdr}.person` p
                JOIN `{workspace_cdr}.cb_search_person` cbsp ON p.person_id = cbsp.person_id
                WHERE p.race_concept_id = 8516 -- African American
                  AND cbsp.has_ehr_data = 1
                  AND cbsp.has_whole_genome_variant = 1
            ) AS filtered_pop ON ce.person_id = filtered_pop.person_id
            WHERE ce.concept_id IN ({debug_concept_list_str})
            GROUP BY c.concept_name
            ORDER BY distinct_people_with_event DESC
            """
        else:  # For "AND" conditions or other complex definitions
            print(
                f"  (For combined disease '{disease_name}', detailed concept counts from source are complex to show here directly from BigQuery for debug.)"
            )
            print(
                f"  Total unique individuals identified with '{disease_name}' from the main query: {df_raw[df_raw['has_Disease'] == 1]['person_id'].nunique()}"
            )
            debug_disease_event_query = (
                None  # No direct concept breakdown in this debug section
            )

        if debug_disease_event_query:
            try:
                debug_df_events = pd.read_gbq(
                    debug_disease_event_query,
                    dialect="standard",
                    use_bqstorage_api=True,
                )
                if not debug_df_events.empty:
                    print(debug_df_events.to_string(index=False))
                else:
                    print(
                        f"No distinct individuals found for the specified '{disease_name}' concept IDs in the base cohort."
                    )
            except Exception as e:
                print(
                    f"Could not fetch individual disease event counts (this is a debug step): {e}"
                )
        print(
            "-----------------------------------------------------------------------------------"
        )

        total_people_from_bq = df_raw["person_id"].nunique()
        total_cases_from_bq = df_raw[df_raw["has_Disease"] == 1]["person_id"].nunique()
        print(f"\n--- Checkpoint 1: Data directly from BigQuery Query ---")
        print(
            f"    Cohort Filters: Race=African American, EHR=1, WGS=1, Any Age (from birth_datetime)"
        )
        print(f"    Total Unique Individuals in Cohort: {total_people_from_bq}")
        print(
            f"    Total Unique Disease Cases identified for '{disease_name}': {total_cases_from_bq}"
        )  # Clarify disease name
        print("------------------------------------------------------")

        print(
            "Applying Python-side age filtering (ages 18-100 based on last visit date calculation)..."
        )
        df_filtered_age = calculate_age(df_raw.copy())

        df_filtered_age.replace([np.inf, -np.inf], np.nan, inplace=True)

        if df_filtered_age.empty:
            print(
                f"No individuals found in age range 18-100 for {disease_name} after age filtering. Skipping further analysis."
            )
            continue

        total_population_final_cohort = df_filtered_age["person_id"].nunique()
        total_disease_cases_final_cohort = df_filtered_age[
            df_filtered_age["has_Disease"] == 1
        ]["person_id"].nunique()

        print(
            f"\n--- Checkpoint 2: Data after Python-side Age Filtering (18-100) using Last Visit Age ---"
        )
        print(
            f"    Cohort Filters: Race=African American, EHR=1, WGS=1, Age=18-100 (at last visit)"
        )
        print(
            f"    Total Unique Individuals in Final Cohort: {total_population_final_cohort}"
        )
        print(
            f"    Total Unique Disease Cases in Final Cohort for '{disease_name}': {total_disease_cases_final_cohort}"
        )  # Clarify disease name
        print("--------------------------------------------------------------")

        overall_prev_final_cohort = compute_overall_prevalence(df_filtered_age)
        print(
            f"\n-- Overall Disease Prevalence in Final Cohort ({disease_name}): {overall_prev_final_cohort:.2f}% --"
        )

        print("\n-- Odds Ratio and P-value Calculation at Each Age Cutoff --")
        or_p_df = calculate_or_p_values_at_age_cutoffs(
            df_filtered_age, age_cutoffs=age_cutoffs_for_analysis
        )

        if not or_p_df.empty:
            # Round numerical columns for display
            or_p_df_display = or_p_df.round(
                {
                    "OR": 2,
                    "Lower CI": 2,
                    "Upper CI": 2,
                    "P-value": 4,
                    "Target Cases": 0,
                    "Target Non-Cases": 0,
                    "Non-carrier Cases": 0,
                    "Non-carrier Non-Cases": 0,
                }
            ).copy()
            # Format P-value to be '<0.0001' if very small
            or_p_df_display["P-value"] = or_p_df_display["P-value"].apply(
                lambda x: "<0.0001"
                if isinstance(x, (float, np.float64)) and x < 0.0001
                else (f"{x:.4f}" if isinstance(x, (float, np.float64)) else "N/A")
            )
            # Format CI string
            or_p_df_display["95% CI"] = or_p_df_display.apply(
                lambda row: f"({row['Lower CI']:.2f}-{row['Upper CI']:.2f})"
                if pd.notna(row["Lower CI"]) and pd.notna(row["Upper CI"])
                else "N/A",
                axis=1,
            )
            # Select and reorder columns for display
            or_p_df_final_display = or_p_df_display[
                [
                    "Group",
                    "Variant",
                    "Age Cutoff",
                    "OR",
                    "95% CI",
                    "P-value",
                    "Target Cases",
                    "Target Non-Cases",
                    "Non-carrier Cases",
                    "Non-carrier Non-Cases",
                ]
            ]
            print(or_p_df_final_display.to_string(index=False))
        else:
            print("No Odds Ratios or P-values could be calculated at age cutoffs.")

        print("\n-- Cumulative Prevalence Calculation --")
        cum_prevalence_df = compute_cumulative_prevalence(
            df_filtered_age, age_cutoffs=age_cutoffs_for_analysis
        )

        if cum_prevalence_df.empty:
            print(
                "No cumulative prevalence data found for any group in the specified age ranges."
            )
        else:
            print("Full Cumulative Prevalence Data:")
            # Round prevalence for display
            cum_prevalence_df_display = cum_prevalence_df.round(
                {"prevalence": 2}
            ).copy()
            # Order the display based on the `group_type` to ensure desired sequencing
            # Define a custom order for group_type
            group_type_order = (
                ["Variant Overall"]
                + [
                    f"{v} Male"
                    for v in list(VARIANTS.keys()) + ["LDLR_pathogenic", "Non-carrier"]
                ]
                + [
                    f"{v} Female"
                    for v in list(VARIANTS.keys()) + ["LDLR_pathogenic", "Non-carrier"]
                ]
                + ["Cohort Overall", "Male Cohort", "Female Cohort"]
            )

            cum_prevalence_df_display["group_type"] = pd.Categorical(
                cum_prevalence_df_display["group_type"],
                categories=group_type_order,
                ordered=True,
            )

            # Sort by group_type, then variant_label, then age_cutoff
            cum_prevalence_df_display = cum_prevalence_df_display.sort_values(
                by=["group_type", "variant_label", "age_cutoff"]
            ).reset_index(drop=True)

            print(
                cum_prevalence_df_display[
                    ["variant_label", "age_cutoff", "prevalence", "total", "cases"]
                ].to_string(index=False)
            )

            print(
                f"\n-- Generating Combined Focused Cumulative Prevalence Plots (Overall, Male, Female) for {disease_name} --"
            )
            plot_combined_focused_prevalence(
                cum_prevalence_df, disease_name, overall_prev_final_cohort
            )

        print(f"\n=== Analysis for {disease_name} Completed. ===")


def main():
    """
    Main function to set the workspace CDR and initiate the analysis.
    """
    os.environ.setdefault("WORKSPACE_CDR", "fc-aou-cdr-prod-ct.C2024Q3R5")
    workspace_cdr = os.environ["WORKSPACE_CDR"]
    print(f"Using WORKSPACE_CDR: {workspace_cdr}")
    run_analysis_per_disease(workspace_cdr)


if __name__ == "__main__":
    main()
