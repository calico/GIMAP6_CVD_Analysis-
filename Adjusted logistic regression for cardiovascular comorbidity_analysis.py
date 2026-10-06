import pandas as pd
import numpy as np
from datetime import datetime
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
import ast
import warnings
from scipy import stats
import re

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For interactive table display, FDR correction, etc.
from IPython.display import display, HTML
from statsmodels.stats.multitest import multipletests

# Suppress specific warnings for cleaner output
warnings.filterwarnings(
    "ignore",
    message="read_gbq is deprecated and will be removed in a future version. Please use pandas_gbq.read_gbq instead:",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Unable to represent RANGE schema as struct using pandas ArrowDtype. Using `object` instead.",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

# =========================
# GLOBAL PLOTTING SETTINGS
# =========================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
    }
)

# =========================
# GCS path for PCA scores
# =========================
PCA_FILE_PATH = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"

# =========================
# Define Gene Variants and Disease Concepts
# =========================
VARIANTS = {
    "GIMAP6_V65I": ["7-150628405-C-T"],
}
DISEASES = {
    "CAD": [
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
    ],
    "HF": [316139],
}

DISEASE_DEFINITIONS_FOR_QUERY = {
    # This definition is used to get the concept IDs for the query logic
    "CAD_AND_HF": {"AND_concepts": [DISEASES["CAD"], DISEASES["HF"]]}
}

ADDITIONAL_COVARIATES_CONCEPTS = {
    "diabetes_status": [201820, 440549, 439064, 4056253, 40484648],
    "hypertension_status": [316866, 320128, 4329847, 442799, 40489956],
    "bmi": [3038553],
    "hyperlipidemia_status": [432867],
}


# =========================
# Query Creation Function
# =========================
def create_query(
    workspace_cdr,
    disease_definition_key,
    disease_concept_map,
    variants_map,
    covariate_concepts,
    ancestry_filter_concept_id=8516,
    diagnosis_timing="first",
):
    """
    Constructs a comprehensive BigQuery SQL query.
    NEW: `diagnosis_timing` parameter controls the case definition.
         'first': Earliest diagnosis of ANY of the combined diseases.
         'latter': The LATER of the first CAD and first HF diagnosis.
    """
    disease_definition = disease_concept_map[disease_definition_key]
    disease_cases_sql = ""
    diagnosis_year_sql = "min_diagnosis_year"  # This column name is used by both logics

    # --- Disease Case SQL Construction ---
    # This logic is now controlled by the `diagnosis_timing` parameter
    if "AND_concepts" not in disease_definition:
        raise ValueError("Script is configured for AND logic (e.g., CAD_AND_HF).")

    # LOGIC 1: Use the EARLIEST diagnosis date among anyone who has BOTH conditions
    if diagnosis_timing == "first":
        print(
            "  Querying for cases based on the FIRST diagnosis date of either disease."
        )
        all_concepts_for_and = [
            item for sublist in disease_definition["AND_concepts"] for item in sublist
        ]
        concept_id_list_overall = ", ".join(map(str, all_concepts_for_and))
        base_and_query_parts = []
        for concept_list in disease_definition["AND_concepts"]:
            concept_list_str = ", ".join(map(str, concept_list))
            base_and_query_parts.append(
                f"SELECT DISTINCT person_id FROM `{workspace_cdr}.cb_search_all_events` WHERE concept_id IN ({concept_list_str})"
            )

        base_query_joins = f"SELECT t0.person_id FROM ({base_and_query_parts[0]}) AS t0"
        for i in range(1, len(base_and_query_parts)):
            base_query_joins += f"\nINNER JOIN ({base_and_query_parts[i]}) AS t{i} ON t0.person_id = t{i}.person_id"

        disease_cases_sql = f"""
            SELECT t_cases.person_id, MIN(EXTRACT(YEAR FROM t_events.entry_date)) AS {diagnosis_year_sql}
            FROM ({base_query_joins}) AS t_cases
            JOIN `{workspace_cdr}.cb_search_all_events` AS t_events ON t_cases.person_id = t_events.person_id
            WHERE t_events.concept_id IN ({concept_id_list_overall})
            GROUP BY t_cases.person_id
        """
    # LOGIC 2: Use the LATTER diagnosis date between the two separate conditions
    elif diagnosis_timing == "latter":
        print(
            "  Querying for cases based on the LATTER diagnosis date of the two diseases."
        )
        cad_concepts_str = ", ".join(map(str, disease_definition["AND_concepts"][0]))
        hf_concepts_str = ", ".join(map(str, disease_definition["AND_concepts"][1]))
        disease_cases_sql = f"""
            WITH cad_diagnosis AS (
                SELECT person_id, MIN(entry_date) AS first_cad_date
                FROM `{workspace_cdr}.cb_search_all_events`
                WHERE concept_id IN ({cad_concepts_str})
                GROUP BY person_id
            ),
            hf_diagnosis AS (
                SELECT person_id, MIN(entry_date) AS first_hf_date
                FROM `{workspace_cdr}.cb_search_all_events`
                WHERE concept_id IN ({hf_concepts_str})
                GROUP BY person_id
            )
            SELECT
                cad.person_id,
                EXTRACT(YEAR FROM GREATEST(cad.first_cad_date, hf.first_hf_date)) AS {diagnosis_year_sql}
            FROM cad_diagnosis AS cad
            INNER JOIN hf_diagnosis AS hf ON cad.person_id = hf.person_id
        """
    else:
        raise ValueError(
            f"Invalid diagnosis_timing: {diagnosis_timing}. Choose 'first' or 'latter'."
        )

    # --- Variant SQL Construction (unchanged) ---
    variant_sqls = []
    for label, vids in variants_map.items():
        vid_list = ", ".join([f"'{vid}'" for vid in vids])
        variant_sqls.append(
            f"""
            LEFT JOIN (
                SELECT DISTINCT person_id_unnested AS person_id
                FROM `{workspace_cdr}.cb_variant_to_person`, UNNEST(person_ids) AS person_id_unnested
                WHERE vid IN ({vid_list})
            ) AS {label.lower()}_carriers ON p.person_id = {label.lower()}_carriers.person_id
        """
        )

    # --- Additional Covariates SQL Construction (unchanged) ---
    covariate_sqls = []
    select_covariate_columns = []
    for cov_name, concept_ids in covariate_concepts.items():
        concept_id_list = ", ".join(map(str, concept_ids))
        if cov_name == "bmi":
            covariate_sqls.append(
                f"""
                LEFT JOIN (
                    SELECT person_id, value_as_number AS {cov_name}_value
                    FROM `{workspace_cdr}.measurement`
                    WHERE measurement_concept_id IN ({concept_id_list})
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY measurement_datetime DESC) = 1
                ) AS {cov_name}_data ON p.person_id = {cov_name}_data.person_id
            """
            )
            select_covariate_columns.append(
                f", {cov_name}_data.{cov_name}_value AS {cov_name}"
            )
        else:
            covariate_sqls.append(
                f"""
                LEFT JOIN (
                    SELECT DISTINCT person_id FROM `{workspace_cdr}.cb_search_all_events`
                    WHERE concept_id IN ({concept_id_list})
                ) AS {cov_name}_table ON p.person_id = {cov_name}_table.person_id
            """
            )
            select_covariate_columns.append(
                f", CASE WHEN {cov_name}_table.person_id IS NOT NULL THEN 1 ELSE 0 END AS {cov_name}"
            )

    # --- Main Query Assembly (unchanged) ---
    last_event_subquery = f"SELECT person_id, MAX(entry_date) AS last_event_datetime FROM `{workspace_cdr}.cb_search_all_events` GROUP BY person_id"

    main_query = f"""
    SELECT
        p.person_id, p.year_of_birth, p.month_of_birth, p.day_of_birth,
        gender.concept_name AS sex,
        events.last_event_datetime,
        CASE WHEN cases.person_id IS NOT NULL THEN 1 ELSE 0 END AS has_disease,
        cases.{diagnosis_year_sql}
        {", ".join([f", CASE WHEN {label.lower()}_carriers.person_id IS NOT NULL THEN 1 ELSE 0 END AS is_{label.lower()}_carrier" for label in variants_map.keys()])}
        {" ".join(select_covariate_columns)}
    FROM `{workspace_cdr}.person` p
    JOIN `{workspace_cdr}.cb_search_person` cbsp ON p.person_id = cbsp.person_id
    JOIN `{workspace_cdr}.concept` gender ON p.gender_concept_id = gender.concept_id
    LEFT JOIN ({disease_cases_sql}) cases ON p.person_id = cases.person_id
    LEFT JOIN ({last_event_subquery}) events ON p.person_id = events.person_id
    {" ".join(variant_sqls)}
    {" ".join(covariate_sqls)}
    WHERE
        cbsp.has_ehr_data = 1
        AND cbsp.has_whole_genome_variant = 1
        AND events.last_event_datetime IS NOT NULL
        AND p.race_concept_id = {ancestry_filter_concept_id}
    """
    return main_query


# =========================
# Helper Functions
# =========================
def run_query(query):
    import pandas_gbq

    print("  Running BigQuery query...")
    return pandas_gbq.read_gbq(
        query,
        project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        dialect="standard",
        use_bqstorage_api=True,
    )


def get_pca_data(pca_file_path, num_pcs_to_use=20):
    import gcsfs

    print("  Reading PCA data...")
    pca_df = pd.read_csv(
        pca_file_path, sep="\t", storage_options={"requester_pays": True}
    )
    pca_df = pca_df.rename(columns={"research_id": "person_id"})
    pca_features_lists = pca_df["pca_features"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
    )
    pca_features_expanded = pd.DataFrame(
        pca_features_lists.tolist(), index=pca_df.index
    )
    pca_cols = [f"PC{i}" for i in range(1, num_pcs_to_use + 1)]
    available_pcs = [
        col for i, col in enumerate(pca_cols) if i < pca_features_expanded.shape[1]
    ]
    pca_features_expanded.columns = [
        f"PC{i+1}" for i in range(pca_features_expanded.shape[1])
    ]
    pca_df_final = pd.concat(
        [pca_df["person_id"], pca_features_expanded[available_pcs]], axis=1
    )
    for col in available_pcs:
        pca_df_final[col] = pd.to_numeric(pca_df_final[col], errors="coerce")
    return pca_df_final.dropna(subset=available_pcs, how="all")


def get_smoking_status_data(workspace_cdr):
    query = f"""
    SELECT t1.person_id, t2.concept_name AS smoking_status_name
    FROM `{workspace_cdr}.observation` t1
    JOIN `{workspace_cdr}.concept` t2 ON t1.observation_concept_id = t2.concept_id
    WHERE t2.concept_name LIKE '%Smoker%' OR t2.concept_name LIKE '%smoking status%' OR t2.concept_name LIKE '%tobacco user%'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY t1.person_id ORDER BY t1.observation_datetime DESC) = 1
    """
    print("  Fetching smoking status data...")
    return run_query(query)


def categorize_smoking(status_name):
    s = str(status_name).lower()
    if "current" in s or "tobacco user" in s:
        return "Current Smoker"
    if "non-user" in s or "never" in s:
        return "Never Smoker"
    if "former" in s or "quit" in s:
        return "Former Smoker"
    return "Unknown/Other"


def categorize_bmi(bmi_value):
    if pd.isna(bmi_value):
        return "Unknown"
    if bmi_value < 18.5:
        return "Underweight"
    if bmi_value < 25:
        return "Normal Weight"
    if bmi_value < 30:
        return "Overweight"
    return "Obese"


def plot_age_distributions(data, disease_label, variant_label, context=""):
    """
    *** NEW: Generates a 2x2 grid of plots, including a combined male+female diagnosis plot. ***
    """
    print(
        f"\n--- Plotting Age Distributions for {disease_label} ({variant_label}) - Context: {context} ---"
    )
    sns.set_style("whitegrid")
    # New layout for 2x2 plots
    plt.figure(figsize=(20, 10))
    variant_col = f"is_{variant_label.lower()}_carrier"

    # Plot 1 (Top-Left): Overall Age Distribution
    plt.subplot(2, 2, 1)
    if "age" in data.columns and not data["age"].isnull().all():
        sns.histplot(data["age"], kde=True, bins=20, color="skyblue", stat="density")
        median_age_overall = data["age"].median()
        plt.axvline(
            median_age_overall,
            color="blue",
            linestyle="--",
            label=f"Median Age: {median_age_overall:.1f} yrs",
        )
        plt.title(f"Overall Age Distribution (N={len(data)})")
        plt.xlabel("Age at Last Observation (Years)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Age data not available", ha="center", va="center")

    # Plot 2 (Top-Right): *** NEW: Age at Diagnosis for ALL CASES ***
    plt.subplot(2, 2, 2)
    if "age_at_diagnosis" in data.columns:
        all_cases = data[
            (data["has_disease"] == 1) & (data["age_at_diagnosis"].notna())
        ].copy()
        if not all_cases.empty:
            sns.histplot(
                data=all_cases,
                x="age_at_diagnosis",
                hue=variant_col,
                kde=True,
                bins=20,
                palette=["grey", "red"],
                common_norm=False,
                stat="density",
            )
            median_control = all_cases.loc[
                all_cases[variant_col] == 0, "age_at_diagnosis"
            ].median()
            median_carrier = all_cases.loc[
                all_cases[variant_col] == 1, "age_at_diagnosis"
            ].median()
            plt.axvline(
                median_control,
                color="grey",
                linestyle=":",
                label=f"Non-Carrier Median: {median_control:.1f} yrs"
                if pd.notna(median_control)
                else "Non-Carrier Median: N/A",
            )
            plt.axvline(
                median_carrier,
                color="red",
                linestyle=":",
                label=f"Carrier Median: {median_carrier:.1f} yrs"
                if pd.notna(median_carrier)
                else "Carrier Median: N/A",
            )
            plt.title(f"Age at Diagnosis - All Cases (N={len(all_cases)})")
            plt.xlabel("Age at Diagnosis (Years)")
            plt.legend()
        else:
            plt.title("Age at Diagnosis - All Cases")
            plt.text(0.5, 0.5, "No cases with diagnosis age", ha="center", va="center")

    # Plot 3 (Bottom-Left): Age at Diagnosis for MALES
    plt.subplot(2, 2, 3)
    if "age_at_diagnosis" in data.columns:
        male_cases = data[
            (data["has_disease"] == 1)
            & (data["sex"] == "Male")
            & (data["age_at_diagnosis"].notna())
        ].copy()
        if not male_cases.empty:
            sns.histplot(
                data=male_cases,
                x="age_at_diagnosis",
                hue=variant_col,
                kde=True,
                bins=20,
                palette=["grey", "red"],
                common_norm=False,
                stat="density",
            )
            median_control = male_cases.loc[
                male_cases[variant_col] == 0, "age_at_diagnosis"
            ].median()
            median_carrier = male_cases.loc[
                male_cases[variant_col] == 1, "age_at_diagnosis"
            ].median()
            plt.axvline(
                median_control,
                color="grey",
                linestyle=":",
                label=f"Non-Carrier Median: {median_control:.1f} yrs"
                if pd.notna(median_control)
                else "Non-Carrier Median: N/A",
            )
            plt.axvline(
                median_carrier,
                color="red",
                linestyle=":",
                label=f"Carrier Median: {median_carrier:.1f} yrs"
                if pd.notna(median_carrier)
                else "Carrier Median: N/A",
            )
            plt.title(f"Age at Diagnosis - Males (N={len(male_cases)})")
            plt.xlabel("Age at Diagnosis (Years)")
            plt.legend()
        else:
            plt.title("Age at Diagnosis - Males")
            plt.text(
                0.5, 0.5, "No male cases with diagnosis age", ha="center", va="center"
            )

    # Plot 4 (Bottom-Right): Age at Diagnosis for FEMALES
    plt.subplot(2, 2, 4)
    if "age_at_diagnosis" in data.columns:
        female_cases = data[
            (data["has_disease"] == 1)
            & (data["sex"] == "Female")
            & (data["age_at_diagnosis"].notna())
        ].copy()
        if not female_cases.empty:
            sns.histplot(
                data=female_cases,
                x="age_at_diagnosis",
                hue=variant_col,
                kde=True,
                bins=20,
                palette=["grey", "red"],
                common_norm=False,
                stat="density",
            )
            median_control = female_cases.loc[
                female_cases[variant_col] == 0, "age_at_diagnosis"
            ].median()
            median_carrier = female_cases.loc[
                female_cases[variant_col] == 1, "age_at_diagnosis"
            ].median()
            plt.axvline(
                median_control,
                color="grey",
                linestyle=":",
                label=f"Non-Carrier Median: {median_control:.1f} yrs"
                if pd.notna(median_control)
                else "Non-Carrier Median: N/A",
            )
            plt.axvline(
                median_carrier,
                color="red",
                linestyle=":",
                label=f"Carrier Median: {median_carrier:.1f} yrs"
                if pd.notna(median_carrier)
                else "Carrier Median: N/A",
            )
            plt.title(f"Age at Diagnosis - Females (N={len(female_cases)})")
            plt.xlabel("Age at Diagnosis (Years)")
            plt.legend()
        else:
            plt.title("Age at Diagnosis - Females")
            plt.text(
                0.5, 0.5, "No female cases with diagnosis age", ha="center", va="center"
            )

    plt.suptitle(f"Age Distributions for {context}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def build_formula(data, variant_col, pca_cols):
    formula = f"has_disease ~ {variant_col} + age + C(sex) + C(smoking_status) + C(bmi_category) + diabetes_status + hypertension_status + hyperlipidemia_status"
    pc_part = " + ".join([pc for pc in pca_cols if pc in data.columns])
    if pc_part:
        formula += " + " + pc_part
    return formula


def run_regression_and_calc_stats(data, disease, variant, pca_cols, context):
    """
    Performs logistic regression and calculates detailed summary statistics for reporting. (Unchanged)
    """
    variant_col = f"is_{variant.lower()}_carrier"
    print(f"\n--- Running Analysis & Stats: {disease} ({variant}) - {context} ---")
    formula = build_formula(data, variant_col, pca_cols)

    required_cols = [
        "has_disease",
        variant_col,
        "age",
        "sex",
        "smoking_status",
        "bmi_category",
        "diabetes_status",
        "hypertension_status",
        "hyperlipidemia_status",
    ] + pca_cols
    df_clean = data.dropna(
        subset=[col for col in required_cols if col in data.columns]
    ).copy()

    # Ensure correct dtypes before model fitting
    for col in [
        variant_col,
        "diabetes_status",
        "hypertension_status",
        "hyperlipidemia_status",
    ]:
        if col in df_clean:
            df_clean[col] = df_clean[col].astype(int)
    if "age" in df_clean:
        df_clean["age"] = df_clean["age"].astype(np.float64)
    for col in ["sex", "smoking_status", "bmi_category"]:
        if col in df_clean:
            df_clean[col] = df_clean[col].astype("category")

    # Initialize results dictionary
    results = {
        "Context": context,
        "OR": np.nan,
        "2.5% CI": np.nan,
        "97.5% CI": np.nan,
        "P-value": np.nan,
        "N_Total": len(df_clean),
        "N_Cases": df_clean["has_disease"].sum(),
        "N_Controls": len(df_clean) - df_clean["has_disease"].sum(),
        "N_Carriers_Cases": df_clean[df_clean["has_disease"] == 1][variant_col].sum(),
        "N_NonCarriers_Cases": df_clean[df_clean["has_disease"] == 1][variant_col]
        .value_counts()
        .get(0, 0),
        "N_Carriers_Controls": df_clean[df_clean["has_disease"] == 0][
            variant_col
        ].sum(),
        "N_NonCarriers_Controls": df_clean[df_clean["has_disease"] == 0][variant_col]
        .value_counts()
        .get(0, 0),
        "Carrier_Freq_Cases": np.nan,
        "Carrier_Freq_Controls": np.nan,
        "Mean_Age_Group": df_clean["age"].mean(),
        "SEM_Age_Group": df_clean["age"].sem(),
        "N_Age_Dx_Carriers": 0,
        "Mean_Age_Dx_Carriers": np.nan,
        "SEM_Age_Dx_Carriers": np.nan,
        "N_Age_Dx_NonCarriers": 0,
        "Mean_Age_Dx_NonCarriers": np.nan,
        "SEM_Age_Dx_NonCarriers": np.nan,
        "Notes": "",
    }

    # Calculate derived stats
    if results["N_Cases"] > 0:
        results["Carrier_Freq_Cases"] = results["N_Carriers_Cases"] / results["N_Cases"]
    if results["N_Controls"] > 0:
        results["Carrier_Freq_Controls"] = (
            results["N_Carriers_Controls"] / results["N_Controls"]
        )

    if "age_at_diagnosis" in df_clean.columns:
        cases_with_age_dx = df_clean[
            (df_clean["has_disease"] == 1) & (df_clean["age_at_diagnosis"].notna())
        ]
        carriers_cases = cases_with_age_dx[cases_with_age_dx[variant_col] == 1]
        noncarriers_cases = cases_with_age_dx[cases_with_age_dx[variant_col] == 0]

        if not carriers_cases.empty:
            results["N_Age_Dx_Carriers"] = len(carriers_cases)
            results["Mean_Age_Dx_Carriers"] = carriers_cases["age_at_diagnosis"].mean()
            results["SEM_Age_Dx_Carriers"] = carriers_cases["age_at_diagnosis"].sem()
        if not noncarriers_cases.empty:
            results["N_Age_Dx_NonCarriers"] = len(noncarriers_cases)
            results["Mean_Age_Dx_NonCarriers"] = noncarriers_cases[
                "age_at_diagnosis"
            ].mean()
            results["SEM_Age_Dx_NonCarriers"] = noncarriers_cases[
                "age_at_diagnosis"
            ].sem()

    if len(df_clean) < 100 or df_clean[variant_col].sum() < 20:
        results["Notes"] = "Skipped: Insufficient data for regression."
        print(f"  Skipping regression for '{context}': insufficient data.")
        return results

    try:
        model = smf.logit(formula, data=df_clean).fit(disp=False)
        print("\n" + "*" * 20 + f" FULL MODEL RESULTS FOR: {context} " + "*" * 20)
        print(model.summary())
        print("*" * (42 + len(context)) + "\n")

        odds_ratios = np.exp(model.params)
        conf_int = np.exp(model.conf_int())
        p_values = model.pvalues
        if variant_col in odds_ratios.index:
            results.update(
                {
                    "OR": odds_ratios[variant_col],
                    "2.5% CI": conf_int.loc[variant_col, 0],
                    "97.5% CI": conf_int.loc[variant_col, 1],
                    "P-value": p_values[variant_col],
                }
            )
    except Exception as e:
        results["Notes"] = f"Error: {str(e)}"
        print(f"  ERROR running regression for '{context}': {e}")

    return results


def summarize_and_report(all_results):
    """Summarizes and displays final results with FDR correction and detailed stats. (Unchanged)"""
    print("\n\n" + "#" * 100)
    print("                              FINAL COMPREHENSIVE ANALYSIS SUMMARY")
    print("#" * 100)

    if not all_results:
        print("No results to summarize.")
        return

    results_df = pd.DataFrame(all_results)

    # FDR Correction
    p_values_for_correction = results_df.dropna(subset=["P-value"])["P-value"]
    num_tests = len(p_values_for_correction)
    if not p_values_for_correction.empty:
        reject_fdr, p_values_fdr_corrected, _, _ = multipletests(
            p_values_for_correction, alpha=0.05, method="fdr_bh"
        )
        results_df.loc[
            p_values_for_correction.index, "FDR_Corrected_P_value"
        ] = p_values_fdr_corrected
        results_df.loc[p_values_for_correction.index, "FDR_Significant"] = [
            "Significant" if r else "Not Significant" for r in reject_fdr
        ]

    results_df["Significance"] = results_df["P-value"].apply(
        lambda p: "Significant" if pd.notna(p) and p < 0.05 else "Not Significant"
    )
    results_df["Num_Tests_in_FDR"] = num_tests if num_tests > 0 else 0

    final_columns = [
        "Context",
        "N_Total",
        "Mean_Age_Group",
        "SEM_Age_Group",
        "N_Cases",
        "N_Controls",
        "N_Carriers_Cases",
        "N_NonCarriers_Cases",
        "N_Carriers_Controls",
        "N_NonCarriers_Controls",
        "Carrier_Freq_Cases",
        "Carrier_Freq_Controls",
        "N_Age_Dx_Carriers",
        "Mean_Age_Dx_Carriers",
        "SEM_Age_Dx_Carriers",
        "N_Age_Dx_NonCarriers",
        "Mean_Age_Dx_NonCarriers",
        "SEM_Age_Dx_NonCarriers",
        "OR",
        "2.5% CI",
        "97.5% CI",
        "P-value",
        "Significance",
        "FDR_Corrected_P_value",
        "FDR_Significant",
        "Num_Tests_in_FDR",
    ]
    summary_df = results_df[[col for col in final_columns if col in results_df.columns]]

    pd.set_option("display.max_columns", None)
    display(HTML(summary_df.to_html(index=False, float_format="%.4f")))


def main():
    os.environ.setdefault("WORKSPACE_CDR", "fc-aou-cdr-prod-ct.C2024Q3R5")
    workspace_cdr = os.environ["WORKSPACE_CDR"]
    print(f"Using WORKSPACE_CDR: {workspace_cdr}")

    # ==================================
    # STEP 1: PRE-FETCH GLOBAL DATA (PCA and Smoking)
    # ==================================
    # These are fetched once and reused for both analysis runs
    print(
        "\n" + "=" * 80 + "\n  PRE-FETCHING GLOBAL DATASETS (PCA, SMOKING)\n" + "=" * 80
    )
    pca_data = get_pca_data(PCA_FILE_PATH)
    pca_cols_for_regression = (
        [col for col in pca_data.columns if col.startswith("PC")]
        if not pca_data.empty
        else []
    )

    smoking_data_raw = get_smoking_status_data(workspace_cdr)
    smoking_data = pd.DataFrame()
    if not smoking_data_raw.empty:
        smoking_data = smoking_data_raw[
            ["person_id", "smoking_status_name"]
        ].drop_duplicates()
        smoking_data["smoking_status"] = smoking_data["smoking_status_name"].apply(
            categorize_smoking
        )
        smoking_data = smoking_data[["person_id", "smoking_status"]]

    # ==================================
    # *** NEW: DUAL ANALYSIS LOOP ***
    # ==================================
    all_results = []  # This will collect results from ALL analyses
    analysis_types = {
        "First_Diagnosis_of_Either": "first",
        "Latter_Diagnosis_of_Both": "latter",
    }

    for analysis_name, timing_param in analysis_types.items():
        print(f"\n\n{'='*100}\n{'='*100}")
        print(f"     STARTING ANALYSIS RUN: {analysis_name}")
        print(f"{'='*100}\n{'='*100}\n")

        # STEP 2: DATA FETCH FOR THE CURRENT ANALYSIS TYPE
        print(
            "\n"
            + "=" * 80
            + f"\n  STEP 2.1: FETCHING COHORT FOR: {analysis_name}\n"
            + "=" * 80
        )
        query = create_query(
            workspace_cdr,
            "CAD_AND_HF",
            DISEASE_DEFINITIONS_FOR_QUERY,
            VARIANTS,
            ADDITIONAL_COVARIATES_CONCEPTS,
            diagnosis_timing=timing_param,
        )  # Pass the timing parameter
        data_raw = run_query(query)
        print(
            f"CHECKPOINT [{analysis_name}]: Raw data fetched. Shape: {data_raw.shape}"
        )

        if data_raw.empty or data_raw["has_disease"].sum() == 0:
            print(
                f"WARNING [{analysis_name}]: No case data returned for this definition. Skipping this analysis run."
            )
            continue

        # STEP 3: DATA PREPROCESSING
        print(
            "\n"
            + "=" * 80
            + f"\n  STEP 2.2: PREPROCESSING DATA FOR: {analysis_name}\n"
            + "=" * 80
        )
        data = data_raw.copy()
        data["month_of_birth"] = data["month_of_birth"].fillna(1).astype(int)
        data["day_of_birth"] = data["day_of_birth"].fillna(1).astype(int)
        birth_date_cols = {
            "year_of_birth": "year",
            "month_of_birth": "month",
            "day_of_birth": "day",
        }
        data["birth_datetime"] = pd.to_datetime(
            data[["year_of_birth", "month_of_birth", "day_of_birth"]].rename(
                columns=birth_date_cols
            )
        )
        data["last_visit_datetime"] = pd.to_datetime(data["last_event_datetime"])
        data = data.dropna(subset=["birth_datetime", "last_visit_datetime"]).copy()
        data["age"] = (
            data["last_visit_datetime"] - data["birth_datetime"]
        ).dt.days / 365.25
        data = data[data["age"] >= 18].copy()
        print(
            f"CHECKPOINT [{analysis_name}]: Calculated 'age'. Shape after filtering for adults >= 18: {data.shape}"
        )

        data["age_at_diagnosis"] = data["min_diagnosis_year"] - data["year_of_birth"]
        data.loc[data["age_at_diagnosis"] < 0, "age_at_diagnosis"] = np.nan
        print(f"CHECKPOINT [{analysis_name}]: Calculated 'age_at_diagnosis'.")

        # Merge pre-fetched PCA and Smoking data
        if not pca_data.empty:
            data = pd.merge(data, pca_data, on="person_id", how="left")
        if not smoking_data.empty:
            data = pd.merge(data, smoking_data, on="person_id", how="left")

        if "smoking_status" not in data.columns:
            data["smoking_status"] = "Unknown/Other"
        data["smoking_status"] = (
            data["smoking_status"].fillna("Unknown/Other").astype("category")
        )

        data["bmi"] = pd.to_numeric(data["bmi"], errors="coerce")
        data = data.dropna(subset=["bmi"]).copy()
        data["bmi_category"] = data["bmi"].apply(categorize_bmi).astype("category")

        data = data.drop(
            columns=[
                "year_of_birth",
                "month_of_birth",
                "day_of_birth",
                "last_event_datetime",
                "birth_datetime",
                "smoking_status_name",
                "min_diagnosis_year",
            ],
            errors="ignore",
        )

        print(
            f"CHECKPOINT [{analysis_name}]: Final data cleaning complete. Ready for analysis."
        )
        disease_counts = data["has_disease"].value_counts()
        print(f"  - Final Participants in Cohort: {len(data)}")
        print(f"  - Cases (Has Disease = 1):    {disease_counts.get(1, 0)}")
        print(f"  - Controls (Has Disease = 0): {disease_counts.get(0, 0)}")

        # STEP 4: RUN STRATIFIED ANALYSES
        analysis_scenarios = {
            "Full_Cohort": {"max_age": 200},
            "Under_70_Cohort": {"max_age": 69},
        }
        variant_label = list(VARIANTS.keys())[0]

        for scenario_name, params in analysis_scenarios.items():
            print(
                f"\n\n{'#'*80}\n##### STARTING SCENARIO: {analysis_name}, {scenario_name} (Age <= {params['max_age']}) #####\n{'#'*80}"
            )

            scenario_data = data[data["age"] <= params["max_age"]].copy()

            if scenario_data.empty:
                print(f"  No data for scenario '{scenario_name}'. Skipping.")
                continue

            # Create a detailed context string for plots and results
            full_context_str = f"{analysis_name}, {scenario_name}"

            print(
                f"CHECKPOINT: Created data subset for scenario '{full_context_str}'. Shape: {scenario_data.shape}"
            )
            plot_age_distributions(
                scenario_data, "CAD_AND_HF", variant_label, context=full_context_str
            )

            # Run regressions and append results to the master list
            all_results.append(
                run_regression_and_calc_stats(
                    scenario_data,
                    "CAD_AND_HF",
                    variant_label,
                    pca_cols_for_regression,
                    f"{full_context_str}, All_Sexes",
                )
            )

            male_data = scenario_data[scenario_data["sex"] == "Male"].copy()
            all_results.append(
                run_regression_and_calc_stats(
                    male_data,
                    "CAD_AND_HF",
                    variant_label,
                    pca_cols_for_regression,
                    f"{full_context_str}, Male_Only",
                )
            )

            female_data = scenario_data[scenario_data["sex"] == "Female"].copy()
            all_results.append(
                run_regression_and_calc_stats(
                    female_data,
                    "CAD_AND_HF",
                    variant_label,
                    pca_cols_for_regression,
                    f"{full_context_str}, Female_Only",
                )
            )

    # ==================================
    # STEP 5: SUMMARIZE AND REPORT ALL RESULTS
    # ==================================
    summarize_and_report(all_results)


if __name__ == "__main__":
    main()
