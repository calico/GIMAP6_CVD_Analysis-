# step1 Random choose the missense Vriants
import pandas_gbq
import pandas as pd
import os
from google.cloud import bigquery

# Get dataset name
dataset = os.environ["WORKSPACE_CDR"]
client = bigquery.Client()
# Let's look at the structure of the variant tables
client = bigquery.Client()

# Check the schema of cb_variant_attribute (likely contains consequence info)
table_ref = client.get_table(f"{dataset}.cb_variant_attribute")
print("cb_variant_attribute schema:")
for field in table_ref.schema:
    print(f"  {field.name}: {field.field_type}")
	
# Check cb_variant_to_person schema (likely contains frequency info)
table_ref = client.get_table(f"{dataset}.cb_variant_to_person")
print("\ncb_variant_to_person schema:")
for field in table_ref.schema:
    print(f"  {field.name}: {field.field_type}")
# Let's see what consequence types are available
sql_query = f"""
SELECT consequence, COUNT(*) as count
FROM `{dataset}.cb_variant_attribute` 
GROUP BY consequence
ORDER BY count DESC
LIMIT 20
"""

df_consequences = pandas_gbq.read_gbq(sql_query, dialect='standard')
print("Available consequence types:")
print(df_consequences)
# Check allele frequency distribution
sql_query = f"""
SELECT 
  MIN(allele_frequency) as min_af,
  MAX(allele_frequency) as max_af,
  AVG(allele_frequency) as avg_af,
  COUNT(*) as total_variants
FROM `{dataset}.cb_variant_attribute`
WHERE allele_frequency IS NOT NULL
"""

df_freq = pandas_gbq.read_gbq(sql_query, dialect='standard')
print("Allele frequency distribution:")
print(df_freq)
sql_query = f"""
SELECT 
  vid,
  contig,
  position,
  ref_allele,
  alt_allele,
  consequence,
  protein_change,
  allele_frequency,
  allele_count,
  allele_number,
  participant_count,
  genes
FROM `{dataset}.cb_variant_attribute` 
WHERE 
  EXISTS (
    SELECT 1 FROM UNNEST(consequence) AS c WHERE c LIKE '%missense%'
  )
  AND allele_frequency BETWEEN 0.01 AND 0.02
  AND allele_frequency IS NOT NULL
ORDER BY RAND()
LIMIT 500
"""
df_missense = pandas_gbq.read_gbq(sql_query, dialect='standard')
print(f"Found {len(df_missense)} missense variants with 1-2% frequency")
print(df_missense.head(200))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_missense)

# step2  Carrier frequency across age strata
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress specific warnings to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define Gene Variants
# This dictionary holds the Variant IDs (Vids) for all the genetic variants you want to analyze.
VARIANTS = {
    "GIMAP6_V65I": ['7-150628405-C-T'],
    "TTR_V122I": ['18-31598655-G-A'],
    "GIMAP6_Q237R": ['7-150627888-T-C'],
    
    # APOE SNPs - we will handle these with special logic
    "APOE_rs429358": ['19-44908684-T-C'], # Presence of 'C' allele
    "APOE_rs7412": ['19-44908822-C-T'],   # Presence of 'T' allele

    # Control Variants
    "CONTROL_1": ['6-31027001-A-G'],
    "CONTROL_2": ['5-179802039-A-C'],
    "CONTROL_3": ['5-111100646-A-C'],
    "CONTROL_4": ['10-93401511-C-T'],
    "CONTROL_5": ['19-8978996-C-T'],
    "CONTROL_6": ['1-159928647-G-A'],
    "CONTROL_7": ['11-83168850-G-A'],
    "CONTROL_8": ['5-119164569-A-G'],
    "CONTROL_9": ['10-80609470-G-A'],
    "CONTROL_10": ['19-10682657-G-T'],
    "CONTROL_11": ['9-37745307-G-A'],
    "CONTROL_12": ['14-73491111-C-A'],
    "CONTROL_13": ['7-142313561-C-G'],
    "CONTROL_14": ['2-11634265-A-G'],
    "CONTROL_15": ['22-50431263-G-C'],
    "CONTROL_16": ['17-42122873-T-C'],
    "CONTROL_17": ['7-108105987-G-A'],
    "CONTROL_18": ['9-34660867-C-T'],
    "CONTROL_19": ['6-149919789-G-A'],
    "CONTROL_20": ['X-50391446-A-T'],
    "CONTROL_21": ['1-43338136-G-T'],
    "CONTROL_22": ['1-39760462-C-T'],
    "CONTROL_23": ['8-10623072-G-C'],
    "CONTROL_24": ['11-46311152-C-T'],
    "CONTROL_25": ['16-2831929-A-G'],
    "CONTROL_26": ['19-19004455-C-T'],
    "CONTROL_27": ['1-219211603-C-G'],
    "CONTROL_28": ['4-95202802-G-C'],
    "CONTROL_29": ['19-40014624-C-T'],
    "CONTROL_30": ['7-102314571-C-A'],
    "CONTROL_31": ['11-86400792-C-A'],
    "CONTROL_32": ['20-35529367-G-A'],
    "CONTROL_33": ['19-15087430-G-A'],
    "CONTROL_34": ['6-9932907-G-A'],
    "CONTROL_35": ['X-105909821-G-A'],
    "CONTROL_36": ['21-39200378-C-T'],
    "CONTROL_37": ['2-74414385-T-A'],
    "CONTROL_38": ['1-179109055-T-G'],
    "CONTROL_39": ['4-93829340-C-G'],
    "CONTROL_40": ['11-59094613-C-T'],
    "CONTROL_41": ['X-49226037-C-T'],
    "CONTROL_42": ['17-41420966-G-A'],
    "CONTROL_43": ['6-134262022-G-C'],
    "CONTROL_44": ['1-58538945-G-A'],
    "CONTROL_45": ['16-3556946-G-T'],
    "CONTROL_46": ['15-20534993-T-C'],
    "CONTROL_47": ['16-67187591-C-T'],
    "CONTROL_48": ['2-240877573-G-A'],
    "CONTROL_49": ['6-31529144-A-G'],
    "CONTROL_50": ['2-88978700-A-G'],
    "CONTROL_51": ['17-74440231-G-A'],
    "CONTROL_52": ['18-50040277-A-G'],
    "CONTROL_53": ['20-21714709-C-T'],
    "CONTROL_54": ['20-44750278-T-C'],
    "CONTROL_55": ['17-80086022-T-C'],
    "CONTROL_56": ['8-86905127-C-A'],
    "CONTROL_57": ['1-86903448-C-A'],
    "CONTROL_58": ['4-154610181-A-G'],
    "CONTROL_59": ['16-87420704-A-G'],
    "CONTROL_60": ['12-8223472-T-C'],
    "CONTROL_61": ['7-103651754-G-A'],
    "CONTROL_62": ['19-47067086-G-A'],
    "CONTROL_63": ['6-32642187-G-T'],
    "CONTROL_64": ['2-151508075-G-A'],
    "CONTROL_65": ['6-26383832-C-T'],
    "CONTROL_66": ['1-31799453-T-C'],
    "CONTROL_67": ['1-169541652-G-C'],
    "CONTROL_68": ['9-35811232-A-G'],
    "CONTROL_69": ['4-120798389-A-T'],
    "CONTROL_70": ['11-72824802-G-C'],
    "CONTROL_71": ['22-44285437-G-A'],
    "CONTROL_72": ['2-167244876-G-A'],
    "CONTROL_73": ['9-711359-C-T'],
    "CONTROL_74": ['4-76367680-C-T'],
    "CONTROL_75": ['19-50662518-G-A'],
    "CONTROL_76": ['19-51104412-A-G'],
    "CONTROL_77": ['12-57004092-A-G'],
    "CONTROL_78": ['1-26951948-G-A'],
    "CONTROL_79": ['7-12344412-T-C'],
    "CONTROL_80": ['1-99875226-G-T'],
    "CONTROL_81": ['19-12649959-C-T'],
    "CONTROL_82": ['19-39173128-G-A'],
    "CONTROL_83": ['3-98087607-G-T'],
    "CONTROL_84": ['9-35957618-T-C'],
    "CONTROL_85": ['14-94378610-C-T'],
    "CONTROL_86": ['14-54991322-A-G'],
    "CONTROL_87": ['1-155190621-G-T'],
    "CONTROL_88": ['4-119057641-C-T'],
    "CONTROL_89": ['8-143325505-T-C'],
    "CONTROL_90": ['13-46276729-G-A'],
    "CONTROL_91": ['7-83155988-G-A'],
    "CONTROL_92": ['1-13778554-C-A'],
    "CONTROL_93": ['12-98533134-T-G'],
    "CONTROL_94": ['17-39715782-G-T'],
    "CONTROL_95": ['5-90693985-A-G'],
    "CONTROL_96": ['8-123151931-G-A'],
    "CONTROL_97": ['5-79734815-A-G'],
    "CONTROL_98": ['4-186167256-T-G'],
    "CONTROL_99": ['10-94942309-G-A'],
    "CONTROL_100": ['1-36472100-T-C'],
    "CONTROL_101": ['16-67187541-G-C'],
    "CONTROL_102": ['7-142221661-A-T'],
    "CONTROL_103": ['20-57498588-C-T'],
    "CONTROL_104": ['16-56885304-A-G'],
    "CONTROL_105": ['1-152787604-G-A'],
    "CONTROL_106": ['16-5077459-T-G'],
    "CONTROL_107": ['18-5396325-G-T'],
    "CONTROL_108": ['12-52601343-C-T'],
    "CONTROL_109": ['22-19209105-C-T'],
    "CONTROL_110": ['11-59211920-G-A'],
    "CONTROL_111": ['11-1246518-C-T'],
    "CONTROL_112": ['17-44666842-A-T'],
    "CONTROL_113": ['2-109615111-C-G'],
    "CONTROL_114": ['17-43904475-G-A'],
    "CONTROL_115": ['4-68654157-G-A'],
    "CONTROL_116": ['7-158138311-C-T'],
    "CONTROL_117": ['10-132200945-A-G'],
    "CONTROL_118": ['15-82436316-G-C'],
    "CONTROL_119": ['1-37931852-C-T'],
    "CONTROL_120": ['19-51136028-C-G'],
    "CONTROL_121": ['2-227008171-C-T'],
    "CONTROL_122": ['5-42718826-G-T'],
    "CONTROL_123": ['1-161998259-C-T'],
    "CONTROL_124": ['19-44101171-A-G'],
    "CONTROL_125": ['1-1495729-G-A'],
    "CONTROL_126": ['19-14090102-T-A'],
    "CONTROL_127": ['12-25519946-T-C'],
    "CONTROL_128": ['1-101239021-C-G'],
    "CONTROL_129": ['1-203171430-C-T'],
    "CONTROL_130": ['2-26473506-C-T'],
    "CONTROL_131": ['20-890224-T-C'],
    "CONTROL_132": ['6-146433996-G-A'],
    "CONTROL_133": ['15-71737855-C-T'],
    "CONTROL_134": ['9-19058638-C-T'],
    "CONTROL_135": ['11-34154715-T-C'],
    "CONTROL_136": ['12-132570115-C-T'],
    "CONTROL_137": ['18-50385224-G-T'],
    "CONTROL_138": ['19-37698654-T-C'],
    "CONTROL_139": ['7-142482737-G-A'],
    "CONTROL_140": ['11-65947112-A-T'],
    "CONTROL_141": ['4-140398927-G-A'],
    "CONTROL_142": ['8-143558433-C-T'],
    "CONTROL_143": ['22-31713925-T-C'],
    "CONTROL_144": ['19-9186520-G-A'],
    "CONTROL_145": ['16-30558028-C-T'],
    "CONTROL_146": ['19-5748246-A-G'],
    "CONTROL_147": ['14-37749781-A-G'],
    "CONTROL_148": ['16-67401014-C-T'],
    "CONTROL_149": ['6-28504476-T-C'],
    "CONTROL_150": ['5-160010614-C-G'],
    "CONTROL_151": ['9-136476654-G-A'],
    "CONTROL_152": ['6-24843299-C-T'],
    "CONTROL_153": ['18-80136587-A-G'],
    "CONTROL_154": ['11-74927429-T-C'],
    "CONTROL_155": ['11-67665398-G-A'],
    "CONTROL_156": ['9-132646029-C-T'],
    "CONTROL_157": ['14-68874941-G-A'],
    "CONTROL_158": ['11-27368673-T-C'],
    "CONTROL_159": ['22-50145701-G-A'],
    "CONTROL_160": ['11-118013843-T-C'],
    "CONTROL_161": ['1-109236451-G-A'],
    "CONTROL_162": ['17-34287653-A-C'],
    "CONTROL_163": ['1-2787560-C-T'],
    "CONTROL_164": ['5-66054584-G-T'],
    "CONTROL_165": ['7-100995659-T-C'],
    "CONTROL_166": ['22-39969459-G-A'],
    "CONTROL_167": ['5-110568548-C-T'],
    "CONTROL_168": ['12-95917258-T-C'],
    "CONTROL_169": ['16-786081-C-A'],
    "CONTROL_170": ['7-75424193-G-T'],
    "CONTROL_171": ['19-8969867-G-A'],
    "CONTROL_172": ['9-41894095-T-A'],
    "CONTROL_173": ['15-23129745-C-T'],
    "CONTROL_174": ['11-3120504-C-T'],
    "CONTROL_175": ['9-137252108-A-G'],
    "CONTROL_176": ['12-40346884-A-G'],
    "CONTROL_177": ['8-85329771-G-A'],
    "CONTROL_178": ['3-156120699-G-C'],
    "CONTROL_179": ['5-76819244-C-T'],
    "CONTROL_180": ['12-10806297-T-G'],
    "CONTROL_181": ['11-1250664-C-G'],
    "CONTROL_182": ['22-23613322-G-C'],
    "CONTROL_183": ['16-68291300-T-A'],
    "CONTROL_184": ['6-169743204-A-G'],
    "CONTROL_185": ['19-14055011-C-T'],
    "CONTROL_186": ['1-15774837-G-A'],
    "CONTROL_187": ['11-66025323-C-T'],
    "CONTROL_188": ['13-95996130-T-C'],
    "CONTROL_189": ['19-13764607-C-T'],
    "CONTROL_190": ['12-106978623-G-A'],
    "CONTROL_191": ['7-93135106-A-G'],
    "CONTROL_192": ['9-131522077-C-T'],
    "CONTROL_193": ['6-32665056-A-G'],
    "CONTROL_194": ['9-22447429-G-T'],
    "CONTROL_195": ['8-673906-C-T'],
    "CONTROL_196": ['5-126360432-T-A'],
    "CONTROL_197": ['6-63280743-A-G'],
    "CONTROL_198": ['9-108893948-T-A'],
    "CONTROL_199": ['1-227747956-G-C'],
    "CONTROL_200": ['1-208217786-C-T'],
}

# Map APOE allele names to their final carrier column names
APOE_MAPPING = {
    'APOE_e4': 'is_apoe_e4_carrier',
    'APOE_e2': 'is_apoe_e2_carrier',
    'APOE_e3': 'is_apoe_e3_carrier'
}

# Query Creation Function
def create_query_simplified(workspace_cdr, variants_map):
    """
    Constructs a simplified BigQuery SQL query to fetch demographics and variant status.
    """
    variant_sqls = []
    for label, vids in variants_map.items():
        vid_list = ', '.join([f"'{vid}'" for vid in vids])
        variant_sqls.append(f"""
            LEFT JOIN (
                SELECT DISTINCT person_id_unnested AS person_id FROM `{workspace_cdr}.cb_variant_to_person`
                CROSS JOIN UNNEST(person_ids) AS person_id_unnested
                WHERE vid IN ({vid_list})
            ) AS {label.lower()}_variant_carriers ON p.person_id = {label.lower()}_variant_carriers.person_id
        """)

    variant_select_sql = ", ".join([f"CASE WHEN {label.lower()}_variant_carriers.person_id IS NOT NULL THEN 1 ELSE 0 END AS is_{label.lower()}_carrier" for label in variants_map.keys()])
    if variant_select_sql:
        variant_select_sql = ", " + variant_select_sql
        
    main_query = f"""
    SELECT
        p.person_id, p.year_of_birth, p.month_of_birth, p.day_of_birth,
        c_gender.concept_name AS sex_at_birth,
        MAX(e.entry_date) AS last_event_datetime
        {variant_select_sql}
    FROM `{workspace_cdr}.person` AS p
    JOIN `{workspace_cdr}.cb_search_person` AS cbsp ON p.person_id = cbsp.person_id
    JOIN `{workspace_cdr}.concept` AS c_gender ON p.gender_concept_id = c_gender.concept_id
    LEFT JOIN `{workspace_cdr}.cb_search_all_events` AS e ON p.person_id = e.person_id
    {' '.join(variant_sqls)}
    WHERE
        p.race_concept_id = 8516 -- Filters for individuals of Black or African American race
        AND cbsp.has_ehr_data = 1 -- Ensures they have Electronic Health Record data
        AND cbsp.has_whole_genome_variant = 1 -- Ensures they have whole genome sequencing data
    GROUP BY
        p.person_id, p.year_of_birth, p.month_of_birth, p.day_of_birth,
        c_gender.concept_name
        {", " + ", ".join([f"{label.lower()}_variant_carriers.person_id" for label in variants_map.keys()]) if variant_sqls else ""}
    HAVING
        MAX(e.entry_date) IS NOT NULL
    """
    return main_query

# APOE Carrier Calculation Function
def calculate_apoe_carriers(df):
    """
    Calculates APOE e2, e3, and e4 carrier status based on user-defined logic
    using two SNPs.
    - ε4: has rs429358-C AND does not have rs7412-T
    - ε2: has rs7412-T
    - ε3: has neither rs429358-C nor rs7412-T
    """
    if 'is_apoe_rs429358_carrier' not in df.columns:
        df['is_apoe_rs429358_carrier'] = 0
    if 'is_apoe_rs7412_carrier' not in df.columns:
        df['is_apoe_rs7412_carrier'] = 0

    df['is_apoe_e4_carrier'] = ((df['is_apoe_rs429358_carrier'] == 1) & (df['is_apoe_rs7412_carrier'] == 0)).astype(int)
    df['is_apoe_e2_carrier'] = df['is_apoe_rs7412_carrier'].astype(int)
    df['is_apoe_e3_carrier'] = ((df['is_apoe_rs429358_carrier'] == 0) & (df['is_apoe_rs7412_carrier'] == 0)).astype(int)
    
    print("  APOE carrier status calculated based on new rules.")
    return df

# Helper Functions
def run_query(query):
    """
    Executes a BigQuery SQL query and returns the results as a Pandas DataFrame.
    """
    try:
        import pandas_gbq
        print("  Attempting to run BigQuery query...")
        results = pandas_gbq.read_gbq(query, project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'), dialect="standard", use_bqstorage_api=True)
        print(f"  BigQuery query executed. Rows returned: {len(results)}")
        return results
    except Exception as e:
        print(f"  ERROR: BigQuery query failed: {e}")
        return pd.DataFrame()

# Analysis Functions
def perform_age_stratified_analysis(data, variant_label, overall_frequencies, age_labels):
    """
    Performs age-stratified analysis of allele frequency for a given variant.
    """
    variant_carrier_col = f'is_{variant_label.lower()}_carrier'
    
    if 'age_bin' not in data.columns or variant_carrier_col not in data.columns:
        return pd.DataFrame()

    # Ensure age_bin is a categorical type with all possible labels to prevent empty groups from being dropped
    data['age_bin'] = data['age_bin'].astype(pd.CategoricalDtype(categories=age_labels, ordered=True))

    age_stratified_summary = data.groupby('age_bin', observed=False).agg(
        total_individuals=(variant_carrier_col, 'count'),
        carrier_count=(variant_carrier_col, 'sum')
    ).reset_index()
    
    age_stratified_summary['carrier_freq'] = age_stratified_summary['carrier_count'] / age_stratified_summary['total_individuals']
    age_stratified_summary['carrier_freq'] = age_stratified_summary['carrier_freq'].fillna(0)

    overall_freq_variant = overall_frequencies.get(variant_label, 0)
    if overall_freq_variant > 0:
        age_stratified_summary['normalized_change'] = (age_stratified_summary['carrier_freq'] - overall_freq_variant) / overall_freq_variant * 100
    else:
        age_stratified_summary['normalized_change'] = 0 

    return age_stratified_summary[['age_bin', 'normalized_change']].rename(columns={'normalized_change': variant_label})

def calculate_control_ci(all_normalized_data, final_control_variants, age_labels):
    """
    Calculates the mean, standard error, and 99.99% confidence interval for control variants.
    """
    control_stats_df = pd.DataFrame({'age_bin': age_labels})

    if not final_control_variants:
        control_stats_df['control_mean'] = 0
        control_stats_df['control_ci_lower'] = 0
        control_stats_df['control_ci_upper'] = 0
        control_stats_df['n_controls'] = 0
        return control_stats_df

    control_data_for_ci = all_normalized_data[[col for col in final_control_variants if col in all_normalized_data.columns]].copy()
    control_data_for_ci = control_data_for_ci.dropna(axis=1, how='all')
    n_controls_present = control_data_for_ci.shape[1]

    if n_controls_present > 0:
        control_mean = control_data_for_ci.mean(axis=1)
        
        if n_controls_present > 1:
            control_std = control_data_for_ci.std(axis=1)
            control_sem = control_std / np.sqrt(n_controls_present)
            margin_of_error = 3.891 * control_sem  # z-score for 99.99% CI
            control_ci_lower = control_mean - margin_of_error
            control_ci_upper = control_mean + margin_of_error
        else:
            control_ci_lower = control_mean
            control_ci_upper = control_mean
            
        temp_df = pd.DataFrame({
            'age_bin': all_normalized_data['age_bin'],
            'control_mean': control_mean,
            'control_ci_lower': control_ci_lower,
            'control_ci_upper': control_ci_upper,
            'n_controls': n_controls_present
        })
        control_stats_df = pd.merge(control_stats_df, temp_df, on='age_bin', how='left')
    else:
        control_stats_df['control_mean'] = 0
        control_stats_df['control_ci_lower'] = 0
        control_stats_df['control_ci_upper'] = 0
        control_stats_df['n_controls'] = 0

    return control_stats_df.fillna(0)

# Plotting Function
def plot_normalized_frequencies(all_normalized_data, specific_variants, control_stats_df, group_title):
    """
    Generates a combined plot for normalized variant frequencies with a 99.99% CI.
    """
    if all_normalized_data.empty:
        print(f"\nNo data to plot for {group_title}.")
        return

    plt.figure(figsize=(14, 8))
    
    for variant_label in specific_variants:
        color = None
        if 'GIMAP6' in variant_label: color = 'red'
        elif 'APOE' in variant_label:
            if 'e4' in variant_label: color = 'darkblue'
            elif 'e2' in variant_label: color = 'skyblue'
            elif 'e3' in variant_label: color = 'green'
        
        if variant_label in all_normalized_data.columns:
            # Convert age_bin to string for plotting to avoid potential categorical issues
            plot_data = all_normalized_data.copy()
            plot_data['age_bin'] = plot_data['age_bin'].astype(str)
            sns.lineplot(x='age_bin', y=variant_label, data=plot_data, 
                         marker='o', label=variant_label, color=color, linewidth=2.5)
        else:
            print(f"  WARNING: Data for variant {variant_label} not found for plotting in {group_title} group.")

    if not control_stats_df.empty and 'control_mean' in control_stats_df.columns:
        n_controls = int(control_stats_df['n_controls'].iloc[0]) if not control_stats_df.empty else 0
        if n_controls > 0:
            # Convert age_bin to string for plotting
            plot_control_data = control_stats_df.copy()
            plot_control_data['age_bin'] = plot_control_data['age_bin'].astype(str)
            
            plt.plot(plot_control_data['age_bin'], plot_control_data['control_mean'], 
                     label=f'Control Mean (N={n_controls})', color='gray', linestyle='--', linewidth=2)
            plt.fill_between(plot_control_data['age_bin'], plot_control_data['control_ci_lower'], plot_control_data['control_ci_upper'], 
                             color='gray', alpha=0.2, label='Control 99.99% CI')
        else:
            print(f"  WARNING: No valid control data to plot CI for {group_title}.")
    else:
        print(f"  No control variants selected for {group_title}. Skipping control CI plot.")

    plt.title(f'Normalized Carrier Frequency Change by Age Group ({group_title})')
    plt.xlabel('Age Group (Years)')
    plt.ylabel('Percentage Change from Overall Cohort Frequency (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
    plt.legend(title='Variant')
    plt.show()

# Main Analysis Orchestrator
def run_focused_analysis_simplified(workspace_cdr, all_variants_map):
    """
    Orchestrates the data extraction, pre-processing, data printing, and plot generation.
    """
    print("=======================================================")
    print("       Starting Simplified Analysis: Allele Freq vs. Age      ")
    print("=======================================================")
    
    print("\nStep 1: Fetching and processing ALL required data from BigQuery...")
    main_query = create_query_simplified(workspace_cdr, all_variants_map)
    data = run_query(main_query)
    
    if data.empty:
        print("\nFATAL ERROR: No data retrieved from BigQuery. Cannot proceed with analysis.")
        return

    print("\nStep 2: Preprocessing cohort data...")
    data = calculate_apoe_carriers(data)

    date_components = data[['year_of_birth', 'month_of_birth', 'day_of_birth']].fillna(1)
    date_components.columns = ['year', 'month', 'day']
    data['birth_datetime'] = pd.to_datetime(date_components)
    data['last_visit_datetime'] = pd.to_datetime(data['last_event_datetime'])
    data = data.dropna(subset=['birth_datetime', 'last_visit_datetime']).copy()
    
    data['age'] = (data['last_visit_datetime'] - data['birth_datetime']).dt.days / 365.25
    data = data[(data['age'] >= 18) & (data['age'] <= 100)].copy()
    data['sex'] = data['sex_at_birth'].apply(lambda x: 'Female' if 'Female' in str(x) else 'Male')

    # --- MODIFIED: Age bins and labels are updated here ---
    print("  Combining age groups under 65 into a single bin.")
    age_bins = [18, 65, 75, 85, 101]
    age_labels = ['18-64', '65-74', '75-84', '85+']
    data['age_bin'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)
    
    print(f"  Preprocessing complete. Final cohort size: {len(data)} individuals.")
    
    all_analysis_variants = list(all_variants_map.keys()) + list(APOE_MAPPING.keys())
    all_analysis_variants.remove('APOE_rs429358')
    all_analysis_variants.remove('APOE_rs7412')

    print("\nStep 3: Calculating overall variant frequencies for normalization baseline...")
    overall_frequencies_for_normalization = {}
    for variant_label in all_analysis_variants:
        variant_carrier_col = f'is_{variant_label.lower()}_carrier'
        if variant_carrier_col in data.columns:
            overall_frequencies_for_normalization[variant_label] = data[variant_carrier_col].sum() / len(data)
        else:
            overall_frequencies_for_normalization[variant_label] = 0

    print("\nStep 4: Filtering control variants by frequency (5% to 10%)...")
    
    MIN_CONTROL_FREQUENCY = 0.05
    MAX_CONTROL_FREQUENCY = 0.10

    specific_variants_to_plot = []
    final_control_variants_to_plot = []

    for label in all_analysis_variants:
        is_control = "CONTROL" in label.upper()
        is_specific = not is_control
        
        if is_specific:
            specific_variants_to_plot.append(label)
        elif is_control:
            overall_freq = overall_frequencies_for_normalization.get(label, 0)
            if MIN_CONTROL_FREQUENCY <= overall_freq <= MAX_CONTROL_FREQUENCY:
                final_control_variants_to_plot.append(label)
            else:
                print(f"  Skipping control variant {label} due to frequency outside 5-10% range ({overall_freq:.4f})")

    if not specific_variants_to_plot:
        print("  WARNING: No specific variants found to plot individually.")
    if not final_control_variants_to_plot:
        print("  WARNING: No control variants passed the frequency filter for CI calculation.")
    
    print(f"  Specific variants for analysis: {specific_variants_to_plot}")
    print(f"  Control variants for CI (after filter): {final_control_variants_to_plot}")

    analysis_groups = [
        ("Overall Cohort", data.copy()),
        ("Male Cohort", data[data['sex'] == 'Male'].copy()),
        ("Female Cohort", data[data['sex'] == 'Female'].copy())
    ]

    print("\nStep 5: Performing analysis, printing data, and plotting for each group...")
    for group_name, group_data in analysis_groups:
        print(f"\n" + "="*80)
        print(f"=== Analyzing Group: {group_name} (Size: {len(group_data)}) ===")
        print(f"="*80)

        if group_data.empty:
            print(f"  SKIPPING: {group_name} data is empty.")
            continue

        all_normalized_data_for_group = pd.DataFrame({'age_bin': age_labels})
        # Use the master list of labels for the loop to ensure all variants are processed
        for variant_label in all_analysis_variants:  
            carrier_col_name = f'is_{variant_label.lower()}_carrier'
            if carrier_col_name not in group_data.columns:
                continue
            normalized_df = perform_age_stratified_analysis(group_data.copy(), variant_label, 
                                                            overall_frequencies_for_normalization, age_labels)
            if not normalized_df.empty:
                # Set age_bin as index to merge correctly even if rows are out of order
                if 'age_bin' not in all_normalized_data_for_group.columns:
                     all_normalized_data_for_group = normalized_df
                else:
                     all_normalized_data_for_group = pd.merge(all_normalized_data_for_group, normalized_df, on='age_bin', how='left')
        
        all_normalized_data_for_group = all_normalized_data_for_group.fillna(0)
        control_stats_df = calculate_control_ci(all_normalized_data_for_group, final_control_variants_to_plot, age_labels)
        
        variants_df = all_normalized_data_for_group[['age_bin'] + [v for v in specific_variants_to_plot if v in all_normalized_data_for_group.columns]]
        final_output_df = pd.merge(variants_df, control_stats_df, on='age_bin')

        print(f"\nDATA for {group_name}:")
        print("Values represent the Percentage Change from Overall Cohort Frequency (%).")
        # Ensure age_bin is treated as a string for consistent printing
        final_output_df['age_bin'] = final_output_df['age_bin'].astype(str)
        print(final_output_df.to_string())
        print("-" * 80)
        
        plot_normalized_frequencies(
            all_normalized_data_for_group,
            specific_variants_to_plot,
            control_stats_df,
            group_name
        )

    print("\n\n" + "#"*100)
    print("                          Simplified Analysis Complete")
    print("#"*100)

# Main Execution Block
def main():
    os.environ.setdefault('WORKSPACE_CDR', 'fc-aou-cdr-prod-ct.C2024Q3R5')
    workspace_cdr = os.environ['WORKSPACE_CDR']
    print(f"Using WORKSPACE_CDR: {workspace_cdr}")
    
    run_focused_analysis_simplified(
        workspace_cdr,
        VARIANTS
    )

if __name__ == "__main__":
    main()
