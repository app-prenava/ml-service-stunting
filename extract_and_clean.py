import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("Starting data extraction...")
data_dir = "data/input/hh14_all_dta"

# 1. Anthropometry (bus_us.dta)
print("Loading anthropometry data...")
bus_us = pd.read_stata(os.path.join(data_dir, "bus_us.dta"), convert_categoricals=False)

# 2. Household Roster (bk_ar1.dta)
print("Loading household roster...")
bk_ar1 = pd.read_stata(os.path.join(data_dir, "bk_ar1.dta"), convert_categoricals=False)
pid_map = bk_ar1.set_index(['hhid14_9', 'pid14'])['pidlink'].to_dict()

# 3. Pregnancy History (b4_ch1.dta)
print("Loading pregnancy history...")
b4_ch1 = pd.read_stata(os.path.join(data_dir, "b4_ch1.dta"), convert_categoricals=False)
b4_ch1 = b4_ch1[(b4_ch1['ch06'] == 2) & (b4_ch1['ch07_id'] < 50)].copy()

def get_child_pidlink(row):
    try:
        if pd.isna(row['ch07_id']): return np.nan
        return pid_map.get((row['hhid14_9'], float(row['ch07_id'])), np.nan)
    except:
        return np.nan

b4_ch1['child_pidlink'] = b4_ch1.apply(get_child_pidlink, axis=1)
b4_ch1 = b4_ch1.dropna(subset=['child_pidlink'])

# Extract Features
# ch05 = birth order, ch10a = mother age, ch16a-c = anc, ch16f = iron, ch18a/c = sick, ch19 = birth location, ch08 = gender, ch24 = birth weight
preg_features = b4_ch1[['child_pidlink', 'pidlink', 'ch05', 'ch10a', 'ch16a', 'ch16b', 'ch16c', 'ch16f', 'ch18a', 'ch18c', 'ch19', 'ch08', 'ch24']]
preg_features = preg_features.rename(columns={'pidlink': 'mother_pidlink', 'ch10a': 'mother_age_preg', 'ch05': 'birth_order', 'ch08': 'child_gender', 'ch24': 'birth_weight'})
preg_features = preg_features.drop_duplicates(subset=['child_pidlink'])

# Get child heights
child_anthro = bus_us[['pidlink', 'us04', 'us06']].rename(columns={'pidlink': 'child_pidlink', 'us04': 'child_height', 'us06': 'child_weight'})
child_anthro = child_anthro.drop_duplicates(subset=['child_pidlink'])
merged = preg_features.merge(child_anthro, on='child_pidlink', how='inner')

# Get mother heights
mother_anthro = bus_us[['pidlink', 'us04', 'us06', 'us13']].rename(columns={'pidlink': 'mother_pidlink', 'us04': 'mother_height', 'us06': 'mother_weight', 'us13': 'maternal_hemoglobin'})
mother_anthro = mother_anthro.drop_duplicates(subset=['mother_pidlink'])
merged = merged.merge(mother_anthro, on='mother_pidlink', how='left')

# Mother Education (b3a_dl1.dta)
print("Loading education data...")
dl1 = pd.read_stata(os.path.join(data_dir, "b3a_dl1.dta"), convert_categoricals=False)
dl1 = dl1[['pidlink', 'dl06']].rename(columns={'pidlink': 'mother_pidlink', 'dl06': 'mother_education'})
dl1 = dl1.drop_duplicates(subset=['mother_pidlink'])
merged = merged.merge(dl1, on='mother_pidlink', how='left')

# Get child age
child_age = bk_ar1[['pidlink', 'ar09']].rename(columns={'pidlink': 'child_pidlink', 'ar09': 'child_age_years'})
child_age = child_age.drop_duplicates(subset=['child_pidlink'])
merged = merged.merge(child_age, on='child_pidlink', how='left')
merged = merged[(merged['child_age_years'] >= 0) & (merged['child_age_years'] <= 5)]

# Get mother current age as fallback
mother_age_now = bk_ar1[['pidlink', 'ar09']].rename(columns={'pidlink': 'mother_pidlink', 'ar09': 'mother_age_current'})
mother_age_now = mother_age_now.drop_duplicates(subset=['mother_pidlink'])
merged = merged.merge(mother_age_now, on='mother_pidlink', how='left')

# 4. Clean and Create Features
print("Cleaning and engineering features...")
numeric_cols = ['child_height', 'mother_height', 'mother_weight', 'maternal_hemoglobin', 'mother_age_preg', 'ch16a', 'ch16b', 'ch16c', 'birth_order', 'mother_education', 'birth_weight', 'child_gender']
for col in numeric_cols:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')
    if col in ['mother_height', 'mother_weight', 'child_height', 'maternal_hemoglobin']:
        merged.loc[merged[col] >= 998, col] = np.nan # IFLS 998/999 missing code for anthro
    elif col == 'birth_weight':
        merged.loc[merged[col] >= 98, col] = np.nan # Missing code for birth weight usually 98 or 99
    elif col == 'child_gender':
        merged.loc[merged[col] > 3, col] = np.nan # 1 = Male, 3 = Female
    else:
        merged.loc[merged[col] >= 98, col] = np.nan # standard missing

# Fallback Mother Age
merged['mother_age'] = merged['mother_age_preg'].fillna(merged['mother_age_current'] - merged['child_age_years'])

# Stunting Target
mean_height = {0: 64.5, 1: 79.5, 2: 89.5, 3: 97.5, 4: 104.5, 5: 109.5}
sd_height = {0: 2.5, 1: 3.0, 2: 3.5, 3: 4.0, 4: 4.5, 5: 4.5}

def calculate_stunting(row):
    age = row['child_age_years']
    height = row['child_height']
    if pd.isna(age) or pd.isna(height) or age > 5:
        return np.nan
    z_score = (height - mean_height[age]) / sd_height[age]
    return 1 if z_score < -2 else 0

merged['stunting'] = merged.apply(calculate_stunting, axis=1)

# Feature Engineering
merged['iron_pills'] = merged['ch16f'].apply(lambda x: 1 if x == 1 else (0 if x == 3 else np.nan))
merged['anc_visits_total'] = merged[['ch16a', 'ch16b', 'ch16c']].sum(axis=1)
merged['pregnancy_sickness'] = ((merged['ch18a'] == 1) | (merged['ch18c'] == 1)).astype(int)
merged['birth_location_facility'] = merged['ch19'].apply(lambda x: 1 if x in [1, 2, 3, 4] else (0 if not pd.isna(x) else np.nan)) # 1-4 is hospital/clinic, else home


# BMI Calculation
merged['maternal_bmi'] = merged['mother_weight'] / ((merged['mother_height'] / 100) ** 2)
merged.loc[(merged['maternal_bmi'] < 10) | (merged['maternal_bmi'] > 60), 'maternal_bmi'] = np.nan

# --- NOISE SIMULATION (USG Error) ---
np.random.seed(42)
# 2.8% error margin for USG length (child_height)
error_multiplier = np.random.normal(loc=1.0, scale=0.028, size=len(merged))
merged['child_height'] = merged['child_height'] * error_multiplier
# ------------------------------------

# Final Dataset
features = ['mother_pidlink', 'child_pidlink', 'stunting', 'mother_height', 'mother_weight', 'maternal_bmi', 'maternal_hemoglobin', 'mother_age', 'mother_education', 'birth_order', 'anc_visits_total', 'iron_pills', 'pregnancy_sickness', 'birth_location_facility', 'birth_weight', 'child_gender', 'child_height', 'child_age_years']
final_df = merged[features].dropna(subset=['stunting'])

# Impute to prevent NaNs
for col in final_df.columns:
    if col not in ['stunting', 'mother_pidlink', 'child_pidlink']:
        final_df[col] = final_df[col].fillna(final_df[col].median())

print(f"Final dataset shape: {final_df.shape}")
print(f"Stunting cases: {final_df['stunting'].sum()} ({final_df['stunting'].mean()*100:.1f}%)")

os.makedirs("data/processed", exist_ok=True)
final_df.to_csv("data/processed/stunting_pregnancy_features.csv", index=False)
print("Saved to data/processed/stunting_pregnancy_features.csv")
