%%HTML
<script src="require.js"></script>

from IPython.display import HTML
HTML('''<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js "></script><script>
code_show=true; 
function code_toggle() {
if (code_show){
$('div.jp-CodeCell > div.jp-Cell-inputWrapper').hide();
} else {
$('div.jp-CodeCell > div.jp-Cell-inputWrapper').show();
}
code_show = !code_show
} 
$( document ).ready(code_toggle);</script><form action="javascript:code_toggle()"><input type="submit" value="Toggle on/off for raw code"></form>
''')

airqual_df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1,
                inplace=True, errors='ignore')
airqual_df

df_airqual_percentage_NaNs = (airqual_df.isna().sum(
) / len(airqual_df) * 100).to_frame(name='%_NaN').sort_values(
    '%_NaN', ascending=False)
df_airqual_percentage_NaNs

airqual_df = airqual_df.dropna()
airqual_df

airqual_df.info()

airqual_df.loc[:, :] = airqual_df.replace(to_replace=-200, value=np.nan)
airqual_df

airqual_df.info()

df_airqual_percentage_NaNs = (airqual_df.isna().sum(
) / len(airqual_df) * 100).to_frame(name='%_NaN').sort_values(
    '%_NaN', ascending=False)
df_airqual_percentage_NaNs

airqual_df = airqual_df.copy()
airqual_df.drop('NMHC(GT)', axis=1, inplace=True, errors='ignore')
airqual_df

df_airqual_percentage_NaNs = (airqual_df.isna().sum(
) / len(airqual_df) * 100).to_frame(name='%_NaN').sort_values(
    '%_NaN', ascending=False)
df_airqual_percentage_NaNs

numerical_columns = airqual_df.select_dtypes(include=[np.number]).columns

n_cols = 3
n_rows = len(numerical_columns) // n_cols + \
    (len(numerical_columns) % n_cols > 0)

plt.figure(figsize=(n_cols * 6, n_rows * 5))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=airqual_df[column])
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
# plt.savefig('boxplots_raw.png', dpi=300)
plt.show()

numeric_cols = airqual_df.select_dtypes(include=[np.number]).columns

# Apply IQR to numeric columns only
Q1 = airqual_df[numeric_cols].quantile(0.25)
Q3 = airqual_df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

condition = ~((airqual_df[numeric_cols] < lower_bound) | (
    airqual_df[numeric_cols] > upper_bound)).any(axis=1)

airqual_df_filtered = airqual_df[condition]
airqual_df_filtered

airqual_df_filtered.info()

df_airqual_percentage_NaNs = (airqual_df_filtered.isna().sum(
) / len(airqual_df_filtered) * 100).to_frame(name='%_NaN').sort_values(
    '%_NaN', ascending=False)
df_airqual_percentage_NaNs

df_airqual_percentage_NaNs = (airqual_df_filtered.isna().sum(
) / len(airqual_df_filtered) * 100).to_frame(name='%_NaN').sort_values(
    '%_NaN', ascending=False)
df_airqual_percentage_NaNs

!pip install missingno

import missingno as msno


columns_of_interest = ['CO(GT)', 'NO2(GT)', 'NOx(GT)']
subset_airqual_df_filtered = airqual_df_filtered[columns_of_interest]

msno.matrix(subset_airqual_df_filtered)
plt.figure(figsize=(9, 6))
# plt.savefig('missing_no.png', dpi=300)
plt.show()

missing_co_gt_indices = airqual_df_filtered[airqual_df_filtered['CO(GT)'].isnull(
)].index.astype(int).tolist()
missing_no2_gt_indices = airqual_df_filtered[airqual_df_filtered['NO2(GT)'].isnull(
)].index.astype(int).tolist()
missing_nox_gt_indices = airqual_df_filtered[airqual_df_filtered['NOx(GT)'].isnull(
)].index.astype(int).tolist()

common_missing_indices = list(set(missing_co_gt_indices) & set(
    missing_no2_gt_indices) & set(missing_nox_gt_indices))

max_length = max(len(missing_co_gt_indices), len(
    missing_no2_gt_indices), len(missing_nox_gt_indices))

missing_co_gt_indices.extend(
    [None] * (max_length - len(missing_co_gt_indices)))
missing_no2_gt_indices.extend(
    [None] * (max_length - len(missing_no2_gt_indices)))
missing_nox_gt_indices.extend(
    [None] * (max_length - len(missing_nox_gt_indices)))

missing_indices_df = pd.DataFrame({
    'Missing CO(GT)': missing_co_gt_indices,
    'Missing NO2(GT)': missing_no2_gt_indices,
    'Missing NOx(GT)': missing_nox_gt_indices
})

print(
    f"Count of common missing indices for CO(GT), NO2(GT), and NOx(GT): {len(common_missing_indices)}")

missing_indices_df

airqual_df_filtered = airqual_df_filtered.drop(common_missing_indices)
airqual_df_filtered

airqual_df_filtered.describe()

airqual_df_filtered.info()

df_airqual_percentage_NaNs = (airqual_df_filtered.isna().sum(
) / len(airqual_df_filtered) * 100).to_frame(name='%_NaN').sort_values(
    '%_NaN', ascending=False)
df_airqual_percentage_NaNs

airqual_df_filtered.describe()

airqual_df_filtered.info()

numerical_columns = airqual_df_filtered.select_dtypes(
    include=[np.number]).columns

n_cols = 3
n_rows = len(numerical_columns) // n_cols + (
    len(numerical_columns) % n_cols > 0)

plt.figure(figsize=(n_cols * 6, n_rows * 5))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=airqual_df_filtered[column])
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
# plt.savefig('boxplots_intiial.png', dpi=300)
plt.show()

airqual_df_filtered.drop(['Date'], axis=1, inplace=True, errors='ignore')
airqual_df_filtered.drop(['Time'], axis=1, inplace=True, errors='ignore')
airqual_df_filtered.drop(['CO(GT)'], axis=1, inplace=True, errors='ignore')
airqual_df_filtered.drop(['C6H6(GT)'], axis=1, inplace=True, errors='ignore')
airqual_df_filtered.drop(['NOx(GT)'], axis=1, inplace=True, errors='ignore')
airqual_df_filtered.drop(['NO2(GT)'], axis=1, inplace=True, errors='ignore')
airqual_df_filtered

airqual_df_filtered.describe()

airqual_df_filtered.info()

numerical_columns = airqual_df_filtered.select_dtypes(
    include=[np.number]).columns

n_cols = 3
n_rows = len(numerical_columns) // n_cols + (
    len(numerical_columns) % n_cols > 0)

plt.figure(figsize=(n_cols * 6, n_rows * 5))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=airqual_df_filtered[column])
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
# plt.savefig('boxplots_final.png', dpi=300)
plt.show()

