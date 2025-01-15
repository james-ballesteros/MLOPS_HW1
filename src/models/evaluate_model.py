CO_score_df_results = pd.concat(
    all_model_results[0], ignore_index=True) \
                        .sort_values(by='Test R2',
                                     ascending=False) \
                        .reset_index(drop=True)
CO_score_df_results.index += 1
CO_score_df_results

NMHC_score_df_results = pd.concat(
    all_model_results[0], ignore_index=True) \
                        .sort_values(by='Test R2',
                                     ascending=False) \
                        .reset_index(drop=True)
NMHC_score_df_results.index += 1
NMHC_score_df_results

NOx_score_df_results = pd.concat(
    all_model_results[0], ignore_index=True) \
                        .sort_values(by='Test R2',
                                     ascending=False) \
                        .reset_index(drop=True)
NOx_score_df_results.index += 1
NOx_score_df_results

NO2_score_df_results = pd.concat(
    all_model_results[0], ignore_index=True) \
                        .sort_values(by='Test R2',
                                     ascending=False) \
                        .reset_index(drop=True)
NO2_score_df_results.index += 1
NO2_score_df_results

O3_score_df_results = pd.concat(
    all_model_results[0], ignore_index=True) \
                        .sort_values(by='Test R2',
                                     ascending=False) \
                        .reset_index(drop=True)
O3_score_df_results.index += 1
O3_score_df_results

