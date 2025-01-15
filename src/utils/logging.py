columns_of_interest = [
    'PT08.S1(CO)', 'PT08.S2(NMHC)',
    'PT08.S3(NOx)', 'PT08.S4(NO2)',
    'PT08.S5(O3)', 'T', 'RH', 'AH'
]

sns.pairplot(airqual_df_filtered[columns_of_interest], diag_kind='kde')

plt.subplots_adjust(top=0.95)
plt.suptitle('Pair Plot of Air Quality and Meteorological Indicators',
             fontsize=16)
# plt.savefig('pair_plots_filtered.png', dpi=300)
plt.show()

correlation_matrix = airqual_df_filtered.corr()

plt.figure(figsize=(9, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Air Quality and Meteorological Variables')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('correlation_analysis.png', dpi=300)
plt.show()

