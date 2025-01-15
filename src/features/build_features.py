median_imputer = SimpleImputer(strategy='median')

columns = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
           'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

for column in columns:
    imputed_data = median_imputer.fit_transform(airqual_df_filtered[[column]])
    airqual_df_filtered.loc[:, column] = imputed_data

airqual_df_filtered

median_imputer = SimpleImputer(strategy='median')

for column in ['CO(GT)', 'NO2(GT)', 'NOx(GT)']:
    airqual_df_filtered[column] = median_imputer.fit_transform(
        airqual_df_filtered[[column]])

airqual_df_filtered

def tune_hyperparameters(df, feature):
    """
    Tune hyperparameters for GradientBoostingRegressor for a given
    target feature using GridSearchCV.

    Parameters:
    - df: DataFrame containing the features and target.
    - feature: The name of the target feature for which to tune the model.

    Returns:
    - A tuple containing the best parameters and the best score.
    """

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2]
    }

    X = df.drop([feature], axis=1)
    y = df[feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1337)

    gb = GradientBoostingRegressor(random_state=1337)
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5,
                               scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score

def train_and_evaluate(df, feature, best_params):
    """
    Train GradientBoostingRegressor with the best hyperparameters
    and evaluate it on the test set.

    Parameters:
    - df: DataFrame containing the features and target.
    - feature: The name of the target feature.
    - best_params: The best hyperparameters found via GridSearchCV.

    Returns:
    - The trained model and the R^2 score on the test set.
    """

    # Prepare Data
    X = df.drop([feature], axis=1)
    y = df[feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1337)

    # Initialize and train the model
    model = GradientBoostingRegressor(random_state=1337, **best_params)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return model, score

# List of features to process
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
            'PT08.S4(NO2)', 'PT08.S5(O3)']

fig, axs = plt.subplots(3, 2, figsize=(18, 24))
axs = axs.flatten()

for i, feature in enumerate(features):
    best_params, best_score = tune_hyperparameters(
        airqual_df_filtered, feature)
    model, score = train_and_evaluate(
        airqual_df_filtered, feature, best_params)

    # Splitting data and making predictions
    X_train, X_test, y_train, y_test = train_test_split(
        airqual_df_filtered.drop([feature], axis=1),
        airqual_df_filtered[feature], test_size=0.2, random_state=1337)
    predictions = model.predict(X_test)

    # Create a DataFrame for plotting
    final = pd.DataFrame({'Measured': y_test, 'Predicted': predictions})

    # Plotting on the ith subplot
    sns.histplot(final, kde=True, element="step", stat="density",
                 common_norm=False, ax=axs[i])
    axs[i].set_title(f'Comparison of Measured and Predicted {feature} Levels')
    axs[i].set_xlabel(f'{feature} Level')
    axs[i].set_ylabel('Density')
    axs[i].legend(labels=['Measured', 'Predicted'])


if len(features) % 2 != 0:
    fig.delaxes(axs[-1])

plt.tight_layout()
# plt.savefig('comparison_measured_predicted_levels.png', dpi=300)
plt.show()

# List of features to process
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
            'PT08.S4(NO2)', 'PT08.S5(O3)']

optimal_params = {
    'PT08.S1(CO)': {'learning_rate': 0.1, 'max_depth': 5,
                    'min_samples_leaf': 1, 'min_samples_split': 2,
                    'n_estimators': 300},
    'PT08.S2(NMHC)': {'learning_rate': 0.1, 'max_depth': 5,
                      'min_samples_leaf': 1, 'min_samples_split': 3,
                      'n_estimators': 300},
    'PT08.S3(NOx)': {'learning_rate': 0.1, 'max_depth': 5,
                     'min_samples_leaf': 2, 'min_samples_split': 2,
                     'n_estimators': 300},
    'PT08.S4(NO2)': {'learning_rate': 0.1, 'max_depth': 5,
                     'min_samples_leaf': 2, 'min_samples_split': 2,
                     'n_estimators': 300},
    'PT08.S5(O3)': {'learning_rate': 0.1, 'max_depth': 5,
                    'min_samples_leaf': 2, 'min_samples_split': 2,
                    'n_estimators': 300}
}

# Subplot grid setup
n = len(features)
cols = 2  
rows = n // cols + (n % cols > 0)

plt.figure(figsize=(12, 4 * rows))

for i, feature in enumerate(features):
    X = airqual_df_filtered.drop(feature, axis=1)
    y = airqual_df_filtered[feature]

    # Initialize and train model
    model = GradientBoostingRegressor(
        random_state=1337, **optimal_params[feature])
    model.fit(X, y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337)

    # Make predictions
    model_predictions = model.predict(X_test)

    # Calculate residuals
    res = pd.DataFrame({'Measured': y_test, 'Predicted': model_predictions})
    res['residuals'] = res['Measured'] - res['Predicted']

    # Create subplot
    plt.subplot(rows, cols, i + 1)
    sns.histplot(data=res, x='residuals', kde=True)
    plt.title(f'Distribution of Residuals for {feature}')
    plt.xlabel('Residuals')
    plt.ylabel('Density')

plt.tight_layout()
# plt.savefig('residuals_distribution.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 4 * rows))

for i, feature in enumerate(features):
    X = airqual_df_filtered.drop(feature, axis=1)
    y = airqual_df_filtered[feature]

    # Initialize and train model
    model = GradientBoostingRegressor(
        random_state=1337, **optimal_params[feature])
    model.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337)
    model_predictions = model.predict(X_test)

    # Calculate residuals
    res = pd.DataFrame({'Predicted': model_predictions, 'Measured': y_test})
    res['residuals'] = res['Measured'] - res['Predicted']

    # Plotting residuals vs. predicted values
    plt.subplot(rows, cols, i + 1)
    sns.scatterplot(x='Predicted', y='residuals', data=res)
    plt.axhline(y=0.0, color='r', linestyle='-', linewidth=4)
    plt.title(f'Residuals vs. Predicted {feature}')
    plt.xlabel(f'Predicted {feature}')
    plt.ylabel('Residuals')

plt.tight_layout()
# plt.savefig('residuals_vs_predicted.png', dpi=300)

plt.show()

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axs = axs.flatten()

for i, feature in enumerate(features):
    X = airqual_df_filtered.drop(feature, axis=1)
    y = airqual_df_filtered[feature]

    # Initialize and train model
    model = GradientBoostingRegressor(
        random_state=1337, **optimal_params[feature])
    model.fit(X, y)

    # Split the dataset for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337)
    model_predictions = model.predict(X_test)

    # Calculate residuals
    res = pd.DataFrame({'Predicted': model_predictions, 'Measured': y_test})
    res['residuals'] = res['Measured'] - res['Predicted']

    # Generate Q-Q plot
    sm.qqplot(res['residuals'], line='45', fit=True, ax=axs[i])
    axs[i].set_title(f'Q-Q Plot of Residuals for {feature}')
    axs[i].set_xlabel('Theoretical Quantiles')
    axs[i].set_ylabel('Sample Quantiles')

if n % cols != 0:
    axs[-1].set_visible(False)

plt.tight_layout()
# plt.savefig('qq_plots.png', dpi=300)
plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=1337)

y_predict = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
             'PT08.S4(NO2)', 'PT08.S5(O3)']

cv_results = {}

for feature in y_predict:
    X = airqual_df_filtered.drop(feature, axis=1)
    y = airqual_df_filtered[feature]

    model = GradientBoostingRegressor(random_state=1337,
                                      **optimal_params[feature])

    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    cv_results[feature] = (scores.mean(), scores.std() * 2)

for feature, (mean_score, confidence) in cv_results.items():
    print(f"{feature}: Mean R2 score {mean_score:.3f} with a 95% confidence interval of +/- {confidence:.3f}")

features = list(cv_results.keys())
mean_scores = [cv_results[feature][0] for feature in features]
confidence_intervals = [cv_results[feature][1] for feature in features]

# Calculate the positions of the bars
x_pos = np.arange(len(features))

# Create the bar plot
plt.figure(figsize=(9, 6))
bar_width = 0.35
bars = plt.bar(x_pos, mean_scores, yerr=confidence_intervals, capsize=5, color='skyblue', edgecolor='black')

# Add the feature names as labels
plt.xticks(x_pos, features, rotation=45, ha="right")

plt.xlabel('Feature')
plt.ylabel('Mean R² Score')
plt.title('Cross-Validation Results')
plt.tight_layout()

# Add a grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
# plt.savefig('cross_validation.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 4 * rows))

for i, feature in enumerate(features):
    X = airqual_df_filtered.drop(feature, axis=1)
    y = airqual_df_filtered[feature]

    model = GradientBoostingRegressor(random_state=1337,
                                      **optimal_params[feature])
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    labels = X.columns[indices]

    plt.subplot(rows, cols, i+1)
    plt.title(f"Feature Importances for {feature}")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), labels, rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

