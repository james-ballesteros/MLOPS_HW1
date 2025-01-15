def model_assess(X_train, X_test, y_train, y_test, model, title="Default"):
    start_time = time.time()  # Start time

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    end_time = time.time()  # End time
    runtime = end_time - start_time  # Calculate runtime

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results = pd.DataFrame([title, train_mse, train_r2, test_mse, test_r2, runtime]).transpose()
    results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2', 'Runtime (s)']
    return y_train_pred, y_test_pred, results

def multi_model_assess(df, models, y_predict):
    all_model_results = []  # This will contain all model results for each dependent variable
    all_X_test = []
    all_X_train = []
    all_y_test_p = []
    all_y_train_p = []
    all_y_train = []
    # First loop will define dependent/independent variables and split data into test/training sets
    n_vars = len(y_predict)
    pbar = tqdm(range(n_vars), desc="Variable Processed", position=0, leave=True)  # Add progress bar

    for dependent in y_predict:
        model_results = []  # Array with dataframes for a given dependent variable
        # Designate independent and dependent variables
        x = df.drop([dependent], axis=1)
        y = df[dependent]
        # Split data into test and training sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1337)

        # Populate the array of observed values for the dependent variable
        all_y_train.append(y_train)

        # Process each of the desired models
        for model, model_name in models:
            y_train_pred, y_test_pred, results = model_assess(X_train, X_test, y_train, y_test, model, title=model_name)

            model_results.append(results)
            all_X_test.append(X_test)
            all_X_train.append(X_train)
            all_y_test_p.append(y_test_pred)
            all_y_train_p.append(y_train_pred)

        all_model_results.append(model_results)
        pbar.update(1)
        pbar.refresh()

    pbar.close()
    return all_model_results, all_X_test, all_X_train, all_y_test_p, all_y_train_p, all_y_train

lr = LinearRegression()
la = Lasso(alpha=0.1)
ri = Ridge(alpha=0.1)
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=1337)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=1337)
kn = KNeighborsRegressor()
sv = SVR()

models = [(lr, 'Linear Regression'),
          (la, 'Lasso Regression'),
          (ri, 'Ridge Regression'),
          (rf, 'Random Forest'),
          (gb, 'Gradient Boosting'),
          (kn, 'K-Neighbors'),
          (sv, 'SVR')]

y_predict = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
             'PT08.S4(NO2)', 'PT08.S5(O3)']

all_model_results, _, _, all_y_test_p, all_y_train_p, all_y_train = multi_model_assess(airqual_df_filtered, models, y_predict)

