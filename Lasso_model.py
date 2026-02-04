
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.impute import SimpleImputer

def compute_mll(y_true, y_pred, sigma):
    """
    Compute Modified Laplace Log-likelihood (MLL) score for FVC prediction in liters.
    """
    sigma_clipped = np.maximum(sigma, 0.07)  # 70 mL = 0.07 L
    delta = np.minimum(np.abs(y_true - y_pred), 1.0)  # 1000 mL = 1.0 L
    mll_scores = - (np.sqrt(2) * delta / sigma_clipped) - np.log(np.sqrt(2) * sigma_clipped)
    return mll_scores, np.mean(mll_scores)

def estimate_sigma_from_cv(X, y, pipeline, rkf):
    """
    Estimate per-sample uncertainty (σ) using residuals from repeated CV.
    Returns: array of σ values aligned with y
    """
    n_samples = len(y)
    sigma_matrix = np.zeros((n_samples, rkf.get_n_splits()))

    for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(X)):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_val_pred = pipeline.predict(X.iloc[val_idx])
        residuals = np.abs(y.iloc[val_idx] - y_val_pred)
        sigma_matrix[val_idx, fold_idx] = residuals

    # Average residuals across folds per sample
    sigma_estimates = np.mean(sigma_matrix, axis=1)
    return sigma_estimates



def train_and_evaluate_lasso(X_train, X_test, y_train, y_test,
                             feature_list, numeric_features, onehot_features,
                             model_name="lasso_regression_pipeline.pkl"):

    print("Numeric features:", numeric_features)
    print("One-hot features:", onehot_features)
    print("number of features given as input:", len(numeric_features) + len(onehot_features))

    # ------------------------------
    # Preprocessor
    # ------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("onehot", "passthrough", onehot_features)
        ]
    )

    # Repeated 5-fold CV (15 folds total) for alpha selection
    rkf_alpha = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)


    # # Base Lasso model
    # lasso = Lasso(max_iter=10000, random_state=42)

    # # RandomizedSearchCV for alpha
    # random_search = RandomizedSearchCV(
    #     estimator=lasso,
    #     param_distributions={"alpha": loguniform(1e-4, 1)},  # search space
    #     n_iter=50,                  # number of random samples
    #     cv=rkf_alpha,               # 15-fold CV
    #     scoring="neg_mean_squared_error",
    #     random_state=42,
    #     n_jobs=-1
    # )

    # # Define grid of alpha values (log-spaced between 1e-4 and 1)
    # alpha_grid = {"alpha": np.logspace(-4, 0, 20)}
    # # GridSearchCV for alpha tuning
    # grid_search = GridSearchCV(
    #     estimator=lasso,
    #     param_grid=alpha_grid,
    #     cv=rkf_alpha,
    #     scoring="neg_mean_squared_error",
    #     n_jobs=-1
    # )

    # pipeline = Pipeline([
    #     ("scaler", preprocessor),
    #     ("lasso", grid_search)
    # ])



    pipeline = Pipeline([
        ("scaler", preprocessor),
        ("lasso", LassoCV(
            alphas=np.logspace(-3, 0, 30),  # grid for alpha
            cv=rkf_alpha,                   # 15-fold CV for alpha
            random_state=42,
            n_jobs=-1,
            max_iter=100000
        ))
    ])

    mae_scores, mse_scores, rmse_scores, r2_scores = [], [], [], []
    fold_ids = []

    for repeat_idx, (train_idx, val_idx) in enumerate(rkf_alpha.split(X_train), start=1):
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_val_pred = pipeline.predict(X_train.iloc[val_idx])
        y_val_true = y_train.iloc[val_idx]

        mae_scores.append(mean_absolute_error(y_val_true, y_val_pred))
        mse_scores.append(mean_squared_error(y_val_true, y_val_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
        r2_scores.append(r2_score(y_val_true, y_val_pred))
        fold_ids.append(repeat_idx)

    cv_results = pd.DataFrame({
        "RepeatFold": fold_ids,
        "MAE": mae_scores,
        "MSE": mse_scores,
        "RMSE": rmse_scores,
        "R2": r2_scores
    })

    print(f"MAE: {mae_scores}")
    print(f"MSE: {mse_scores}")
    # Convert each np.float64 to a normal float
    rmse = [float(x) for x in rmse_scores]
    print(f"RMSE: {rmse}")
    print(f"R2: {r2_scores}")
    print(cv_results)

    print("\nRepeated 5-Fold CV Results (15 runs):")
    print(cv_results.describe())

    # ------------------------------
    # Fit final model
    # ------------------------------
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    # best_alpha = pipeline.named_steps["lasso"].best_params_["alpha"]
    # print(f"\nBest alpha chosen by RandomizedSearchCV: {best_alpha:.6f}")

    # Best alpha from LassoCV
    best_alpha = pipeline.named_steps["lasso"].alpha_
    print(f"\nBest alpha chosen by LassoCV: {best_alpha:.6f}")

    # ------------------------------
    # Plot Training and Test R² vs alpha
    # ------------------------------

    alphas = np.logspace(-3, 1, 30)  # range of alpha values
    r2_train_scores = []
    r2_test_scores = []

    # Scale training and test data using the fitted preprocessor
    X_train_scaled = pipeline.named_steps["scaler"].transform(X_train)
    X_test_scaled = pipeline.named_steps["scaler"].transform(X_test)

    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=10000, random_state=42)
        lasso.fit(X_train_scaled, y_train)

        # R² on training set
        y_train_pred1 = lasso.predict(X_train_scaled)
        r2_train_scores.append(r2_score(y_train, y_train_pred1))

        # R² on test set
        y_test_pred1 = lasso.predict(X_test_scaled)
        r2_test_scores.append(r2_score(y_test, y_test_pred1))

    # Best alpha from your fitted pipeline
    best_alpha = pipeline.named_steps["lasso"].alpha_

    plt.figure(figsize=(8, 6))
    plt.plot(np.log10(alphas), r2_train_scores, marker="o", label="Training R²")
    plt.plot(np.log10(alphas), r2_test_scores, marker="s", label="Test R²")
    plt.axvline(np.log10(best_alpha), color="red", linestyle="--",
                label=f"Best alpha = {best_alpha:.4f}")
    plt.xlabel("log10(alpha)")
    plt.ylabel("R² Score")
    plt.title("LASSO Regression: Training vs Test R² across alpha")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    # ------------------------------
    # Compute MLL on test set
    # ------------------------------
    # Estimate σ from CV on training set
    sigma_train = estimate_sigma_from_cv(X_train, y_train, pipeline, rkf_alpha)

    # Use mean σ from training CV as proxy for test set
    sigma_test = np.full_like(y_test_pred, sigma_train.mean())

    # Compute MLL
    mll_scores, mean_mll = compute_mll(y_test.values, y_test_pred, sigma_test)
    print(f"- MLL (adaptive σ): {mean_mll:.4f}")


    print("\nTest set performance:")
    print(f"- MAE:  {test_mae:.2f} L")
    print(f"- MSE:  {test_mse:.2f} L²")
    print(f"- RMSE: {test_rmse:.2f} L")
    print(f"- R²:   {test_r2:.3f}")

    # Training set performance
    y_train_pred = pipeline.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    print("\nTraining set performance:")
    print(f"- MAE:  {train_mae:.2f} L")
    print(f"- MSE:  {train_mse:.2f} L²")
    print(f"- RMSE: {train_rmse:.2f} L")
    print(f"- R²:   {train_r2:.3f}")

    # ------------------------------
    # Inspect coefficients
    # ------------------------------
    # coefs = pipeline.named_steps["lasso"].best_estimator_.coef_

    coefs = pipeline.named_steps["lasso"].coef_   #for LASSOCV
    feature_names = pipeline.named_steps["scaler"].get_feature_names_out()
    # Strip transformer prefixes
    clean_feature_names = [name.split("__")[-1] for name in feature_names]
    coef_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Coefficient": coefs,
        "AbsCoefficient": np.abs(coefs)
    }).sort_values(by="AbsCoefficient", ascending=False)

    print("\nModel coefficients (sorted by |coef|):")
    print(coef_df[["Feature", "Coefficient"]])

    # Selected features (non-zero coefficients)
    selected_features = coef_df[coef_df["Coefficient"] != 0]["Feature"].tolist()
    print(f"\nSelected features (non-zero): {selected_features}")
    print(f"\nSelected features (non-zero): {len(selected_features)}")
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=coef_df.sort_values(by="AbsCoefficient", ascending=True),
        x="Coefficient", y="Feature", palette="viridis"
    )
    plt.title("LASSO Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Save model
    # ------------------------------
    joblib.dump(pipeline, model_name)

    # ------------------------------
    # Predicted vs Actual Plot
    # ------------------------------
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color="red", linestyle="--", label="Perfect Prediction")
    plt.xlabel("Actual Values (L)")
    plt.ylabel("Predicted Values (L)")
    plt.title("Predicted vs Actual (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Residual Plot
    # ------------------------------
    residuals = y_test - y_test_pred
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values (L)")
    plt.ylabel("Residuals (L)")
    plt.title("Residual Plot (Test Set)")
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Worst 5 patients (largest errors)
    # ------------------------------
    errors = np.abs(y_test - y_test_pred)
    error_df = pd.DataFrame({
        "Baseline_FVC": X_test["Baseline FVC Volume L"].values if "Baseline FVC Volume L" in X_test.columns else np.nan,
        "Target_FVC": y_test.values,
        "Predicted_FVC": y_test_pred,
        "Abs_Error": errors,
        "Baseline_FVC_Week": X_test["Baseline FVC Week"].values if "Baseline FVC Week" in X_test.columns else np.nan,
        "Followup_FVC_Week": X_test["Followup FVC Week"].values if "Followup FVC Week" in X_test.columns else np.nan,
    })

    worst5 = error_df.sort_values(by="Abs_Error", ascending=False).head(5)
    print("\nTop 5 patients with largest errors:")
    print(worst5)

    for idx, row in worst5.iterrows():
        patient_df = pd.DataFrame({
            "Week": [row["Baseline_FVC_Week"], row["Followup_FVC_Week"]],
            "Actual_FVC": [row["Baseline_FVC"], row["Target_FVC"]],
            "Predicted_FVC": [row["Baseline_FVC"], row["Predicted_FVC"]]
        })
        plot_df = patient_df.melt(id_vars="Week",
                                  value_vars=["Actual_FVC", "Predicted_FVC"],
                                  var_name="Type", value_name="FVC")
        plt.figure(figsize=(7, 5))
        sns.lineplot(data=plot_df, x="Week", y="FVC", hue="Type", marker="o")
        plt.title(f"Patient {idx}: Actual vs Predicted FVC (Worst Error)")
        plt.xlabel("Week")
        plt.ylabel("FVC (Liters)")
        plt.tight_layout()
        plt.show()

    # ------------------------------
    # Best 5 patients (increasing FVC + lowest error)
    # ------------------------------
    increasing = error_df["Target_FVC"] > error_df["Baseline_FVC"]
    best_patients = error_df[increasing].sort_values(by="Abs_Error", ascending=True).head(5)

    print("\nTop 5 patients with increasing FVC and lowest error:")
    print(best_patients)

    for idx, row in best_patients.iterrows():
        patient_df = pd.DataFrame({
            "Week": [row["Baseline_FVC_Week"], row["Followup_FVC_Week"]],
            "Actual_FVC": [row["Baseline_FVC"], row["Target_FVC"]],
            "Predicted_FVC": [row["Baseline_FVC"], row["Predicted_FVC"]]
        })

        plot_df = patient_df.melt(id_vars="Week",
                                value_vars=["Actual_FVC", "Predicted_FVC"],
                                var_name="Type", value_name="FVC")

        plt.figure(figsize=(7, 5))
        sns.lineplot(data=plot_df, x="Week", y="FVC", hue="Type", marker="o")
        plt.title(f"Patient {idx}: Actual vs Predicted FVC (Low Error, Increasing FVC)")
        plt.xlabel("Week")
        plt.ylabel("FVC (Liters)")
        plt.legend(title="Measurement")
        plt.tight_layout()
        plt.show()
        
    import shap

    # Extract model and preprocessor
    lasso_model = pipeline.named_steps["lasso"]
    preprocessor = pipeline.named_steps["scaler"]

    # Transform data
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create explainer
    explainer = shap.LinearExplainer(lasso_model, X_train_scaled, feature_perturbation="interventional")

    # First compute shap_values
    raw_shap_values = explainer(X_test_scaled)

    # Now wrap into Explanation object with feature names
    shap_values = shap.Explanation(values=raw_shap_values.values,
                                base_values=raw_shap_values.base_values,
                                data=X_test_scaled,
                                feature_names=feature_names)

    # Example: waterfall plot for one patient
    patient_index = 0
    shap.plots.waterfall(shap_values[patient_index])

    shap.summary_plot(shap_values, X_test_scaled, max_display=5)
    shap.plots.beeswarm(shap_values)

    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", max_display=5)


    return {
        "cv_results": cv_results,
        "test_results": {
            "MAE": test_mae,
            "MSE": test_mse,
            "RMSE": test_rmse,
            "R2": test_r2
        },
        "coef_df": coef_df,
        "worst5": worst5,
        "best_patients": best_patients
    }


# Clinical_Contineous_data = ['Baseline FVC Volume L', 'Age','FEV1 Volume L'
#         ]

# Clinical_Categorical_data = [
#         'Primary Diagnosis_CHP', 
#        'Primary Diagnosis_CTD-ILD',
#        'Primary Diagnosis_Emphysema', 'Primary Diagnosis_Exposure-related',
#        'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP',
#        'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP',
#        'Primary Diagnosis_Idiopathic pleuroparenchymal fibroelastosis (IPPFE)',
#        'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_NSIP',
#        'Primary Diagnosis_No information',
#        'Primary Diagnosis_Occupational-related ILD',
#        'Primary Diagnosis_Sarcoidosis',
#        'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)',
#        'Primary Diagnosis_UILD', 'Sex_Female', 'Sex_Male',
#        'Smoking History_Active Smoker', 'Smoking History_Ex Smoker',
#        'Smoking History_Never Smoker', 'Smoking History_No Knowledge']

# Clinical_Categorical_data = ['Sex_Male', 'Primary Diagnosis_CTD-ILD', 'Primary Diagnosis_Exposure-related', 'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP', 'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP', 'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_No information', 'Primary Diagnosis_Occupational-related ILD', 'Primary Diagnosis_Other', 'Primary Diagnosis_Sarcoidosis', 'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)', 'Primary Diagnosis_UILD', 'Smoking History_Ex Smoker', 'Smoking History_Never Smoker', 'Smoking History_No Knowledge']


# X_train, X_test, y_train, y_test = joblib.load("/home/pansurya/OSIC_thesis/radiomics_files/data_splits_with_clinical_harmonization.pkl")
# # Radiomics: automatically grab all columns with certain prefixes
# Radiomics_data = [col for col in X_train.columns 
#                   if col.startswith(("wavelet", "original", "log-sigma"))]
# all_features = Clinical_Contineous_data + Radiomics_data


# X_train_clini_log = X_train.copy()
# X_test_clini_log  = X_test.copy()

# # Transform Age
# for col in ['Age']:
#     X_train_clini_log[col] = np.log(X_train_clini_log[col])
#     X_test_clini_log[col] = np.log(X_test_clini_log[col])

# # Median imputation (only needed if missing values exist)
# imp_median = SimpleImputer(strategy='median')
# imp_median.fit(X_train_clini_log)

# X_train_median_filled = pd.DataFrame(
#     imp_median.transform(X_train_clini_log),
#     columns=X_train_clini_log.columns,
#     index=X_train_clini_log.index
# )

# X_test_median_filled = pd.DataFrame(
#     imp_median.transform(X_test_clini_log),
#     columns=X_test_clini_log.columns,
#     index=X_test_clini_log.index
# )

# results1 = train_and_evaluate_lasso(X_train_median_filled, X_test_median_filled, y_train, y_test, all_features, all_features, Clinical_Categorical_data, model_name="/home/pansurya/OSIC_thesis/LASSO_model/LASSO_With_All_Clinical_Radiomics_with_harmonizationpycombat.pkl")


# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import ElasticNetCV, ElasticNet
# from sklearn.model_selection import RepeatedKFold
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from sklearn.impute import SimpleImputer

# def train_and_evaluate_elasticnet(X_train, X_test, y_train, y_test,
#                                   feature_list, numeric_features, onehot_features,
#                                   model_name="elasticnet_regression_pipeline.pkl"):

#     print("Numeric features:", numeric_features)
#     print("One-hot features:", onehot_features)
#     print("number of features given as input:", len(numeric_features) + len(onehot_features))

#     # ------------------------------
#     # Preprocessor
#     # ------------------------------
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", StandardScaler(), numeric_features),
#             ("onehot", "passthrough", onehot_features)
#         ]
#     )

#     # Repeated 5-fold CV (15 folds total)
#     rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

#     # ------------------------------
#     # Elastic Net with CV
#     # ------------------------------
#     pipeline = Pipeline([
#         ("scaler", preprocessor),
#         ("elasticnet", ElasticNetCV(
#             l1_ratio = np.linspace(0.7, 1.0, 7),  # 0.70, 0.75, ..., 1.0
#             alphas=np.logspace(-3, 1, 30),        # covers weak → strong regularization
#             cv=rkf,
#             n_jobs=-1,
#             max_iter=100000,
#             random_state=42
#         ))
#     ])

#     # ------------------------------
#     # Cross-validation performance
#     # ------------------------------
#     mae_scores, mse_scores, rmse_scores, r2_scores, fold_ids = [], [], [], [], []

#     for repeat_idx, (train_idx, val_idx) in enumerate(rkf.split(X_train), start=1):
#         pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
#         y_val_pred = pipeline.predict(X_train.iloc[val_idx])
#         y_val_true = y_train.iloc[val_idx]

#         mae_scores.append(mean_absolute_error(y_val_true, y_val_pred))
#         mse_scores.append(mean_squared_error(y_val_true, y_val_pred))
#         rmse_scores.append(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
#         r2_scores.append(r2_score(y_val_true, y_val_pred))
#         fold_ids.append(repeat_idx)

#     cv_results = pd.DataFrame({
#         "RepeatFold": fold_ids,
#         "MAE": mae_scores,
#         "MSE": mse_scores,
#         "RMSE": rmse_scores,
#         "R2": r2_scores
#     })

#     print("\nRepeated 5-Fold CV Results (15 runs):")
#     print(cv_results.describe())

#     # ------------------------------
#     # Fit final model
#     # ------------------------------
#     pipeline.fit(X_train, y_train)
#     y_test_pred = pipeline.predict(X_test)

#     best_alpha = pipeline.named_steps["elasticnet"].alpha_
#     best_l1_ratio = pipeline.named_steps["elasticnet"].l1_ratio_
#     print(f"\nBest alpha chosen by ElasticNetCV: {best_alpha:.6f}")
#     print(f"Best l1_ratio chosen by ElasticNetCV: {best_l1_ratio:.2f}")

#     # ------------------------------
#     # Plot Training and Test R² vs alpha (for best l1_ratio)
#     # ------------------------------
#     alphas = np.logspace(-3, 1, 30)
#     r2_train_scores, r2_test_scores = [], []

#     # Scale training and test data using fitted preprocessor
#     X_train_scaled = pipeline.named_steps["scaler"].transform(X_train)
#     X_test_scaled = pipeline.named_steps["scaler"].transform(X_test)

#     for a in alphas:
#         enet = ElasticNet(alpha=a, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42)
#         enet.fit(X_train_scaled, y_train)

#         r2_train_scores.append(r2_score(y_train, enet.predict(X_train_scaled)))
#         r2_test_scores.append(r2_score(y_test, enet.predict(X_test_scaled)))

#     plt.figure(figsize=(8, 6))
#     plt.plot(np.log10(alphas), r2_train_scores, marker="o", label="Training R²")
#     plt.plot(np.log10(alphas), r2_test_scores, marker="s", label="Test R²")
#     plt.axvline(np.log10(best_alpha), color="red", linestyle="--",
#                 label=f"Best alpha = {best_alpha:.4f}")
#     plt.xlabel("log10(alpha)")
#     plt.ylabel("R² Score")
#     plt.title(f"Elastic Net: Training vs Test R² (l1_ratio={best_l1_ratio:.2f})")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # ------------------------------
#     # Evaluate on test set
#     # ------------------------------
#     test_mae = mean_absolute_error(y_test, y_test_pred)
#     test_mse = mean_squared_error(y_test, y_test_pred)
#     test_rmse = np.sqrt(test_mse)
#     test_r2 = r2_score(y_test, y_test_pred)

#     print("\nTest set performance:")
#     print(f"- MAE:  {test_mae:.2f} L")
#     print(f"- MSE:  {test_mse:.2f} L²")
#     print(f"- RMSE: {test_rmse:.2f} L")
#     print(f"- R²:   {test_r2:.3f}")

#     # Training set performance
#     y_train_pred = pipeline.predict(X_train)
#     train_mae = mean_absolute_error(y_train, y_train_pred)
#     train_mse = mean_squared_error(y_train, y_train_pred)
#     train_rmse = np.sqrt(train_mse)
#     train_r2 = r2_score(y_train, y_train_pred)

#     print("\nTraining set performance:")
#     print(f"- MAE:  {train_mae:.2f} L")
#     print(f"- MSE:  {train_mse:.2f} L²")
#     print(f"- RMSE: {train_rmse:.2f} L")
#     print(f"- R²:   {train_r2:.3f}")

#     # ------------------------------
#     # Inspect coefficients
#     # ------------------------------
#     coefs = pipeline.named_steps["elasticnet"].coef_
#     feature_names = pipeline.named_steps["scaler"].get_feature_names_out()
#     clean_feature_names = [name.split("__")[-1] for name in feature_names]

#     coef_df = pd.DataFrame({
#         "Feature": clean_feature_names,
#         "Coefficient": coefs,
#         "AbsCoefficient": np.abs(coefs)
#     }).sort_values(by="AbsCoefficient", ascending=False)

#     print("\nModel coefficients (sorted by |coef|):")
#     print(coef_df[["Feature", "Coefficient"]])

#     # Selected features (non-zero coefficients)
#     selected_features = coef_df[coef_df["Coefficient"] != 0]["Feature"].tolist()
#     print(f"\nSelected features (non-zero): {selected_features}")
#     print(f"\nSelected features (non-zero): {len(selected_features)}")

#     plt.figure(figsize=(8, 6))
#     sns.barplot(
#         data=coef_df.sort_values(by="AbsCoefficient", ascending=True),
#         x="Coefficient", y="Feature", palette="viridis"
#     )
#     plt.title("Elastic Net Regression Coefficients")
#     plt.xlabel("Coefficient")
#     plt.ylabel("Feature")
#     plt.tight_layout()
#     plt.show()

#     # ------------------------------
#     # Predicted vs Actual Plot
#     # ------------------------------
#     plt.figure(figsize=(7, 6))
#     sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7)
#     plt.plot([y_test.min(), y_test.max()],
#              [y_test.min(), y_test.max()],
#              color="red", linestyle="--", label="Perfect Prediction")
#     plt.xlabel("Actual Values (L)")
#     plt.ylabel("Predicted Values (L)")
#     plt.title("Predicted vs Actual (Test Set)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # ------------------------------
#     # Residual Plot
#     # ------------------------------
#     residuals = y_test - y_test_pred
#     plt.figure(figsize=(7, 6))
#     sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.7)
#     plt.axhline(0, color="red", linestyle="--")
#     plt.xlabel("Predicted Values (L)")
#     plt.ylabel("Residuals (L)")
#     plt.title("Residual Plot (Test Set)")
#     plt.tight_layout()
#     plt.show()

#         # ------------------------------
#     # Worst 5 patients (largest errors)
#     # ------------------------------
#     errors = np.abs(y_test - y_test_pred)
#     error_df = pd.DataFrame({
#         "Baseline_FVC": X_test["Baseline FVC Volume L"].values if "Baseline FVC Volume L" in X_test.columns else np.nan,
#         "Target_FVC": y_test.values,
#         "Predicted_FVC": y_test_pred,
#         "Abs_Error": errors,
#         "Baseline_FVC_Week": X_test["Baseline FVC Week"].values if "Baseline FVC Week" in X_test.columns else np.nan,
#         "Followup_FVC_Week": X_test["Followup FVC Week"].values if "Followup FVC Week" in X_test.columns else np.nan,
#     })

#     worst5 = error_df.sort_values(by="Abs_Error", ascending=False).head(5)
#     print("\nTop 5 patients with largest errors:")
#     print(worst5)

#     for idx, row in worst5.iterrows():
#         patient_df = pd.DataFrame({
#             "Week": [row["Baseline_FVC_Week"], row["Followup_FVC_Week"]],
#             "Actual_FVC": [row["Baseline_FVC"], row["Target_FVC"]],
#             "Predicted_FVC": [row["Baseline_FVC"], row["Predicted_FVC"]]
#         })
#         plot_df = patient_df.melt(id_vars="Week",
#                                   value_vars=["Actual_FVC", "Predicted_FVC"],
#                                   var_name="Type", value_name="FVC")
#         plt.figure(figsize=(7, 5))
#         sns.lineplot(data=plot_df, x="Week", y="FVC", hue="Type", marker="o")
#         plt.title(f"Patient {idx}: Actual vs Predicted FVC (Worst Error)")
#         plt.xlabel("Week")
#         plt.ylabel("FVC (Liters)")
#         plt.tight_layout()
#         plt.show()

#     # ------------------------------
#     # Best 5 patients (increasing FVC + lowest error)
#     # ------------------------------
#     increasing = error_df["Target_FVC"] > error_df["Baseline_FVC"]
#     best_patients = error_df[increasing].sort_values(by="Abs_Error", ascending=True).head(5)

#     print("\nTop 5 patients with increasing FVC and lowest error:")
#     print(best_patients)

#     for idx, row in best_patients.iterrows():
#         patient_df = pd.DataFrame({
#             "Week": [row["Baseline_FVC_Week"], row["Followup_FVC_Week"]],
#             "Actual_FVC": [row["Baseline_FVC"], row["Target_FVC"]],
#             "Predicted_FVC": [row["Baseline_FVC"], row["Predicted_FVC"]]
#         })

#         plot_df = patient_df.melt(id_vars="Week",
#                                 value_vars=["Actual_FVC", "Predicted_FVC"],
#                                 var_name="Type", value_name="FVC")

#         plt.figure(figsize=(7, 5))
#         sns.lineplot(data=plot_df, x="Week", y="FVC", hue="Type", marker="o")
#         plt.title(f"Patient {idx}: Actual vs Predicted FVC (Low Error, Increasing FVC)")
#         plt.xlabel("Week")
#         plt.ylabel("FVC (Liters)")
#         plt.legend(title="Measurement")
#         plt.tight_layout()
#         plt.show()

#     # ------------------------------
#     # Save model
#     # ------------------------------
#     joblib.dump(pipeline, model_name)

#     return {
#         "cv_results": cv_results,
#         "test_results": {
#             "MAE": test_mae,
#             "MSE": test_mse,
#             "RMSE": test_rmse,
#             "R2": test_r2
#         },
#         "coef_df": coef_df
#     }


# Clinical_Contineous_data = ['Baseline FVC Volume L', 'Age',
#        'FEV1 Volume L',]

# Clinical_Categorical_data = [
#        'Primary Diagnosis_CHP', 'Primary Diagnosis_CTD-ILD',
#        'Primary Diagnosis_Emphysema', 'Primary Diagnosis_Exposure-related',
#        'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP',
#        'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP',
#        'Primary Diagnosis_Idiopathic pleuroparenchymal fibroelastosis (IPPFE)',
#        'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_NSIP',
#        'Primary Diagnosis_No information',
#        'Primary Diagnosis_Occupational-related ILD',
#        'Primary Diagnosis_Sarcoidosis',
#        'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)',
#        'Primary Diagnosis_UILD', 'Sex_Female', 'Sex_Male',
#        'Smoking History_Active Smoker', 'Smoking History_Ex Smoker',
#        'Smoking History_Never Smoker', 'Smoking History_No Knowledge']

#Clinical_Categorical_data = ['Sex_Male', 'Primary Diagnosis_CTD-ILD', 'Primary Diagnosis_Exposure-related', 'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP', 'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP', 'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_No information', 'Primary Diagnosis_Occupational-related ILD', 'Primary Diagnosis_Other', 'Primary Diagnosis_Sarcoidosis', 'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)', 'Primary Diagnosis_UILD', 'Smoking History_Ex Smoker', 'Smoking History_Never Smoker', 'Smoking History_No Knowledge']

X_train, X_test, y_train, y_test = joblib.load("/home/pansurya/OSIC_thesis/radiomics_files/data_splits_with_clinical_harmonization.pkl")

# Radiomics: automatically grab all columns with certain prefixes
Radiomics_data = [col for col in X_train.columns 
                  if col.startswith(("wavelet", "original", "log-sigma"))]
all_features = Radiomics_data
Clinical_Categorical_data = []
results1 = train_and_evaluate_lasso(X_train, X_test, y_train, y_test, all_features, all_features, Clinical_Categorical_data, model_name="/home/pansurya/OSIC_thesis/LASSO_model/OnlyWithRadiomics_LASSO_With_HarmonizationPycombat.pkl")

# X_train, X_test, y_train, y_test = joblib.load("/home/pansurya/OSIC_thesis/radiomics_files/data_splits_with_clinical_harmonization.pkl")
# # Radiomics: automatically grab all columns with certain prefixes
# Radiomics_data = [col for col in X_train.columns 
#                   if col.startswith(("wavelet", "original", "log-sigma"))]
# all_features = Clinical_Contineous_data + Radiomics_data


# X_train_clini_log = X_train.copy()
# X_test_clini_log  = X_test.copy()

# # Transform Age
# for col in ['Age']:
#     X_train_clini_log[col] = np.log(X_train_clini_log[col])
#     X_test_clini_log[col] = np.log(X_test_clini_log[col])

# # Median imputation (only needed if missing values exist)
# imp_median = SimpleImputer(strategy='median')
# imp_median.fit(X_train_clini_log)

# X_train_median_filled = pd.DataFrame(
#     imp_median.transform(X_train_clini_log),
#     columns=X_train_clini_log.columns,
#     index=X_train_clini_log.index
# )

# X_test_median_filled = pd.DataFrame(
#     imp_median.transform(X_test_clini_log),
#     columns=X_test_clini_log.columns,
#     index=X_test_clini_log.index
# )

# results1 = train_and_evaluate_elasticnet(X_train_median_filled, X_test_median_filled, y_train, y_test, all_features, all_features, Clinical_Categorical_data, model_name="/home/pansurya/OSIC_thesis/EN_model/EN_With_All_Clinical_Radiomics_with_harmonizationPycombat.pkl")