# import os, csv, logging, shutil, threading
# from collections import OrderedDict
# from multiprocessing import Pool, cpu_count
# import numpy as np
# import nibabel as nib
# import SimpleITK as sitk
# import radiomics
# from radiomics.featureextractor import RadiomicsFeatureExtractor
# import matplotlib.pyplot as plt

# threading.current_thread().name = "Main"

# # ============================================================
# # Paths
# # ============================================================
# ROOT = "/home/pansurya/OSIC_thesis"
# RESULTS_DIR = os.path.join(os.getcwd(), "radiomics_files")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# PARAMS = os.path.join(ROOT, RESULTS_DIR, "Pyradiomics_Params.yaml")
# LOG = os.path.join(ROOT, RESULTS_DIR, "Radiomicslog3ROIThreshold.txt")
# OUTPUTCSV = os.path.join(ROOT, RESULTS_DIR, "RadiomicsFeaturesresults3ROIThreshold_main.csv")

# TEMP_DIR = "_TEMP3ROIThreshold"
# REMOVE_TEMP_DIR = True
# NUM_OF_WORKERS = max(cpu_count() - 4, 1)

# # ============================================================
# # Logging
# # ============================================================
# rLogger = radiomics.logger
# logHandler = logging.FileHandler(filename=LOG, mode="a")
# logHandler.setLevel(logging.INFO)
# logHandler.setFormatter(logging.Formatter("%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s"))
# rLogger.addHandler(logHandler)

# # ============================================================
# # Patients to exclude
# # ============================================================
EXCLUDE_PATIENTS = {
    "1000986","1000641","1000658","1001215","1001138","1002744",
    "1001244","1001117","1001137","1000985",
    "1001198","1002407","1000635","1001029","1001115", '1001026', 
    '1001127', '1000688', '408817', '1000982', '1000638'
}

# # # ============================================================
# # # Helper: Get spacing in (Z,Y,X)
# # # ============================================================
# # def get_spacing_from_nifti(nii):
# #     zooms = nii.header.get_zooms()[:3]  # (x, y, z)
# #     return (zooms[2], zooms[1], zooms[0])

# # # ============================================================
# # # Helper: Resample volume (your function)
# # # ============================================================
# # def resample_volume(ct_array, original_spacing, new_spacing=(1, 1, 1), interpolator=None):

# #     if interpolator is None:
# #         interpolator = sitk.sitkBSpline

# #     image_itk = sitk.GetImageFromArray(ct_array)

# #     # SimpleITK expects spacing in (x, y, z)
# #     image_itk.SetSpacing(tuple(float(s) for s in original_spacing[::-1]))

# #     original_size = np.array(image_itk.GetSize(), dtype=np.int32)
# #     original_spacing_np = np.array(original_spacing, dtype=np.float64)
# #     new_spacing_np = np.array(new_spacing, dtype=np.float64)

# #     new_size = np.round(original_size * (original_spacing_np[::-1] / new_spacing_np[::-1])).astype(int)

# #     resampler = sitk.ResampleImageFilter()
# #     resampler.SetOutputSpacing(tuple(float(s) for s in new_spacing_np[::-1]))
# #     resampler.SetSize([int(s) for s in new_size])
# #     resampler.SetOutputDirection(image_itk.GetDirection())
# #     resampler.SetOutputOrigin(image_itk.GetOrigin())
# #     resampler.SetInterpolator(interpolator)

# #     resampled_itk = resampler.Execute(image_itk)
# #     resampled_array = sitk.GetArrayFromImage(resampled_itk)

# #     actual_spacing = resampled_itk.GetSpacing()[::-1]

# #     return resampled_array, actual_spacing


# # ============================================================
# # Build list of cases
# # ============================================================
# def collect_cases(patients_root):
#     cases = []
 
#     for patient in os.listdir(patients_root):
#         pdir = os.path.join(patients_root, patient)
#         if not os.path.isdir(pdir):
#             continue

#         for baseline in os.listdir(pdir):
#             if baseline.startswith("baseline_"):
#                 baseline_dir = os.path.join(pdir, baseline)
#                 if not os.path.isdir(baseline_dir):
#                     continue

#                 baseline_date = baseline.replace("baseline_", "")
#                 series_folders = [f for f in os.listdir(baseline_dir) if f.startswith("1.")]
#                 if not series_folders:
#                     continue
#                 series_uid = series_folders[0]
#                 # -----------------------------
#                 # Find NIfTI folder
#                 # -----------------------------
#                 nifty_dirs = [d for d in os.listdir(baseline_dir) if d.endswith("_NIfTY")]
#                 if not nifty_dirs:
#                     print(f"⚠️ No NIfTI folder in {baseline_dir}, skipping...")
#                     continue

#                 nifty_folder = os.path.join(baseline_dir, nifty_dirs[0])
#                 nifty_files = [f for f in os.listdir(nifty_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

#                 if not nifty_files:
#                     print(f"⚠️ NIfTI folder is empty: {nifty_folder}, skipping...")
#                     continue

#                 ct_path = os.path.join(nifty_folder, nifty_files[0])

#                 # -----------------------------
#                 # Find MASK folder
#                 # -----------------------------
#                 mask_dirs = [d for d in os.listdir(baseline_dir) if d.endswith("_3ROI")]
#                 if not mask_dirs:
#                     print(f"⚠️ No mask folder in {baseline_dir}, skipping...")
#                     continue

#                 mask_folder = os.path.join(baseline_dir, mask_dirs[0])
#                 mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

#                 if not mask_files:
#                     print(f"⚠️ Mask folder is empty: {mask_folder}, skipping...")
#                     continue

#                 mask_path = os.path.join(mask_folder, mask_files[0])

#                 print(mask_path)
#                 patient_id = os.path.basename(ct_path).split("_")[0]
#                 print(patient_id)
#                 if patient_id in EXCLUDE_PATIENTS:
#                     continue

#                 existing_csv = os.path.join(TEMP_DIR, f"features_{patient_id}.csv")
#                 if os.path.exists(existing_csv):
#                     logging.getLogger("radiomics.batch").info(
#                         f"Skipping patient {patient_id} (already processed)"
#                     )
#                     continue

#                 cases.append({
#                     "PatientID": patient_id,
#                     "BaselineDate": baseline_date,
#                     "SeriesUID": series_uid,
#                     "Image": ct_path,
#                     "Mask": mask_path
#                 })

#     return cases

# # ============================================================
# # Run extraction for one case
# # ============================================================
# def run(case):
#     ptLogger = logging.getLogger("radiomics.batch")
#     feature_vector = OrderedDict(case)

#     try:
#         threading.current_thread().name = case["PatientID"]
#         extractor = RadiomicsFeatureExtractor(PARAMS)

#         # # Disable PyRadiomics resampling
#         # extractor.settings["resampledPixelSpacing"] = None
#         # extractor.settings["interpolator"] = None

#         # Load CT + mask
#         # nii_ct = nib.load(case["Image"])
#         # nii_mask = nib.load(case["Mask"])

#     #     ct_array = nii_ct.get_fdata().astype(np.int16)
#     #     mask_array = nii_mask.get_fdata().astype(np.int16)

#     #     ct_spacing = get_spacing_from_nifti(nii_ct)
#     #     mask_spacing = get_spacing_from_nifti(nii_mask)

#     #     # Resample both to 1mm
#     #     new_spacing = (1,1,1)
#     #     resampled_ct, ct_new_spacing = resample_volume(ct_array, ct_spacing)
#     #     resampled_mask, mask_new_spacing = resample_volume(
#     #     mask_array,
#     #     mask_spacing,
#     #     new_spacing=new_spacing,
#     #     interpolator=sitk.sitkNearestNeighbor
#     # )
#     #     print(f"Resampled CT spacing: {ct_new_spacing}, Mask spacing: {mask_new_spacing}")
#     #     # Convert back to SimpleITK
#     #     ct_itk = sitk.GetImageFromArray(resampled_ct)
#     #     ct_itk.SetSpacing(ct_new_spacing[::-1])

#     #     mask_itk = sitk.GetImageFromArray(resampled_mask)
#     #     mask_itk.SetSpacing(mask_new_spacing[::-1])


#     #     print("\n--- PyRadiomics CT metadata ---")
#     #     print("Size (X,Y,Z):", ct_itk.GetSize())
#     #     print("Spacing (X,Y,Z):", ct_itk.GetSpacing())
#     #     print("Origin:", ct_itk.GetOrigin())
#     #     print("Direction:", ct_itk.GetDirection())

#     #     print("\n--- PyRadiomics MASK metadata ---")
#     #     print("Size (X,Y,Z):", mask_itk.GetSize())
#     #     print("Spacing (X,Y,Z):", mask_itk.GetSpacing())
#     #     print("Origin:", mask_itk.GetOrigin())
#     #     print("Direction:", mask_itk.GetDirection())


#         # # Extract features
#         # feats = extractor.execute(nii_ct, nii_mask, label=1)
#         # for k, v in feats.items():
#         #     if k.startswith(("original", "log", "wavelet")):
#         #         feature_vector[k] = v

#         # mask_labels = {'right_lung':1, 'left_lung':2}
#         mask_labels = {'Fibrosis':1}
#         for k_lbl, v_lbl in mask_labels.items():
#             feats = extractor.execute(case["Image"], case["Mask"], label=v_lbl)
#             for k, v in feats.items():
#                 if k.startswith(("original", "log", "wavelet")):
#                     feature_vector[f"{k}_{k_lbl}"] = v

#         os.makedirs(TEMP_DIR, exist_ok=True)
#         out_file = os.path.join(TEMP_DIR, f"features_{case['PatientID']}.csv")
#         with open(out_file, "w") as f:
#             writer = csv.DictWriter(f, fieldnames=list(feature_vector.keys()))
#             writer.writeheader()
#             writer.writerow(feature_vector)

#         ptLogger.info("Processed patient %s", case["PatientID"])

#     except Exception:
#         ptLogger.error("Feature extraction failed for %s", case["PatientID"], exc_info=True)

#     return feature_vector

# # ============================================================
# # Main
# # ============================================================
# if __name__ == "__main__":
#     logger = logging.getLogger("radiomics.batch")

#     patients_root = "/scratch/bds/OSIC/PATIENTS_DICOM_STRUCTURE_MAIN"
#     cases = collect_cases(patients_root)
#     logger.info("Found %d cases", len(cases))

#     pool = Pool(NUM_OF_WORKERS)
#     results = pool.map(run, cases)

#     if results:
#         with open(OUTPUTCSV, "w") as out:
#             writer = csv.DictWriter(out, fieldnames=list(results[0].keys()))
#             writer.writeheader()
#             writer.writerows(results)

#         logger.info("Saved results to %s", OUTPUTCSV)


# # ==============================================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")


def train_and_evaluate_elasticnet(
        X_train, X_test, y_train, y_test,
        feature_list, numeric_features, onehot_features,
        model_name="elasticnet_regression_pipeline.pkl"):

    print("Numeric features:", numeric_features)
    print("One-hot features:", onehot_features)
    print("Total features:", len(numeric_features) + len(onehot_features))

    # ---------------------------------------------------------
    # PREPROCESSOR
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("onehot", "passthrough", onehot_features)
        ]
    )

    # ---------------------------------------------------------
    # ELASTICNETCV (THIS DOES ALL CV AUTOMATICALLY)
    # ---------------------------------------------------------
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    elastic = ElasticNetCV(
        l1_ratio=np.linspace(0.1, 0.3, 10),
        alphas=np.logspace(-3.5, 1, 30),
        cv=rkf,
        n_jobs=-1,
        max_iter=300000,
        random_state=42
    )

    pipeline = Pipeline([
        ("scaler", preprocessor),
        ("elasticnet", elastic)
    ])

    # ---------------------------------------------------------
    # FIT FINAL MODEL (CV HAPPENS INSIDE ElasticNetCV)
    # ---------------------------------------------------------
    print("\nTraining ElasticNetCV (this includes all CV folds)...")
    pipeline.fit(X_train, y_train)

    best_alpha = pipeline.named_steps["elasticnet"].alpha_
    best_l1 = pipeline.named_steps["elasticnet"].l1_ratio_

    print(f"\nBest alpha: {best_alpha:.6f}")
    print(f"Best l1_ratio: {best_l1:.3f}")

    # ---------------------------------------------------------
    # TRAINING PERFORMANCE
    # ---------------------------------------------------------
    y_train_pred = pipeline.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    print("\nTraining performance:")
    print(f"MAE:  {train_mae:.3f}")
    print(f"MSE:  {train_mse:.3f}")
    print(f"RMSE: {train_rmse:.3f}")
    print(f"R²:   {train_r2:.3f}")

    # ---------------------------------------------------------
    # TEST PERFORMANCE
    # ---------------------------------------------------------
    y_test_pred = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTest performance:")
    print(f"MAE:  {test_mae:.3f}")
    print(f"MSE:  {test_mse:.3f}")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"R²:   {test_r2:.3f}")

    # ---------------------------------------------------------
    # PER-FOLD CV METRICS (FAST, NO DUPLICATE FITTING)
    # ---------------------------------------------------------
    cv_mae, cv_mse, cv_rmse, cv_r2 = [], [], [], []

    for train_idx, val_idx in rkf.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        X_tr_scaled = pipeline.named_steps["scaler"].transform(X_tr)
        X_val_scaled = pipeline.named_steps["scaler"].transform(X_val)

        enet = ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1,
            max_iter=10000,
            random_state=42
        )
        enet.fit(X_tr_scaled, y_tr)

        y_pred = enet.predict(X_val_scaled)

        cv_mae.append(mean_absolute_error(y_val, y_pred))
        cv_mse.append(mean_squared_error(y_val, y_pred))
        cv_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        cv_r2.append(r2_score(y_val, y_pred))

    print("\n=== Per-fold CV Metrics (15 folds) ===")
    print("MAE per fold:", cv_mae)
    print("MSE per fold:", cv_mse)
    print("RMSE per fold:", cv_rmse)
    print("R² per fold:", cv_r2)

    print("\n=== CV Mean ± Std ===")
    print(f"MAE:  {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")
    print(f"MSE:  {np.mean(cv_mse):.4f} ± {np.std(cv_mse):.4f}")
    print(f"RMSE: {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
    print(f"R²:   {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

    # ---------------------------------------------------------
    # CV PATH (MSE vs alpha)
    # ---------------------------------------------------------
    enet_cv = pipeline.named_steps["elasticnet"]
    alphas = enet_cv.alphas_
    mse_path = enet_cv.mse_path_

    if mse_path.ndim == 3:
        l1_list = np.linspace(0.1, 0.3, 10)
        best_idx = np.argmin(np.abs(l1_list - best_l1))
        mse_mean = mse_path[best_idx].mean(axis=1)
        mse_std = mse_path[best_idx].std(axis=1)
    else:
        mse_mean = mse_path.mean(axis=1)
        mse_std = mse_path.std(axis=1)

    # ---------------------------------------------------------
    # PLOT 1: MSE vs alpha
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        np.log10(alphas), mse_mean, yerr=mse_std,
        fmt="o-", color="blue", ecolor="gray", capsize=4,
        label="MSE ± std"
    )
    plt.axvline(np.log10(best_alpha), color="red", linestyle="--",
                label=f"Best alpha = {best_alpha:.4f}")
    plt.xlabel("log10(alpha)")
    plt.ylabel("Mean Squared Error")
    plt.title(f"ElasticNet: MSE vs alpha (l1_ratio={best_l1:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/MSE_vs_alpha_R.png", dpi=300, bbox_inches="tight")
    plt.close()

    # # ---------------------------------------------------------
    # # PLOT 2: Training vs Test R² vs alpha
    # # ---------------------------------------------------------
    # X_train_scaled = pipeline.named_steps["scaler"].transform(X_train)
    # X_test_scaled = pipeline.named_steps["scaler"].transform(X_test)

    # r2_train_scores = []
    # r2_test_scores = []

    # for a in alphas:
    #     enet = ElasticNet(alpha=a, l1_ratio=best_l1, max_iter=10000)
    #     enet.fit(X_train_scaled, y_train)
    #     r2_train_scores.append(r2_score(y_train, enet.predict(X_train_scaled)))
    #     r2_test_scores.append(r2_score(y_test, enet.predict(X_test_scaled)))

    # plt.figure(figsize=(8, 6))
    # plt.plot(np.log10(alphas), r2_train_scores, marker="o", label="Training R²")
    # plt.plot(np.log10(alphas), r2_test_scores, marker="s", label="Test R²")
    # plt.axvline(np.log10(best_alpha), color="red", linestyle="--",
    #             label=f"Best alpha = {best_alpha:.4f}")
    # plt.xlabel("log10(alpha)")
    # plt.ylabel("R² Score")
    # plt.title(f"ElasticNet: Training vs Test R² (l1_ratio={best_l1:.2f})")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

   


    # ---------------------------------------------------------
    # COEFFICIENTS
    # ---------------------------------------------------------
    coefs = enet_cv.coef_
    feature_names = pipeline.named_steps["scaler"].get_feature_names_out()
    clean_names = [f.split("__")[-1] for f in feature_names]

    coef_df = pd.DataFrame({
        "Feature": clean_names,
        "Coefficient": coefs,
        "AbsCoefficient": np.abs(coefs)
    }).sort_values("AbsCoefficient", ascending=False)

    print("\nTop coefficients:")
    print(coef_df.head(20))

     # Selected features (non-zero coefficients)
    selected_features = coef_df[coef_df["Coefficient"] != 0]["Feature"].tolist()
    print(f"\nSelected features (non-zero): {selected_features}")
    print(f"\nSelected features (non-zero): {len(selected_features)}")

    # ---------------------------------------------------------
    # PLOT 3: Coefficient barplot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 10))
    sns.barplot(
        data=coef_df.sort_values("AbsCoefficient", ascending=True),
        x="Coefficient", y="Feature", palette="viridis"
    )
    plt.title("ElasticNet Coefficients")
    plt.tight_layout()
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/Coefficient_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

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
    plt.savefig(f"/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/Residual_Plot_(Test_Set)_R.png", dpi=300, bbox_inches="tight")
    plt.close()

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
        plt.savefig(f"/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/worst5_{idx}_R.png", dpi=300, bbox_inches="tight")
        plt.close()

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
        plt.savefig(f"/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/best_patients_worst5_{idx}_R.png", dpi=300, bbox_inches="tight")
        plt.close()

    import shap

    # Extract model and preprocessor
    lasso_model = pipeline.named_steps["elasticnet"]
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
    shap.plots.waterfall(shap_values[patient_index], show=False)
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/shap_waterfall_patient_0_R.png", dpi=300, bbox_inches="tight")
    plt.close()


    shap.summary_plot(shap_values, X_test_scaled, max_display=5)
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/shap_beeswarm_R.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", max_display=5, show=False)
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/shap_summary_bar_R.png", dpi=300, bbox_inches="tight")
    plt.close()


    # ------------------------------
    # Save model
    # ------------------------------
    joblib.dump(pipeline, model_name)

    return {
       
        "test_results": {
            "MAE": test_mae,
            "MSE": test_mse,
            "RMSE": test_rmse,
            "R2": test_r2
        },
        "coef_df": coef_df
    }


# Clinical_Contineous_data = ['Baseline FVC Volume L',  'FEV1 Volume L', 'Age', 'left_lung_volume_ml', 'right_lung_volume_ml', 'total_lung_volume_ml', 'ggo_left_ml', 'ggo_right_ml', 'ggo_total_ml', 'ggo_percent_total', 'fib_left_ml', 'fib_right_ml', 'fib_total_ml', 'fib_percent_total'
#        ]

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

#Clinical_Categorical_data = ['Sex_Male', 'Primary Diagnosis_CTD-ILD', 'Primary Diagnosis_Exposure-related', 'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP', 'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP', 'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_No information', 'Primary Diagnosis_Occupational-related ILD', 'Primary Diagnosis_Sarcoidosis', 'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)', 'Primary Diagnosis_UILD', 'Smoking History_Ex Smoker', 'Smoking History_Never Smoker', 'Smoking History_No Knowledge']
Clinical_Categorical_data = []

X_train, X_test, y_train, y_test = joblib.load("/home/pansurya/OSIC_thesis/radiomics_files/data_splits_clinical_without_harmonization.pkl")
# Radiomics: automatically grab all columns with certain prefixes
Radiomics_data = [col for col in X_train.columns 
                  if col.startswith(("wavelet", "original", "log-sigma"))]
# all_features = Clinical_Contineous_data + Radiomics_data
all_features = Radiomics_data

X_train_clini_log = X_train.copy()
X_test_clini_log  = X_test.copy()

# Transform Age
for col in ['Age']:
    X_train_clini_log[col] = np.log(X_train_clini_log[col])
    X_test_clini_log[col] = np.log(X_test_clini_log[col])

# Median imputation (only needed if missing values exist)
imp_median = SimpleImputer(strategy='median')
imp_median.fit(X_train_clini_log)

X_train_median_filled = pd.DataFrame(
    imp_median.transform(X_train_clini_log),
    columns=X_train_clini_log.columns,
    index=X_train_clini_log.index
)

X_test_median_filled = pd.DataFrame(
    imp_median.transform(X_test_clini_log),
    columns=X_test_clini_log.columns,
    index=X_test_clini_log.index
)

results1 = train_and_evaluate_elasticnet(X_train_median_filled, X_test_median_filled, y_train, y_test, all_features, all_features, Clinical_Categorical_data, model_name="/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/EN_With_All_Radiomics_without_harmonization.pkl")

