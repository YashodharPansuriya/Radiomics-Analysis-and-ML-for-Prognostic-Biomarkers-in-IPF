# import os
# import re
# import shutil
# import pandas as pd

# src_root = os.path.abspath("/scratch/bds/OSIC/")
# dst_root = os.path.abspath("/scratch/bds/OSIC/PATIENTS_DICOM_STRUCTURE_MAIN")

# os.makedirs(dst_root, exist_ok=True)

# # --- Load Subject IDs from text file ---
# with open("/home/pansurya/OSIC_thesis/followup_patients_48_57.txt", "r") as f:
#     content = f.read().strip()

# # Split by comma and strip whitespace
# raw_subjects = [s.strip() for s in content.split(",") if s.strip()]

# # --- Normalize subject IDs ---
# def normalize_subject_id(s):
#     if not s or pd.isna(s):
#         return None
#     s = str(s).strip()
#     return s.split(".")[0] if re.fullmatch(r"\d+\.0", s) else s

# subjects = [normalize_subject_id(x) for x in raw_subjects]
# subjects = [s for s in subjects if s]  # drop None

# # --- Copy patient folders ---
# report = []

# for subj in subjects:
#     src_path = os.path.join(src_root, subj)
#     dst_path = os.path.join(dst_root, subj)

#     if not os.path.exists(src_path):
#         report.append({'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': 'NotFound'})
#         continue

#     if os.path.exists(dst_path):
#         report.append({'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': 'ExistsSkipped'})
#         continue

#     try:
#         shutil.copytree(src_path, dst_path)
#         print(f"Copied {src_path} → {dst_path}")
#         report.append({'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': 'Copied'})
#     except Exception as e:
#         print(f"Error copying {src_path}: {e}")
#         report.append({'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': f'Error: {e}'})

# # --- Build report DataFrame ---
# report_df = pd.DataFrame(report)
# report_df.to_csv(os.path.join(dst_root, "copy_report.csv"), index=False)



# import os
# import re
# import shutil
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed

# src_root = os.path.abspath("/scratch/bds/OSIC/")
# dst_root = os.path.abspath("/scratch/bds/OSIC/PATIENTS_DICOM_STRUCTURE_MAIN")

# os.makedirs(dst_root, exist_ok=True)

# # --- Load Subject IDs from text file ---
# with open("/home/pansurya/OSIC_thesis/followup_patients_48_57.txt", "r") as f:
#     content = f.read().strip()

# # Split by comma and strip whitespace
# raw_subjects = [s.strip() for s in content.split(",") if s.strip()]

# # --- Normalize subject IDs ---
# def normalize_subject_id(s):
#     if not s or pd.isna(s):
#         return None
#     s = str(s).strip()
#     return s.split(".")[0] if re.fullmatch(r"\d+\.0", s) else s

# subjects = [normalize_subject_id(x) for x in raw_subjects]
# subjects = [s for s in subjects if s]  # drop None

# # --- Worker function for copying ---
# def copy_subject(subj):
#     print(subj)
#     src_path = os.path.join(src_root, subj)
#     dst_path = os.path.join(dst_root, subj)

#     if not os.path.exists(src_path):
#         return {'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': 'NotFound'}

#     if os.path.exists(dst_path):
#         return {'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': 'ExistsSkipped'}

#     try:
#         shutil.copytree(src_path, dst_path)
#         return {'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': 'Copied'}
#     except Exception as e:
#         return {'SubjectID': subj, 'SrcPath': src_path, 'DstPath': dst_path, 'Status': f'Error: {e}'}

# # --- Run in parallel ---
# report = []
# max_workers = 8  # adjust based on your CPU / disk performance

# with ThreadPoolExecutor(max_workers=max_workers) as executor:
#     futures = {executor.submit(copy_subject, subj): subj for subj in subjects}
#     for future in as_completed(futures):
#         result = future.result()
#         subj = result['SubjectID']
#         print(f"{subj}: {result['Status']}")
#         report.append(result)

# # --- Build report DataFrame ---
# report_df = pd.DataFrame(report)
# report_df.to_csv(os.path.join(dst_root, "copy_report.csv"), index=False)
# report_df

#create the baseline and followup folders based on acquisition date in DICOM_info.csv files
# import os, shutil, pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed

# root = r"/scratch/bds/OSIC/"
# patients_root = os.path.join(root, "PATIENTS_DICOM_STRUCTURE_MAIN")
# if not os.path.isdir(patients_root):
#     raise FileNotFoundError(f"PATIENTS_DICOM_STRUCTURE not found at: {patients_root}")

# def win_long_path(path):
#     if os.name == 'nt':
#         path = os.path.abspath(path)
#         if not path.startswith('\\\\?\\'):
#             path = '\\\\?\\' + path
#     return path

# def _find_acq_date_col(cols):
#     cols_l = [c.lower() for c in cols]
#     for candidate in (
#         "acquisition date", "acq date", "acquisition_date", "acq_date",
#         "acquisitiondate", "acqdate", "date"
#     ):
#         for i, c in enumerate(cols_l):
#             if candidate in c:
#                 return cols[i]
#     for i, c in enumerate(cols_l):
#         if "acquis" in c and "date" in c:
#             return cols[i]
#     return None

# def process_patient(patient_name):
#     """Process one patient folder and return (report_rows, removed_count)."""
#     patient_path = os.path.join(patients_root, patient_name)
#     if not os.path.isdir(patient_path):
#         return [], 0

#     report_rows = []
#     removed_folders = 0

#     # find all DICOM_info.csv files
#     dicom_info_files = []
#     for root_dir, _, files in os.walk(patient_path):
#         for fn in files:
#             if fn.lower() == "dicom_info.csv" or fn.lower().endswith("dicom_info.csv"):
#                 dicom_info_files.append(os.path.join(root_dir, fn))

#     if not dicom_info_files:
#         report_rows.append({"PatientFolder": patient_name, "Status": "NoDICOMInfoFound", "Details": None})
#         return report_rows, removed_folders

#     rows = []
#     for csv_path in dicom_info_files:
#         try:
#             df_csv = pd.read_csv(csv_path, dtype=str)
#         except Exception as e:
#             report_rows.append({"PatientFolder": patient_name, "Status": "CSVReadError", "Details": f"{csv_path} | {e}"})
#             continue

#         acq_col = _find_acq_date_col(df_csv.columns)
#         if acq_col is None:
#             report_rows.append({"PatientFolder": patient_name, "Status": "NoAcqDateColumn", "Details": csv_path})
#             continue

#         for idx, val in df_csv[acq_col].astype(str).astype(object).items():
#             raw_date = val if pd.notna(val) else None
#             series_src = os.path.dirname(csv_path)
#             parsed = pd.to_datetime(raw_date, errors="coerce")
#             rows.append({"PatientFolder": patient_name, "CsvPath": csv_path,
#                          "SeriesSrc": series_src, "AcqDateRaw": raw_date, "AcqDate": parsed})

#     valid_rows = [r for r in rows if pd.notna(r["AcqDate"])]
#     if not valid_rows:
#         report_rows.append({"PatientFolder": patient_name, "Status": "NoValidAcqDates", "Details": len(rows)})
#         return report_rows, removed_folders

#     by_date = {}
#     for r in valid_rows:
#         date_key = r["AcqDate"].normalize()
#         by_date.setdefault(date_key, []).append(r)

#     sorted_dates = sorted(by_date.keys())
#     baseline_date = sorted_dates[0]

#     for date_key in sorted_dates:
#         tag = "baseline" if date_key == baseline_date else "followup"
#         date_str = pd.to_datetime(date_key).strftime("%Y-%m-%d")
#         dest_group_folder = os.path.join(patient_path, f"{tag}_{date_str}")
#         os.makedirs(dest_group_folder, exist_ok=True)

#         seen_srcs = set()
#         for r in by_date[date_key]:
#             src = r["SeriesSrc"]
#             if src in seen_srcs:
#                 continue
#             seen_srcs.add(src)
#             series_name = os.path.basename(src.rstrip(os.sep))
#             dst = os.path.join(dest_group_folder, series_name)

#             if os.path.exists(dst):
#                 status = "ExistsSkipped"
#             else:
#                 try:
#                     shutil.copytree(win_long_path(src), win_long_path(dst))
#                     status = "Copied"
#                 except Exception:
#                     try:
#                         os.makedirs(dst, exist_ok=True)
#                         for root_dir, _, files in os.walk(src):
#                             rel_path = os.path.relpath(root_dir, src)
#                             target_dir = os.path.join(dest_group_folder, rel_path)
#                             os.makedirs(target_dir, exist_ok=True)
#                             for file in files:
#                                 src_file = os.path.join(root_dir, file)
#                                 dst_file = os.path.join(target_dir, file)
#                                 shutil.copy2(win_long_path(src_file), win_long_path(dst_file))
#                         status = "CopiedManual"
#                     except Exception as e2:
#                         status = f"Error:{e2}"

#             if status in ("Copied", "CopiedManual"):
#                 try:
#                     shutil.rmtree(win_long_path(src))
#                     removed_folders += 1
#                 except Exception as e:
#                     print(f"Could not remove {src}: {e}")

#             report_rows.append({"PatientFolder": patient_name, "AcqDate": date_str,
#                                 "Tag": tag, "SrcSeries": src, "DstSeries": dst, "Status": status})

#     return report_rows, removed_folders

# # --- Run in parallel ---
# all_reports = []
# total_removed = 0
# patient_names = sorted(os.listdir(patients_root))

# with ThreadPoolExecutor(max_workers=8) as executor:  # adjust workers
#     futures = {executor.submit(process_patient, name): name for name in patient_names}
#     for fut in as_completed(futures):
#         rows, removed = fut.result()
#         all_reports.extend(rows)
#         total_removed += removed

# report_df = pd.DataFrame(all_reports)
# print("Finished. Summary counts:")
# print(report_df['Status'].value_counts(dropna=False))
# print(f"Removed old folders: {total_removed}")

# from IPython.display import display
# display(report_df.head(200))


# # #remove intermediate "series instance UID" folder in baseline_* and followup_* directories
# import os
# import shutil
# from concurrent.futures import ThreadPoolExecutor, as_completed

# flattened_count = 0
# errors = []
# # root = r"/home/pansurya/OSIC_thesis/"
# # patients_root = os.path.join(root, "PATIENTS_DICOM_STRUCTURE_MAIN")
# # if not os.path.isdir(patients_root):
# #     raise FileNotFoundError(f"PATIENTS_DICOM_STRUCTURE not found at: {patients_root}")

# def process_patient(patient_name):
#     """Process one patient folder and return (flattened_count, errors)."""
#     patient_path = os.path.join(patients_root, patient_name)
#     if not os.path.isdir(patient_path):
#         return 0, []

#     local_flattened = 0
#     local_errors = []

#     # Look for baseline_* and followup_* folders
#     for visit_name in os.listdir(patient_path):
#         visit_path = os.path.join(patient_path, visit_name)
#         if not os.path.isdir(visit_path):
#             continue
#         if not (visit_name.lower().startswith("baseline") or visit_name.lower().startswith("followup")):
#             continue

#         # If visit folder contains exactly one subfolder (the "X" folder)
#         subfolders = [f for f in os.listdir(visit_path) if os.path.isdir(os.path.join(visit_path, f))]
#         if len(subfolders) == 1:
#             x_folder = os.path.join(visit_path, subfolders[0])
#             try:
#                 # Move all contents of X into visit_path
#                 for item in os.listdir(x_folder):
#                     src_item = os.path.join(x_folder, item)
#                     dst_item = os.path.join(visit_path, item)

#                     # If destination exists, merge or skip
#                     if os.path.exists(dst_item):
#                         if os.path.isdir(src_item) and os.path.isdir(dst_item):
#                             for root_dir, _, files in os.walk(src_item):
#                                 rel_path = os.path.relpath(root_dir, src_item)
#                                 target_dir = os.path.join(dst_item, rel_path)
#                                 os.makedirs(target_dir, exist_ok=True)
#                                 for file in files:
#                                     shutil.move(os.path.join(root_dir, file),
#                                                 os.path.join(target_dir, file))
#                         else:
#                             print(f"Skipping existing: {dst_item}")
#                     else:
#                         shutil.move(src_item, dst_item)

#                 # Remove the now-empty X folder
#                 shutil.rmtree(x_folder)
#                 local_flattened += 1
#                 print(f"Flattened: {x_folder}")
#             except Exception as e:
#                 local_errors.append((x_folder, str(e)))

#     return local_flattened, local_errors


# # --- Run in parallel ---
# patient_names = sorted(os.listdir(patients_root))
# with ThreadPoolExecutor(max_workers=8) as executor:  # adjust workers
#     futures = {executor.submit(process_patient, name): name for name in patient_names}
#     for fut in as_completed(futures):
#         fcount, errs = fut.result()
#         flattened_count += fcount
#         errors.extend(errs)

# print(f"\nFlattened {flattened_count} intermediate folders.")
# if errors:
#      print("Errors:", errors)

# #remove intermediate AXIAL, LOCALIZER, dicom etc. folders in baseline_* and followup_* directories
# def flatten_single_subfolder(path):
#     """
#     If a folder contains exactly one subfolder, move its contents up and remove it.
#     Repeat until no single-subfolder chain remains.
#     """
#     while True:
#         items = os.listdir(path)
#         subfolders = [f for f in items if os.path.isdir(os.path.join(path, f))]
#         files = [f for f in items if os.path.isfile(os.path.join(path, f))]

#         # Stop if there are files or more than one subfolder
#         if files or len(subfolders) != 1:
#             break

#         only_sub = os.path.join(path, subfolders[0])
#         # Move everything from the only subfolder up
#         for item in os.listdir(only_sub):
#             shutil.move(os.path.join(only_sub, item), os.path.join(path, item))
#         os.rmdir(only_sub)  # remove the now-empty folder

# def flatten_all(root):
#     for dirpath, dirnames, filenames in os.walk(root, topdown=False):
#         flatten_single_subfolder(dirpath)

# # Run flattening
# flatten_all(patients_root)
# print("Flattening complete.")


# # create only single baseline for patients who does not have followup DICOM data
# import os
# import shutil
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # --- Configure root paths ---
# root = r"/scratch/bds/OSIC/"
# patients_root = os.path.join(root, "PATIENTS_DICOM_STRUCTURE_MAIN")
# if not os.path.isdir(patients_root):
#     raise FileNotFoundError(f"PATIENTS_DICOM_STRUCTURE not found at: {patients_root}")

# # --- Helpers ---
# def _find_acq_date_col(cols):
#     cols_l = [c.lower() for c in cols]
#     for candidate in (
#         "acquisition date", "acq date", "acquisition_date", "acq_date",
#         "acquisitiondate", "acqdate", "date"
#     ):
#         for i, c in enumerate(cols_l):
#             if candidate in c:
#                 return cols[i]
#     for i, c in enumerate(cols_l):
#         if "acquis" in c and "date" in c:
#             return cols[i]
#     return None

# def _find_patient_csv(patient_path):
#     """
#     Prefer a DICOM_info.csv at the patient root. If not found, search recursively.
#     Returns path or None.
#     """
#     root_csv = os.path.join(patient_path, "DICOM_info.csv")
#     if os.path.isfile(root_csv):
#         return root_csv

#     for root_dir, _, files in os.walk(patient_path):
#         for fn in files:
#             if fn.lower() == "dicom_info.csv" or fn.lower().endswith("dicom_info.csv"):
#                 return os.path.join(root_dir, fn)
#     return None

# def _extract_baseline_date_from_csv(csv_path):
#     """
#     Read CSV and return the earliest valid acquisition date (normalized) as YYYY-MM-DD string.
#     Returns None if no valid dates.
#     """
#     try:
#         df = pd.read_csv(csv_path, dtype=str)
#     except Exception:
#         return None

#     acq_col = _find_acq_date_col(df.columns)
#     if acq_col is None:
#         return None

#     dates = pd.to_datetime(df[acq_col].astype(str), errors="coerce").dropna()
#     if dates.empty:
#         return None

#     # Use earliest date (baseline)
#     baseline_date = dates.min().normalize()
#     return pd.to_datetime(baseline_date).strftime("%Y-%m-%d")

# def process_patient(patient_name):
#     """
#     For one patient:
#     - If a baseline_* folder already exists under patient root, skip.
#     - Else, read earliest acquisition date from DICOM_info.csv.
#     - Create baseline_<YYYY-MM-DD> under patient root.
#     - Move series folders and the CSV into that baseline folder.
#     Returns list of report rows.
#     """
#     patient_path = os.path.join(patients_root, patient_name)
#     if not os.path.isdir(patient_path):
#         return [{"PatientFolder": patient_name, "Status": "NotADirectory"}]

#     # Detect an existing baseline folder at patient root and skip
#     child_dirs = [d for d in os.listdir(patient_path)
#                   if os.path.isdir(os.path.join(patient_path, d))]
#     has_baseline = any(d.lower().startswith("baseline") for d in child_dirs)
#     if has_baseline:
#         return [{"PatientFolder": patient_name, "Status": "AlreadyHasBaseline"}]

#     # Find CSV and baseline date
#     csv_path = _find_patient_csv(patient_path)
#     if not csv_path:
#         return [{"PatientFolder": patient_name, "Status": "NoCSVFound"}]

#     baseline_date_str = _extract_baseline_date_from_csv(csv_path)
#     if not baseline_date_str:
#         return [{"PatientFolder": patient_name, "Status": "NoValidDatesInCSV", "CSV": csv_path}]

#     baseline_folder_name = f"baseline_{baseline_date_str}"
#     baseline_path = os.path.join(patient_path, baseline_folder_name)

#     # If the target baseline_* already exists (rare), skip to avoid nesting
#     if os.path.exists(baseline_path):
#         return [{"PatientFolder": patient_name, "Status": "BaselineFolderAlreadyExists", "BaselinePath": baseline_path}]

#     # Create baseline folder
#     os.makedirs(baseline_path, exist_ok=True)

#     # Move CSV into baseline folder (keep filename)
#     try:
#         # If CSV is already inside some series subfolder, keep its name when moving
#         dst_csv = os.path.join(baseline_path, os.path.basename(csv_path))
#         if os.path.abspath(csv_path) != os.path.abspath(dst_csv):
#             shutil.move(csv_path, dst_csv)
#     except Exception as e:
#         # Non-fatal: report but continue
#         csv_move_status = f"CSVMoveError:{e}"
#     else:
#         csv_move_status = "CSVMoved"

#     # Move all series folders directly under patient root into baseline,
#     # but do NOT move baseline/followup folders (if any) or the baseline we just created.
#     moved_series = 0
#     skipped = []
#     errors = []

#     for item in sorted(os.listdir(patient_path)):
#         src_item = os.path.join(patient_path, item)

#         # Skip files (we only move directories that are series)
#         if not os.path.isdir(src_item):
#             continue

#         # Skip our newly created baseline folder and any baseline/followup folders
#         lname = item.lower()
#         if lname.startswith("baseline") or lname.startswith("followup"):
#             skipped.append(item)
#             continue

#         # Move this series directory into baseline
#         dst_item = os.path.join(baseline_path, item)
#         try:
#             if os.path.exists(dst_item):
#                 # Merge content if destination exists (safety)
#                 for root_dir, dirs, files in os.walk(src_item):
#                     rel = os.path.relpath(root_dir, src_item)
#                     target_dir = os.path.join(dst_item, rel)
#                     os.makedirs(target_dir, exist_ok=True)
#                     for f in files:
#                         shutil.move(os.path.join(root_dir, f), os.path.join(target_dir, f))
#                 # Remove emptied source tree
#                 shutil.rmtree(src_item)
#             else:
#                 shutil.move(src_item, dst_item)
#             moved_series += 1
#         except Exception as e:
#             errors.append((item, str(e)))

#     return [{
#         "PatientFolder": patient_name,
#         "Status": "BaselineCreated" if moved_series > 0 else "BaselineCreatedNoSeriesMoved",
#         "BaselineFolder": baseline_folder_name,
#         "CSVMoveStatus": csv_move_status,
#         "MovedSeriesCount": moved_series,
#         "Skipped": ";".join(skipped) if skipped else None,
#         "Errors": ";".join(f"{a}:{b}" for a, b in errors) if errors else None
#     }]

# # --- Run across patients (parallel or sequential) ---
# all_reports = []
# patient_names = sorted(os.listdir(patients_root))

# # Choose your level of parallelism. For filesystem operations, 4–8 threads is usually safe.
# MAX_WORKERS = 6

# with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#     futures = {executor.submit(process_patient, name): name for name in patient_names}
#     for fut in as_completed(futures):
#         rows = fut.result()
#         all_reports.extend(rows)

# # --- Reporting ---
# report_df = pd.DataFrame(all_reports)
# print("Finished. Summary counts:")
# print(report_df["Status"].value_counts(dropna=False))
# print("\nSample of report rows:")
# print(report_df.head(20).to_string(index=False))

# # If you prefer a CSV report:
# out_csv = os.path.join(root, "baseline_creation_report.csv")
# report_df.to_csv(out_csv, index=False)
# print(f"\nReport written to: {out_csv}")


# #score and keep the best DICOM series in each baseline_* and followup_* folder, remove the others
# import os
# import shutil
# import pydicom
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ------------------ Scoring Function ------------------ #
# def calculate_series_score(series_path):
#     """Calculate a combined score for a DICOM series folder."""
#     dcm_files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
#     if not dcm_files:
#         return None

#     slice_positions = []
#     pixel_spacings = []
#     valid_files = 0
#     modality = None
#     slice_thickness = None
#     series_desc = None
#     recon_kernel = None

#     for f in dcm_files:
#         try:
#             ds = pydicom.dcmread(os.path.join(series_path, f), stop_before_pixels=True)
#             valid_files += 1

#             if modality is None:
#                 modality = getattr(ds, "Modality", None)
#             if slice_thickness is None:
#                 try:
#                     slice_thickness = float(getattr(ds, "SliceThickness", 0))
#                 except Exception:
#                     slice_thickness = None
#             if series_desc is None:
#                 series_desc = str(getattr(ds, "SeriesDescription", "") or "")
#             if recon_kernel is None:
#                 recon_kernel = str(getattr(ds, "ReconstructionKernel", "") or "")

#             if hasattr(ds, "ImagePositionPatient"):
#                 slice_positions.append(ds.ImagePositionPatient[2])
#             if hasattr(ds, "PixelSpacing"):
#                 try:
#                     pixel_spacings.append(float(ds.PixelSpacing[0]))
#                 except Exception:
#                     pass

#         except Exception:
#             pass

#     if valid_files == 0:
#         return None

#     # --- Numeric scoring ---
#     num_slices = valid_files
#     spacing_std = np.std(np.diff(sorted(slice_positions))) if len(slice_positions) > 1 else 0
#     avg_pixel_spacing = np.mean(pixel_spacings) if pixel_spacings else 1.0
#     score_numeric = num_slices * 1.0 - spacing_std * 10.0 - avg_pixel_spacing * 5.0

#     # --- Metadata heuristic ---
#     info = {
#         "modality": modality,
#         "files": dcm_files,
#         "slice_thickness": slice_thickness,
#         "series_desc": series_desc,
#         "recon_kernel": recon_kernel
#     }
#     s_meta = 0
#     if (info.get('modality') or '').upper() != 'CT':
#         s_meta -= 1000
#     s_meta += len(info['files'])
#     th = info.get('slice_thickness') or 0
#     if 0 < th <= 1.5:
#         s_meta += 50
#     elif 1.5 < th <= 3:
#         s_meta += 20
#     desc = (info.get('series_desc') or '').upper() + ' ' + (info.get('recon_kernel') or '').upper()
#     if any(k in desc for k in ('LUNG', 'CHEST', 'THORAX')):
#         s_meta += 30
#     if any(k in desc for k in ('LOCALIZER', 'SCOUT')):
#         s_meta -= 100

#     return round(score_numeric + s_meta, 2)

# # ------------------ Parallel Processing ------------------ #
# def process_visit(patient_id, visit):
#     """Score all series in a visit and return best one."""
#     visit_path = os.path.join(patients_root, patient_id, visit)
#     if not os.path.isdir(visit_path):
#         return None

#     series_folders = [
#         f for f in os.listdir(visit_path)
#         if os.path.isdir(os.path.join(visit_path, f))
#     ]
#     if not series_folders:
#         return None

#     # Score series in parallel
#     scores = []
#     with ThreadPoolExecutor(max_workers=20) as executor:  # adjust workers as needed
#         future_to_series = {
#             executor.submit(calculate_series_score, os.path.join(visit_path, sf)): sf
#             for sf in series_folders
#         }
#         for future in as_completed(future_to_series):
#             sf = future_to_series[future]
#             try:
#                 score = future.result()
#                 if score is not None:
#                     scores.append((sf, score))
#             except Exception as e:
#                 print(f"Error scoring {sf}: {e}")

#     if not scores:
#         return None

#     # Pick best series
#     best_series, best_score = max(scores, key=lambda x: x[1])

#     # Remove others
#     removed = 0
#     for sf, score in scores:
#         if sf != best_series:
#             try:
#                 shutil.rmtree(os.path.join(visit_path, sf))
#                 removed += 1
#             except Exception as e:
#                 print(f"Could not remove {sf}: {e}")

#     return {
#         "patient": patient_id,
#         "visit": visit,
#         "kept_series": best_series,
#         "score": best_score,
#         "removed": removed
#     }

# # ------------------ Main Loop ------------------ #
# results = []
# for patient_id in sorted(os.listdir(patients_root)):
#     patient_path = os.path.join(patients_root, patient_id)
#     if not os.path.isdir(patient_path):
#         continue

#     for visit in os.listdir(patient_path):
#         res = process_visit(patient_id, visit)
#         if res:
#             results.append(res)

# # ------------------ Summary ------------------ #
# print("\nSummary:")
# for r in results:
#     print(f"Patient: {r['patient']}, Visit: {r['visit']}, Kept: {r['kept_series']} (Score: {r['score']}), Removed: {r['removed']}")

# #process using the multithreding convert dicom files into nifty format
# import os
# from pathlib import Path
# import numpy as np
# import pydicom
# import nibabel as nib
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # -----------------------------
# # Load and sort DICOM slices
# # -----------------------------
# Function to load and sort DICOM slices robustly
# def load_scans(dcm_path):
#     # List all .dcm files in the directory
#     files = [os.path.join(dcm_path, f) for f in os.listdir(dcm_path) if f.lower().endswith('.dcm')]
    
#     # Read only DICOMs that contain pixel data (skip reports, RTSTRUCT, etc.)
#     slices = []
#     for f in files:
#         ds = pydicom.dcmread(f)
#         if hasattr(ds, "PixelData"):
#             slices.append(ds)

#     # Define a safe sort key
#     def sort_key(ds):
#         if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
#             return float(ds.ImagePositionPatient[2])
#         elif hasattr(ds, "SliceLocation"):
#             return float(ds.SliceLocation)
#         elif hasattr(ds, "InstanceNumber"):
#             return int(ds.InstanceNumber)
#         else:
#             return 0  # fallback if nothing is available

#     # Sort slices
#     slices.sort(key=sort_key)

#     return slices

# # -----------------------------
# # Convert pixel arrays to HU
# # -----------------------------
# def transform_to_hu(slices):
#     images = np.stack([file.pixel_array for file in slices])
#     images = images.astype(np.int16)

#     for n in range(len(slices)):
#         intercept = slices[n].RescaleIntercept
#         slope = slices[n].RescaleSlope

#         if slope != 1:
#             images[n] = slope * images[n].astype(np.float64)
#             images[n] = images[n].astype(np.int16)

#         images[n] += np.int16(intercept)
#     #Clip    
#     images = np.clip(images, a_min = -1000,a_max = 200)
#     return np.array(images, dtype=np.int16)

# # -----------------------------
# # Extract voxel spacing
# # -----------------------------
# def get_original_spacing(scans):
#     slice_thickness = float(scans[0].SliceThickness)
#     pixel_spacing = [float(sp) for sp in scans[0].PixelSpacing]
#     return (slice_thickness, pixel_spacing[0], pixel_spacing[1])

# # -----------------------------
# # Worker function: process one series
# # -----------------------------
# def process_series(series):
#     patient_id = series["patient_id"]
#     scan_date = series["scan_date"]
#     series_path = series["series_path"]

#     try:
#         scans = load_scans(series_path)
#         hu_scans = transform_to_hu(scans)

#         slice_thickness, spacing_y, spacing_x = get_original_spacing(scans)
#         affine = np.diag([spacing_x, spacing_y, slice_thickness, 1])

#         output_folder = series_path.parent / f"{patient_id}_{scan_date}_NIfTY"
#         output_folder.mkdir(parents=True, exist_ok=True)

#         output_path = output_folder / f"{patient_id}_{scan_date}.nii.gz"
#         nifti_img = nib.Nifti1Image(hu_scans.astype(np.int16), affine)
#         nib.save(nifti_img, str(output_path))

#         return f"✅ Saved NIfTI for {patient_id}, {scan_date} → {output_path}"
#     except Exception as e:
#         return f"❌ Error processing {patient_id}, {scan_date}: {e}"

# # -----------------------------
# # Main pipeline with parallelism
# # -----------------------------
# def convert_all_dicoms_to_nifti(root_dir, test_patient=None, max_workers=4):
#     root = Path(root_dir)
#     all_series = []

#     for patient in sorted(os.listdir(root)):
#         if test_patient and patient != test_patient:
#             continue
#         patient_path = root / patient
#         if not patient_path.is_dir():
#             continue

#         for date_folder in sorted(os.listdir(patient_path)):
#             date_path = patient_path / date_folder
#             if not date_path.is_dir():
#                 continue

#             for series_folder in sorted(os.listdir(date_path)):
#                 series_path = date_path / series_folder
#                 if not series_path.is_dir():
#                     continue

#                 dicoms = [f for f in series_path.iterdir() if f.suffix.lower() == ".dcm"]
#                 if dicoms:
#                     all_series.append({
#                         "patient_id": patient,
#                         "scan_date": date_folder,
#                         "series_path": series_path,
#                         "dicom_files": dicoms
#                     })

#     print(f"Found {len(all_series)} series")

#     # Parallel execution
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_series, series) for series in all_series]
#         for future in as_completed(futures):
#             print(future.result())

# # -----------------------------
# # Run
# # -----------------------------
# if __name__ == "__main__":
#     # Adjust max_workers to number of CPU cores you want to use
#     convert_all_dicoms_to_nifti(patients_root, test_patient=None, max_workers=8)

# #generate lung masks from nifty files using lungmask package
# import os
# from pathlib import Path
# import nibabel as nib
# import SimpleITK as sitk
# from lungmask import LMInferer

# # -----------------------------
# # Initialize lungmask inferer
# # -----------------------------
# inferer = LMInferer()

# # -----------------------------
# # Process all patients
# # -----------------------------
# def generate_masks_from_nifti(root_dir, test_patient=None):
#     root = Path(root_dir)

#     for patient in sorted(os.listdir(root)):
#         if test_patient and patient != test_patient:
#             continue
#         patient_path = root / patient
#         if not patient_path.is_dir():
#             continue

#         for date_folder in sorted(os.listdir(patient_path)):
#             date_path = patient_path / date_folder
#             if not date_path.is_dir():
#                 continue

#             # Check if mask folder already exists for this patient/date
#             mask_folder = date_path / f"{patient}_{date_folder}_Mask"
#             if mask_folder.exists():
#                 print(f"⏩ Skipping {patient}/{date_folder} (mask folder already exists)")
#                 continue

#             # Look for the NIfTI folder
#             for subfolder in os.listdir(date_path):
#                 if subfolder.endswith("_NIfTY"):
#                     nifti_folder = date_path / subfolder
#                     for file in os.listdir(nifti_folder):
#                         if file.endswith(".nii.gz"):
#                             nifti_path = nifti_folder / file
#                             print(f"\nProcessing {nifti_path}")

#                             # Load CT NIfTI
#                             nii = nib.load(str(nifti_path))
#                             ct_array = nii.get_fdata().astype("int16")

#                             # Convert to SimpleITK image (preserve spacing/origin/direction)
#                             sitk_ct = sitk.GetImageFromArray(ct_array)
#                             nifti_spacing = nii.header.get_zooms()[:3]
#                             sitk_ct.SetSpacing(tuple(map(float, (nifti_spacing[2], nifti_spacing[1], nifti_spacing[0]))))

#                             # Run lungmask
#                             segmentation = inferer.apply(sitk_ct)

#                             # Convert segmentation back to NIfTI
#                             mask_img = nib.Nifti1Image(segmentation.astype("int16"), nii.affine)

#                             # Create output folder
#                             mask_folder.mkdir(parents=True, exist_ok=True)

#                             # Save mask NIfTI
#                             output_path = mask_folder / f"{patient}_{date_folder}_mask.nii.gz"
#                             nib.save(mask_img, str(output_path))

#                             print(f"✅ Saved mask: {output_path}")

# # -----------------------------
# # Run
# # -----------------------------
# if __name__ == "__main__":
#     generate_masks_from_nifti("/scratch/bds/OSIC/PATIENTS_DICOM_STRUCTURE_MAIN", test_patient=None)

#--------------------------------------------------------------------------------

# import os, csv, logging, shutil, threading
# from collections import OrderedDict
# from datetime import datetime
# from multiprocessing import Pool, cpu_count

# import SimpleITK as sitk
# import radiomics
# from radiomics.featureextractor import RadiomicsFeatureExtractor

# threading.current_thread().name = "Main"

# # File variables
# ROOT = f"/home/pansurya/OSIC_thesis"

# # Define your results directory
# RESULTS_DIR = os.path.join(os.getcwd(), "radiomics_files")

# # Create it if it doesn't exist
# os.makedirs(RESULTS_DIR, exist_ok=True)

# print(f"Radiomic files will be stored in: {RESULTS_DIR}")
# PARAMS = os.path.join(ROOT, RESULTS_DIR, "Pyradiomics_Params.yaml")  # your YAML settings
# LOG = os.path.join(ROOT, RESULTS_DIR, "Radiomicslog.txt")
# OUTPUTCSV = os.path.join(ROOT, RESULTS_DIR, "RadiomicsFeaturesresults_main.csv")

# TEMP_DIR = "_TEMP"
# REMOVE_TEMP_DIR = True
# NUM_OF_WORKERS = max(cpu_count() - 1, 1)

# # Logging
# rLogger = radiomics.logger
# logHandler = logging.FileHandler(filename=LOG, mode="a")
# logHandler.setLevel(logging.INFO)
# logHandler.setFormatter(logging.Formatter("%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s"))
# rLogger.addHandler(logHandler)

# # ---------------------------
# # Build list of cases from folder structure
# # ---------------------------
# def collect_cases(patients_root):
#     cases = []
    
#     for patient in os.listdir(patients_root):
#         pdir = os.path.join(patients_root, patient)
#         if not os.path.isdir(pdir):
#             continue

#         # find baseline folder
#         for baseline in os.listdir(pdir):
#             if baseline.startswith("baseline_"):
#                 baseline_dir = os.path.join(pdir, baseline)
#                 if not os.path.isdir(baseline_dir):
#                     continue

#                 # parse date from folder name
#                 baseline_date = baseline.replace("baseline_", "")

#                 # find series UID folder
#                 series_folders = [f for f in os.listdir(baseline_dir) if f.startswith("1.")]
#                 if not series_folders:
#                     continue
#                 series_uid = series_folders[0]
                
#                 # find CT NIfTI and mask
#                 nifty_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_NIfTY")]
#                 mask_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_Mask")]
#                 if not nifty_dir or not mask_dir:
#                     continue
               
#                 ct_path = os.path.join(baseline_dir, nifty_dir[0], os.listdir(os.path.join(baseline_dir, nifty_dir[0]))[0])
#                 mask_path = os.path.join(baseline_dir, mask_dir[0], os.listdir(os.path.join(baseline_dir, mask_dir[0]))[0])

#                 # parse patient ID from mask filename (e.g. 865785_baseline_1969-10-06_mask.nii.gz)
#                 fname = os.path.basename(mask_path)
#                 patient_id = fname.split("_")[0]
                

#                 cases.append({
#                     "PatientID": patient_id,
#                     "BaselineDate": baseline_date,
#                     "SeriesUID": series_uid,
#                     "Image": ct_path,
#                     "Mask": mask_path
#                 })
#     return cases

# # ---------------------------
# # Run extraction for one case
# # ---------------------------
# def run(case):
#     ptLogger = logging.getLogger("radiomics.batch")
#     feature_vector = OrderedDict(case)

#     try:
#         threading.current_thread().name = case["PatientID"]
#         extractor = RadiomicsFeatureExtractor(PARAMS)

#         mask_labels = {'right_lung':1, 'left_lung':2}
#         for k_lbl, v_lbl in mask_labels.items():
#             feats = extractor.execute(case["Image"], case["Mask"], label=v_lbl)
#             for k, v in feats.items():
#                 if k.startswith(("original", "log", "wavelet")):
#                     feature_vector[f"{k}_{k_lbl}"] = v


#         # Save temporary file
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

# # ---------------------------
# # Main
# # ---------------------------
# if __name__ == "__main__":
#     logger = logging.getLogger("radiomics.batch")
#     sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
#     patients_path = "/scratch/bds/OSIC/"
#     patients_root = os.path.join(patients_path, "PATIENTS_DICOM_STRUCTURE_MAIN")  # adjust path
#     cases = collect_cases(patients_root)
#     logger.info("Found %d cases", len(cases))

#     pool = Pool(NUM_OF_WORKERS)
#     results = pool.map(run, cases)

#     # Write combined CSV
#     if results:
#         with open(OUTPUTCSV, "w") as out:
#             writer = csv.DictWriter(out, fieldnames=list(results[0].keys()))
#             writer.writeheader()
#             writer.writerows(results)
#         logger.info("Saved results to %s", OUTPUTCSV)

#         if REMOVE_TEMP_DIR:
#             shutil.rmtree(TEMP_DIR, ignore_errors=True)



# import torch
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Device:", torch.cuda.get_device_name(0))


#--------------------------------------------------------------------

# import os
# from pathlib import Path
# import nibabel as nib
# import SimpleITK as sitk
# from lungmask import LMInferer


# # -----------------------------
# # Initialize lungmask inferer
# # -----------------------------
# inferer = LMInferer(modelname="R231CovidWeb")

# # inferer = LMInferer()

# # -----------------------------
# # Process all patients
# # -----------------------------
# def generate_masks_from_nifti(root_dir, test_patient=None):
#     root = Path(root_dir)

#     for patient in os.listdir(root):
#         if test_patient and patient != test_patient:
#             continue

#         patient_path = root / patient
#         if not patient_path.is_dir():
#             continue

#         for date_folder in sorted(os.listdir(patient_path)):

#             # ✅ ✅ PROCESS ONLY BASELINE FOLDERS
#             if not date_folder.lower().startswith("baseline"):
#                 continue

#             date_path = patient_path / date_folder
#             if not date_path.is_dir():
#                 continue

#             # Look for the NIfTI folder
#             for subfolder in os.listdir(date_path):
#                 if subfolder.endswith("_NIfTY"):
#                     nifti_folder = date_path / subfolder

#                     for file in os.listdir(nifti_folder):
#                         if file.endswith(".nii.gz"):
#                             nifti_path = nifti_folder / file
#                             print(f"\nProcessing {nifti_path}")

#                             # Load CT NIfTI
#                             nii = nib.load(str(nifti_path))
#                             ct_array = nii.get_fdata().astype("int16")

#                             # Convert to SimpleITK image
#                             sitk_ct = sitk.GetImageFromArray(ct_array)

#                             # NIfTI spacing is (x, y, z). SimpleITK expects (z, y, x)
#                             nifti_spacing = nii.header.get_zooms()[:3]
#                             sitk_ct.SetSpacing(tuple(map(float, (nifti_spacing[2],
#                                                                  nifti_spacing[1],
#                                                                  nifti_spacing[0]))))

#                             # Run lungmask
#                             segmentation = inferer.apply(sitk_ct)

#                             # Convert segmentation back to NIfTI (whole lung mask)
#                             mask_img = nib.Nifti1Image((segmentation).astype("int16"), nii.affine)

#                             # Create output folder
#                             patient_id = patient
#                             scan_date = date_folder
#                             output_folder = date_path / f"{patient_id}_{scan_date}_R231CovidWeb2ROI"
#                             output_folder.mkdir(parents=True, exist_ok=True)

#                             # Save mask
#                             output_path = output_folder / f"{patient_id}_{scan_date}_R231CovidWeb2ROI.nii.gz"
#                             nib.save(mask_img, str(output_path))

#                             print(f"✅ Saved mask: {output_path}")

# # -----------------------------
# # Run
# # -----------------------------
# if __name__ == "__main__":
#     generate_masks_from_nifti("/scratch/bds/OSIC/PATIENTS_DICOM_STRUCTURE_MAIN", test_patient=None)


#---------------------------------------------------------------------


# #without resampling
# import os, csv, logging, shutil, threading
# from collections import OrderedDict
# from datetime import datetime
# from multiprocessing import Pool, cpu_count

# import SimpleITK as sitk
# import radiomics
# from radiomics.featureextractor import RadiomicsFeatureExtractor

# threading.current_thread().name = "Main"

# # File variables
# ROOT = f"/home/pansurya/OSIC_thesis"

# # Define your results directory
# RESULTS_DIR = os.path.join(os.getcwd(), "radiomics_files")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# print(f"Radiomic files will be stored in: {RESULTS_DIR}")
# PARAMS = os.path.join(ROOT, RESULTS_DIR, "Pyradiomics_Params.yaml")
# LOG = os.path.join(ROOT, RESULTS_DIR, "Radiomicslog.txt")
# OUTPUTCSV = os.path.join(ROOT, RESULTS_DIR, "RadiomicsFeaturesresultscovid_main.csv")

# TEMP_DIR = "_TEMP"
# REMOVE_TEMP_DIR = True
# NUM_OF_WORKERS = max(3, 1)

# # Logging
# rLogger = radiomics.logger
# logHandler = logging.FileHandler(filename=LOG, mode="a")
# logHandler.setLevel(logging.INFO)
# logHandler.setFormatter(logging.Formatter("%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s"))
# rLogger.addHandler(logHandler)

# # ---------------------------
# # Patients to exclude
# # ---------------------------
# EXCLUDE_PATIENTS = {
#     "1000986","1000641","1000658","1001215","1001138","1002744",
#     "1001244","1001117","1001137","1000985",
#     "1001198","1002407","1000635","1001029","1001115"
# }

# # ---------------------------
# # Build list of cases
# # ---------------------------
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

#                 nifty_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_NIfTY")]
#                 mask_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_R231CovidWeb")]
#                 if not nifty_dir or not mask_dir:
#                     continue

#                 ct_path = os.path.join(baseline_dir, nifty_dir[0], os.listdir(os.path.join(baseline_dir, nifty_dir[0]))[0])
#                 mask_path = os.path.join(baseline_dir, mask_dir[0], os.listdir(os.path.join(baseline_dir, mask_dir[0]))[0])

#                 patient_id = os.path.basename(mask_path).split("_")[0]

#                 # ✅ Skip excluded patients
#                 if patient_id in EXCLUDE_PATIENTS:
#                     continue

#                 # ✅ Skip patients already processed (CSV exists)
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


# # ---------------------------
# # Run extraction for one case
# # ---------------------------
# def run(case):
#     ptLogger = logging.getLogger("radiomics.batch")
#     feature_vector = OrderedDict(case)

#     try:
#         threading.current_thread().name = case["PatientID"]
#         extractor = RadiomicsFeatureExtractor(PARAMS)

#         mask_labels = {'lung':1}
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

# # ---------------------------
# # Main
# # ---------------------------
# if __name__ == "__main__":
#     logger = logging.getLogger("radiomics.batch")
#     sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

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

#---------------------------------------------------------------------
#with resampling
# import os, csv, logging, shutil, threading
# from collections import OrderedDict
# from multiprocessing import Pool
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

# PARAMS = os.path.join(ROOT, RESULTS_DIR, "50Resampling.yaml")
# LOG = os.path.join(ROOT, RESULTS_DIR, "Radiomicslog50resampling2ROIyaml.txt")
# OUTPUTCSV = os.path.join(ROOT, RESULTS_DIR, "RadiomicsFeaturesresultscovidResmpling50resampling2ROIyaml.csv")

# TEMP_DIR = "_TEMP50resampling2ROIyaml"
# REMOVE_TEMP_DIR = True
# NUM_OF_WORKERS = max(4, 1)

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
# EXCLUDE_PATIENTS = {
#     "1000986","1000641","1000658","1001215","1001138","1002744",
#     "1001244","1001117","1001137","1000985",
#     "1001198","1002407","1000635","1001029","1001115"
# }

# # ============================================================
# # Helper: Get spacing in (Z,Y,X)
# # ============================================================
# def get_spacing_from_nifti(nii):
#     zooms = nii.header.get_zooms()[:3]  # (x, y, z)
#     return (zooms[2], zooms[1], zooms[0])

# # ============================================================
# # Helper: Resample volume (your function)
# # ============================================================
# def resample_volume(ct_array, original_spacing, new_spacing=(1, 1, 1), interpolator=None):

#     if interpolator is None:
#         interpolator = sitk.sitkBSpline

#     image_itk = sitk.GetImageFromArray(ct_array)

#     # SimpleITK expects spacing in (x, y, z)
#     image_itk.SetSpacing(tuple(float(s) for s in original_spacing[::-1]))

#     original_size = np.array(image_itk.GetSize(), dtype=np.int32)
#     original_spacing_np = np.array(original_spacing, dtype=np.float64)
#     new_spacing_np = np.array(new_spacing, dtype=np.float64)

#     new_size = np.round(original_size * (original_spacing_np[::-1] / new_spacing_np[::-1])).astype(int)

#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputSpacing(tuple(float(s) for s in new_spacing_np[::-1]))
#     resampler.SetSize([int(s) for s in new_size])
#     resampler.SetOutputDirection(image_itk.GetDirection())
#     resampler.SetOutputOrigin(image_itk.GetOrigin())
#     resampler.SetInterpolator(interpolator)

#     resampled_itk = resampler.Execute(image_itk)
#     resampled_array = sitk.GetArrayFromImage(resampled_itk)

#     actual_spacing = resampled_itk.GetSpacing()[::-1]

#     return resampled_array, actual_spacing


# # ============================================================
# # Build list of cases
# # ============================================================
# def collect_cases(patients_root):
#     cases = []
#     p = ['666514', '1000699', '1000425', '1002407', '827561', '914798', '944599', '542329', '387742', '419414', '1000207', '347343', '490343', '1000528', '1006781', '704055', '311740', '1000769', '178448', '333209', '861452', '173533', '977088', '940374', '390713', '1002403', '682952', '938640', '1000414', '884561', '824870', '1001002', '307885', '1000932', '1000470', '1001005', '1000828', '930434', '1001225', '250726', '631158', '374153', '1000893', '724578', '1001377', '502077', '612298', '716523', '394765', '1002744']
#     for patient in p:
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

#                 nifty_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_NIfTY")]
#                 mask_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_R231CovidWeb")]
#                 if not nifty_dir or not mask_dir:
#                     continue

#                 ct_path = os.path.join(baseline_dir, nifty_dir[0], os.listdir(os.path.join(baseline_dir, nifty_dir[0]))[0])
#                 mask_path = os.path.join(baseline_dir, mask_dir[0], os.listdir(os.path.join(baseline_dir, mask_dir[0]))[0])

#                 patient_id = os.path.basename(mask_path).split("_")[0]

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
#         nii_ct = nib.load(case["Image"])
#         nii_mask = nib.load(case["Mask"])

#         ct_array = nii_ct.get_fdata().astype(np.int16)
#         mask_array = nii_mask.get_fdata().astype(np.int16)

#         ct_spacing = get_spacing_from_nifti(nii_ct)
#         mask_spacing = get_spacing_from_nifti(nii_mask)

#         # Resample both to 1mm
#         new_spacing = (1,1,1)
#         resampled_ct, ct_new_spacing = resample_volume(ct_array, ct_spacing)
#         resampled_mask, mask_new_spacing = resample_volume(
#         mask_array,
#         mask_spacing,
#         new_spacing=new_spacing,
#         interpolator=sitk.sitkNearestNeighbor
#     )
#         print(f"Resampled CT spacing: {ct_new_spacing}, Mask spacing: {mask_new_spacing}")
#         # Convert back to SimpleITK
#         ct_itk = sitk.GetImageFromArray(resampled_ct)
#         ct_itk.SetSpacing(ct_new_spacing[::-1])

#         mask_itk = sitk.GetImageFromArray(resampled_mask)
#         mask_itk.SetSpacing(mask_new_spacing[::-1])


#         print("\n--- PyRadiomics CT metadata ---")
#         print("Size (X,Y,Z):", ct_itk.GetSize())
#         print("Spacing (X,Y,Z):", ct_itk.GetSpacing())
#         print("Origin:", ct_itk.GetOrigin())
#         print("Direction:", ct_itk.GetDirection())

#         print("\n--- PyRadiomics MASK metadata ---")
#         print("Size (X,Y,Z):", mask_itk.GetSize())
#         print("Spacing (X,Y,Z):", mask_itk.GetSpacing())
#         print("Origin:", mask_itk.GetOrigin())
#         print("Direction:", mask_itk.GetDirection())


#         # Extract features
#         feats = extractor.execute(ct_itk, mask_itk, label=1)

#         for k, v in feats.items():
#             if k.startswith(("original", "log", "wavelet")):
#                 feature_vector[k] = v


#         # mask_labels = {'right_lung':1, 'left_lung':2}
#         # for k_lbl, v_lbl in mask_labels.items():
#         #     feats = extractor.execute(case["Image"], case["Mask"], label=v_lbl)
#         #     for k, v in feats.items():
#         #         if k.startswith(("original", "log", "wavelet")):
#         #             feature_vector[f"{k}_{k_lbl}"] = v

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


# # --------------------------------------------------------------------------
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
# LOG = os.path.join(ROOT, RESULTS_DIR, "Radiomicslog2ROICovid.txt")
# OUTPUTCSV = os.path.join(ROOT, RESULTS_DIR, "RadiomicsFeaturesresults2ROICovid_main.csv")

# TEMP_DIR = "_TEMP2ROICovid"
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
# EXCLUDE_PATIENTS = {
#     "1000986","1000641","1000658","1001215","1001138","1002744",
#     "1001244","1001117","1001137","1000985",
#     "1001198","1002407","1000635","1001029","1001115", '1001026', '1001127', '1000688', '408817', '1000982', '1000638'
# }

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
#                 nifty_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_NIfTY")]
#                 mask_dir = [d for d in os.listdir(baseline_dir) if d.endswith("_R231CovidWeb2ROI")]
#                 if not nifty_dir or not mask_dir:
#                     continue

#                 ct_path = os.path.join(baseline_dir, nifty_dir[0], os.listdir(os.path.join(baseline_dir, nifty_dir[0]))[0])
#                 mask_path = os.path.join(baseline_dir, mask_dir[0], os.listdir(os.path.join(baseline_dir, mask_dir[0]))[0])

#                 patient_id = os.path.basename(mask_path).split("_")[0]

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

#         mask_labels = {'right_lung':1, 'left_lung':2}
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

#=========================================================================================================================================


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
#             alphas=np.logspace(-3.5, -1, 30),        # covers weak → strong regularization
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
#     alphas = np.logspace(-3.5, -1, 30)
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

#     import shap

#     # Extract model and preprocessor
#     lasso_model = pipeline.named_steps["elasticnet"]
#     preprocessor = pipeline.named_steps["scaler"]

#     # Transform data
#     X_train_scaled = preprocessor.transform(X_train)
#     X_test_scaled = preprocessor.transform(X_test)

#     # Get feature names
#     feature_names = preprocessor.get_feature_names_out()

#     # Create explainer
#     explainer = shap.LinearExplainer(lasso_model, X_train_scaled, feature_perturbation="interventional")

#     # First compute shap_values
#     raw_shap_values = explainer(X_test_scaled)

#     # Now wrap into Explanation object with feature names
#     shap_values = shap.Explanation(values=raw_shap_values.values,
#                                 base_values=raw_shap_values.base_values,
#                                 data=X_test_scaled,
#                                 feature_names=feature_names)

#     # Example: waterfall plot for one patient
#     patient_index = 0
#     shap.plots.waterfall(shap_values[patient_index])

#     shap.summary_plot(shap_values, X_test_scaled, max_display=5)
#     shap.plots.beeswarm(shap_values)

#     shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", max_display=5)


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


# Clinical_Contineous_data = ['Baseline FVC Volume L',  'FEV1 Volume L', 'Age'
#        ]

# Clinical_Categorical_data = [
#        'Primary Diagnosis_CHP', 
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

# #Clinical_Categorical_data = ['Sex_Male', 'Primary Diagnosis_CTD-ILD', 'Primary Diagnosis_Exposure-related', 'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP', 'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP', 'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_No information', 'Primary Diagnosis_Occupational-related ILD', 'Primary Diagnosis_Sarcoidosis', 'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)', 'Primary Diagnosis_UILD', 'Smoking History_Ex Smoker', 'Smoking History_Never Smoker', 'Smoking History_No Knowledge']


# X_train, X_test, y_train, y_test = joblib.load("/home/pansurya/OSIC_thesis/radiomics_files/data_splits_clinical_without_harmonization2ROI.pkl")
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

# results1 = train_and_evaluate_elasticnet(X_train_median_filled, X_test_median_filled, y_train, y_test, all_features, all_features, Clinical_Categorical_data, model_name="/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/EN_With_All_Clinical_Radiomics_without_harmonization2ROI.pkl")

# # ============================================================

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
# import matplotlib
# matplotlib.use("Agg")


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
#             alphas=np.logspace(-3.5, 1, 30),        # covers weak → strong regularization
#             cv=rkf,
#             n_jobs=-1,
#             max_iter=300000,
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
#     enet_cv = pipeline.named_steps["elasticnet"]
#     alphas = enet_cv.alphas_
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
#     plt.savefig("/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/Training_vs_Test_R²_2ROICovid.png", dpi=300, bbox_inches="tight")
#     plt.close()


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
#         plt.savefig(f"/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/worst5_{idx}_2ROICovid.png", dpi=300, bbox_inches="tight")
#         plt.close()

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
#         plt.savefig(f"/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/best_patients_worst5_{idx}_2ROICovid.png", dpi=300, bbox_inches="tight")
#         plt.close()

#     import shap

#     # Extract model and preprocessor
#     lasso_model = pipeline.named_steps["elasticnet"]
#     preprocessor = pipeline.named_steps["scaler"]

#     # Transform data
#     X_train_scaled = preprocessor.transform(X_train)
#     X_test_scaled = preprocessor.transform(X_test)

#     # Get feature names
#     feature_names = preprocessor.get_feature_names_out()

#     # Create explainer
#     explainer = shap.LinearExplainer(lasso_model, X_train_scaled, feature_perturbation="interventional")

#     # First compute shap_values
#     raw_shap_values = explainer(X_test_scaled)

#     # Now wrap into Explanation object with feature names
#     shap_values = shap.Explanation(values=raw_shap_values.values,
#                                 base_values=raw_shap_values.base_values,
#                                 data=X_test_scaled,
#                                 feature_names=feature_names)

#     # Example: waterfall plot for one patient
#     patient_index = 0
#     shap.plots.waterfall(shap_values[patient_index], show=False)
#     plt.savefig("/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/shap_waterfall_patient_0_2ROICovid.png", dpi=300, bbox_inches="tight")
#     plt.close()


#     shap.summary_plot(shap_values, X_test_scaled, max_display=5)
#     shap.plots.beeswarm(shap_values, show=False)
#     plt.savefig("/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/shap_beeswarm_2ROICovid.png", dpi=300, bbox_inches="tight")
#     plt.close()

#     shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", max_display=5, show=False)
#     plt.savefig("/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/shap_summary_bar_2ROICovid.png", dpi=300, bbox_inches="tight")
#     plt.close()


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


# Clinical_Contineous_data = ['Baseline FVC Volume L',  'FEV1 Volume L', 'Age'
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

# #Clinical_Categorical_data = ['Sex_Male', 'Primary Diagnosis_CTD-ILD', 'Primary Diagnosis_Exposure-related', 'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP', 'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP', 'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_No information', 'Primary Diagnosis_Occupational-related ILD', 'Primary Diagnosis_Sarcoidosis', 'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)', 'Primary Diagnosis_UILD', 'Smoking History_Ex Smoker', 'Smoking History_Never Smoker', 'Smoking History_No Knowledge']


# X_train, X_test, y_train, y_test = joblib.load("/home/pansurya/OSIC_thesis/radiomics_files/data_splits_clinical_without_harmonization2ROI_covid.pkl")
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

# results1 = train_and_evaluate_elasticnet(X_train_median_filled, X_test_median_filled, y_train, y_test, all_features, all_features, Clinical_Categorical_data, model_name="/home/pansurya/OSIC_thesis/With_all radiomics_clincal_features_model/EN_With_All_Clinical_Radiomics_without_harmonization2ROI_Covid.pkl")

#=============================================================

# import nibabel as nib
# import numpy as np
# import os
# import glob
# from pathlib import Path # Helps with path manipulation
# from scipy.ndimage import binary_fill_holes 

# def create_combined_multilabel_mask(ct_path, lung_mask_path, output_path):
#     """
#     Generates a single multi-label NIfTI mask file containing 3 ROIs:
#     1: Healthy, 2: GGO, 3: Fibrosis
#     """
#     try:
#         ct_img = nib.load(ct_path)
#         mask_img = nib.load(lung_mask_path)
        
#         ct_data = ct_img.get_fdata()
#         lung_mask = mask_img.get_fdata()

#         # Initialize mask to 0s (background outside the lung)
#         combined_mask = np.zeros(ct_data.shape, dtype=np.uint8)
        
#         # NEW STEP: Set all voxels *inside* the lung mask to Label 4 first
#         # This captures all remaining tissue not defined by 1, 2, 3
#         combined_mask[lung_mask > 0] = 4 

#         # Now, overwrite the specific HU ranges with their respective labels
#         # We apply the strictest definitions first to ensure no overlap:
        
#         # Label 3: Fibrosis (-500 to -200 HU)
#         combined_mask[(ct_data >= -500) & (ct_data <= -200) & (combined_mask == 4)] = 3
        
#         # Label 2: GGO/Early Change (-700 to -501 HU)
#         combined_mask[(ct_data >= -700) & (ct_data <= -501) & (combined_mask == 4)] = 2
        
#         # Label 1: Healthy (-950 to -701 HU)
#         combined_mask[(ct_data >= -950) & (ct_data <= -701) & (combined_mask == 4)] = 1
        
#         # Note: Voxels with HU > -200 or < -950 remain as Label 4 ("Other")
        
#         # --- Post-Processing (Hole Filling on each label) ---
#         cleaned_mask = np.zeros(ct_data.shape, dtype=np.uint8)
#         for label_val in [1, 2, 3, 4]: # Iterate through all 4 labels
#             binary_roi = combined_mask == label_val
#             filled_roi = binary_fill_holes(binary_roi).astype(np.uint8)
#             cleaned_mask[filled_roi > 0] = label_val

#         # Save the multi-label NIfTI
#         new_mask = nib.Nifti1Image(cleaned_mask, ct_img.affine, ct_img.header)
#         nib.save(new_mask, output_path)
#         print(f"✅ Successfully saved mask to: {output_path}")

#     except Exception as e:
#         print(f"❌ Error processing {ct_path}: {e}")

# # --- Main Automation Logic ---

# # Define the root directory where all patient folders are located
# ROOT_DIR = '/scratch/bds/OSIC/PATIENTS_DICOM_STRUCTURE_MAIN'

# # Use glob to find all 'baseline_*' folders within any patient ID folder
# baseline_folders = glob.glob(os.path.join(ROOT_DIR, '*', 'baseline_*', ''))

# print(f"Found {len(baseline_folders)} baseline folders to process.")

# for base_folder_path in baseline_folders:
#     # 1. Identify the NIfTY and Mask subfolders using a pattern match
#     nifty_folder = glob.glob(os.path.join(base_folder_path, '*NIfTY*'), recursive=False)
#     mask_folder = glob.glob(os.path.join(base_folder_path, '*Mask*'), recursive=False)
    
#     if not nifty_folder or not mask_folder:
#         print(f"Skipping {base_folder_path}: Could not find both NIfTY and Mask folders.")
#         continue
    
#     nifty_folder_path = nifty_folder[0]
#     mask_folder_path = mask_folder[0]

#     # 2. Find the actual NIfTI files inside those folders
#     ct_file = glob.glob(os.path.join(nifty_folder_path, '*.nii.gz'))[0]
#     mask_file = glob.glob(os.path.join(mask_folder_path, '*.nii.gz'))[0]
    
#     # 3. Define the output path and file name
#     # We want a new folder named '*_3ROI' *inside* the 'baseline_*' folder
#     new_roi_folder_name = Path(base_folder_path).stem + '_3ROI'
    
#     # *** THIS LINE IS UPDATED ***
#     output_folder_path = os.path.join(base_folder_path, new_roi_folder_name)
    
#     # Ensure the output folder exists
#     os.makedirs(output_folder_path, exist_ok=True)
    
#     # Define the final output filename (matching the folder name as requested)
#     output_file_name = new_roi_folder_name + '.nii.gz'
#     final_output_path = os.path.join(output_folder_path, output_file_name)
    
#     # 4. Run the mask creation function
#     print(f"\nProcessing patient CT: {os.path.basename(ct_file)}")
#     create_combined_multilabel_mask(ct_file, mask_file, final_output_path)

# print("\nAutomation complete!")


#==============================================================

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

# PARAMS = os.path.join(ROOT, RESULTS_DIR, "50Resampling.yaml")
# LOG = os.path.join(ROOT, RESULTS_DIR, "RadiomicslogMaskyaml.txt")
# OUTPUTCSV = os.path.join(ROOT, RESULTS_DIR, "RadiomicsFeaturesresultsMaskyaml_main.csv")

# TEMP_DIR = "_TEMPMaskyaml"
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
# EXCLUDE_PATIENTS = {
#     "1000986","1000641","1000658","1001215","1001138","1002744",
#     "1001244","1001117","1001137","1000985",
#     "1001198","1002407","1000635","1001029","1001115", '1001026', '1001127', '1000688', '408817', '1000982', '1000638'
# }

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
#                 mask_dirs = [d for d in os.listdir(baseline_dir) if d.endswith("_Mask")]
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

#         mask_labels = {'right_lung':1, 'left_lung':2}
#         # mask_labels = {'Healthy':1, 'GGO':2, 'Fibrosis':3}
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
        l1_ratio=np.linspace(0.1, 1, 10),
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
        l1_list = np.linspace(0.1, 1, 10)
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
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/MSE_vs_alpha_RcH.png", dpi=300, bbox_inches="tight")
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
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/Coefficient_barplot_RcH.png", dpi=300, bbox_inches="tight")
    plt.close()

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
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/Predicted_vs_Actual_(Test Set)_RcH.png", dpi=300, bbox_inches="tight")

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
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/Only_with_ClinicalData/Residual_Plot_(Test Set)_RcH.png", dpi=300, bbox_inches="tight")

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
        plt.savefig(f"/home/pansurya/OSIC_thesis/EN_model/All_Clinical_RadiomicsData/worst5_{idx}_RCH.png", dpi=300, bbox_inches="tight")
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
        plt.savefig(f"/home/pansurya/OSIC_thesis/EN_model/All_Clinical_RadiomicsData/best_patients_worst5_{idx}_RCH.png", dpi=300, bbox_inches="tight")
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
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/All_Clinical_RadiomicsData/shap_waterfall_patient_0_RCH.png", dpi=300, bbox_inches="tight")
    plt.close()


    shap.summary_plot(shap_values, X_test_scaled, max_display=5)
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/All_Clinical_RadiomicsData/shap_beeswarm_RCH.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", max_display=5, show=False)
    plt.savefig("/home/pansurya/OSIC_thesis/EN_model/All_Clinical_RadiomicsData/shap_summary_bar_RCH.png", dpi=300, bbox_inches="tight")
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


Clinical_Contineous_data = ['Baseline FVC Volume L',  'FEV1 Volume L', 'Age'
       ]

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

Clinical_Categorical_data = ['Sex_Male', 'Primary Diagnosis_CTD-ILD', 'Primary Diagnosis_Exposure-related', 'Primary Diagnosis_Fibrotic HP (FHP)', 'Primary Diagnosis_INSIP', 'Primary Diagnosis_IPF', 'Primary Diagnosis_Idiopathic OP', 'Primary Diagnosis_Miscellaneous', 'Primary Diagnosis_No information', 'Primary Diagnosis_Occupational-related ILD', 'Primary Diagnosis_Sarcoidosis', 'Primary Diagnosis_Smoking Related ILD (DIP / RB / RB-ILD)', 'Primary Diagnosis_UILD', 'Smoking History_Ex Smoker', 'Smoking History_Never Smoker', 'Smoking History_No Knowledge']


X_train, X_test, y_train, y_test = joblib.load("/home/pansurya/OSIC_thesis/radiomics_files/data_splits_with_clinical_harmonization.pkl")
# Radiomics: automatically grab all columns with certain prefixes
Radiomics_data = [col for col in X_train.columns 
                  if col.startswith(("wavelet", "original", "log-sigma"))]
# all_features = Clinical_Contineous_data + Radiomics_data
all_features = Radiomics_data + Clinical_Contineous_data

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

results1 = train_and_evaluate_elasticnet(X_train_median_filled, X_test_median_filled, y_train, y_test, all_features, all_features, Clinical_Categorical_data, model_name="/home/pansurya/OSIC_thesis/EN_model/All_Clinical_RadiomicsData/EN_With_All_Clinical_Radiomics_with_harmonization_H.pkl")

