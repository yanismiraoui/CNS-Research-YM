import os

DATA_DIR = '../RAW_PPMI'

subject_folders = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
for subject in subject_folders:
    scantype_folders = os.listdir(os.path.join(DATA_DIR, subject))
    for scan_type in scantype_folders:
        if scan_type.startswith('.'):
            continue
        scantype_subfolders = os.listdir(os.path.join(DATA_DIR, subject, scan_type))
        for scan_date in scantype_subfolders:
            scan_path = os.path.join(DATA_DIR, subject, scan_type, scan_date)
            if os.path.isdir(scan_path):
                print("CONVERTING SCAN: ", scan_path)
                dcm2niix_command = f"dcm2niix -f %p_%s -g y {scan_path}"
                os.system(dcm2niix_command)