import os
import sys

def ppmi_to_nifti(DATA_DIR):
    subject_folders = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
    for subject in subject_folders:
        scantype_folders = os.listdir(os.path.join(DATA_DIR, subject))
        for scan_type in scantype_folders:
            if scan_type.startswith('.'):
                continue
            scantype_subfolders = os.listdir(os.path.join(DATA_DIR, subject, scan_type))
            for scan_date in scantype_subfolders:
                scan_path = os.path.join(DATA_DIR, subject, scan_type, scan_date)
                if len([f for f in os.listdir(scan_path) if f.endswith('.nii')]) > 0:
                    print("SKIPPING SCAN (NIFTI ALREADY EXISTS): ", scan_path)
                    continue
                if os.path.isdir(scan_path):
                    print("CONVERTING SCAN: ", scan_path)
                    dcm2niix_command = "dcm2niix -f %p_%s -g y " + str(scan_path)
                    os.system(dcm2niix_command)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        DATA_DIR = args[0]
        ppmi_to_nifti(DATA_DIR)
    elif len(args) < 1:
        DATA_DIR = '../PPMI'
        print("No data directory provided. Using default directory: ", DATA_DIR)
        ppmi_to_nifti(DATA_DIR)
    else:
        print("Too many arguments provided.")