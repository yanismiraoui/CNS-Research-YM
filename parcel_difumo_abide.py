import os
import json
import pickle
import sys
from tqdm import tqdm
import argparse
import math
from nilearn.maskers import NiftiMapsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
import nibabel as nb

def process_batch(batch_files, directory):
    store_matrix = {}

    def get_bold_tr(bold_path):
        json_sidecar_path = '{}.json'.format(
            bold_path.split('.nii.gz')[0]
        )
        if os.path.isfile(json_sidecar_path):
            with open(json_sidecar_path, 'r') as jfile:
                jdata=jfile.read()
            t_r = json.loads(jdata)['RepetitionTime']
        else:
            t_r = nb.load(bold_path).header.get_zooms()[-1]
        return round(float(t_r), 5)

    dataset = datasets.fetch_atlas_difumo(
                    dimension=1024,
                    resolution_mm=2
                )
    atlas_maps = dataset.maps

    for nifti_path in tqdm(batch_files):
        full_path = os.path.join(directory, nifti_path)
        t_r = get_bold_tr(full_path)
        masker = NiftiMapsMasker(
                maps_img=atlas_maps,
                smoothing_fwhm=3,
                standardize="zscore",
                detrend=True,
                memory="nilearn_cache",
                verbose=5,
                high_pass=0.008,
                t_r=t_r,
                )
        time_series = masker.fit_transform(full_path)
        correlation_measure = ConnectivityMeasure(
                kind="correlation",
                standardize="zscore_sample",
                )
        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
        store_matrix[full_path] = correlation_matrix
    return store_matrix

def save_results(data, batch_number, directory):
    output_filename = os.path.join(directory, f'abide_batch_{batch_number}_results.pkl')
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile)
    print(f"Results saved to {output_filename}")

def main(directory, batch_number):
    nifti_paths = [f for f in os.listdir(directory) if f.endswith(".nii.gz")]
    total_files = len(nifti_paths)
    batch_size = math.ceil(total_files / 4)
    
    start_index = (batch_number - 1) * batch_size
    end_index = start_index + batch_size
    batch_files = nifti_paths[start_index:end_index]

    print(f"Processing batch {batch_number} ({len(batch_files)} files)")
    results = process_batch(batch_files, directory)
    save_results(results, batch_number, "./difumo_results/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a specific batch of .nii.gz files.')
    parser.add_argument('directory', type=str, help='Directory containing .nii.gz files')
    parser.add_argument('batch_number', type=int, choices=range(1, 5), help='Batch number to process (1-4)')
    
    args = parser.parse_args()
    main(args.directory, args.batch_number)
