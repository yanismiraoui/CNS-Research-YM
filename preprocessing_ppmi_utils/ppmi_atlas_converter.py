import numpy as np
import pandas as pd
import nilearn
import os
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

DATA_DIR = '../PPMI'

# get all the .nii in a folder and its subfolders
def get_nifti_files(path):
    nifti_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".nii"):
                nifti_files.append(os.path.join(root, file))
    return nifti_files

nifti_files = get_nifti_files(DATA_DIR)
# Sample only 10% of the files for testing purposes
# nifti_files = nifti_files[:int(len(nifti_files)/10)]


shape_saver = {}
for nifti_path_dcm2niix in nifti_files:
    try:
        ## CORTICAL ##
        dataset = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
        atlas_filename = dataset.maps
        labels = dataset.labels

        masker = NiftiLabelsMasker(
            labels_img=atlas_filename,
            standardize="zscore_sample",
            standardize_confounds="zscore_sample",
            memory="nilearn_cache",
            verbose=5,
            labels=labels,
            resampling_target="labels",
        )

        time_series = masker.fit_transform(nifti_path_dcm2niix)
        cortical_shape = time_series.shape

        ## SUBCORTICAL ##
        dataset_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", symmetric_split=False)
        atlas_filename_sub = dataset_sub.maps
        labels_sub = dataset_sub.labels

        masker_sub = NiftiLabelsMasker(
            labels_img=atlas_filename_sub,
            standardize="zscore_sample",
            standardize_confounds="zscore_sample",
            memory="nilearn_cache",
            verbose=5,
            labels=labels_sub,
        )

        time_series_sub = masker_sub.fit_transform(nifti_path_dcm2niix)
        subcortical_shape = time_series_sub.shape

        # Concatenate the two time series together
        time_series_concat = np.concatenate((time_series, time_series_sub), axis=1)
        concat_shape = time_series_concat.shape

        # Compute correlation matrix
        correlation_measure = ConnectivityMeasure(
            kind="correlation",
            standardize="zscore_sample",
        )
        correlation_matrix = correlation_measure.fit_transform([time_series_concat])[0]
        correlation_matrix_shape = correlation_matrix.shape
        # Save the shapes
        shape_saver[nifti_path_dcm2niix] = [cortical_shape, subcortical_shape, concat_shape, correlation_matrix_shape]
    except Exception as e:
        shape_saver[nifti_path_dcm2niix] = [None, None, None, None]
        print(e)

# Save the shapes to a csv
shape_saver_df = pd.DataFrame.from_dict(shape_saver, orient='index')
shape_saver_df.columns = ['cortical_shape', 'subcortical_shape', 'concat_shape']
shape_saver_df.to_csv('shape_saver.csv')
