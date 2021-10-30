import os
import pandas as pd 
import numpy as np
from sampling import sample_tracks, sample_track_terminations, sample_objects, get_sample_hypervolumes, save_sample
from pathlib import Path
from nd2_dask.nd2_reader import nd2_reader
import zarr
import re


md_cols = ['file', 'px_microns', 'axes', 'cohort', 'treatment', 'scale', 
           'translate', 't', 'z', 'y', 'x', 'channel_0', 'channel_1', 'channel_2', 
           'channel_3', 'frame_rate', 'roi_t', 'roi_x', 'roi_y', 'roi_size', 
           'datetime', 'platelets_info_path', 'platelet_tracks']


# -----------------------------------------------
# Get random sample from files listed in metadata
# -----------------------------------------------

def get_sample_from_metadata(
    metadata,
    tracks_dir, 
    image_dir,
    save_dir=None,
    name='',
    sample_type='tracks', 
    n_each=100, 
    stratify_by=['cohort', 'treatment'], 
    scale_cols=('z', 'y', 'x'), 
    channel=2,
    tracks_file='platelet_tracks', 
    image_file='file',
    array_order=('t', 'z_pixels', 'y_pixels', 'x_pixels'), 
    sub_dirs=['cohort', 'treatment'], 
    shape=None,
    debug_without_tracks=False, 
    frames=30, 
    box=60, 
    id_col='particle', 
    time_col='t', 
    min_track_length=30,
    labels_dir=None, 
    batch_name=None,
    use_h5=False, 
    image_channel=2,
    repeat_file=False, 
    weights='log-frequency',
    #
    ):
    '''
    Obtain information about where to find a sample of
    track segments from 
    '''
    metadata = add_cohort_tx_pairs(metadata, stratify_by)
    stratify_by = metadata['stratify_by'].unique()
    samples_dict = {}
    for cond in stratify_by:
        if save_dir is not None:
            cond_bf = make_bash_friendly(cond)
            cond_dir = os.path.join(save_dir, name + '_' + cond_bf)
            os.makedirs(cond_dir, exist_ok=True)
            existing_samples = [smpl for smpl in os.listdir(cond_dir) if smpl.endswith('.smpl')]
            print(existing_samples)
        df = metadata[metadata['stratify_by'] == cond]
        df = df.reset_index()
        print(len(df))
        print(df.index.values)
        print(df.head())
        no_exp = len(df)
        no_each = np.floor_divide(n_each, no_exp)
        remainder = n_each % no_exp
        n_per_exp = [no_each, ] * no_exp
        n_per_exp[0] += remainder
        samples_list = []
        print(n_per_exp)
        for i, n in enumerate(n_per_exp):
            path = os.path.join(tracks_dir, df.loc[i, tracks_file])
            print(path)
            tracks = pd.read_csv(path)
            if sub_dirs is not None:
                to_join = [image_dir, ] + [df[sd][0] for sd in sub_dirs]
                image_nested_dir = os.path.join(*to_join)
                print(image_nested_dir)
            else:
                image_nested_dir = image_dir
            if use_h5:
                image_name = df.loc[i, image_file] + '.h5'
            else:
                image_name = df.loc[i, image_file] + '.nd2'
            image_path = os.path.join(image_nested_dir, image_name)
            print(image_path)
            if not debug_without_tracks:
                if shape is None and not use_h5:
                    layer_list = nd2_reader(image_path)
                    shape = layer_list[channel][0].shape
                    del layer_list
                elif use_h5:
                    from sampling import read_from_h5
                    print('reading the h5: ', image_path)
                    h5 = read_from_h5(image_path)
                    shape = h5.shape
                    print('h5 shape: ', shape)
                tracks_name = Path(path).stem
                scale = (1, ) + tuple([df.loc[i, ax] for ax in scale_cols])
                if labels_dir is not None and batch_name is not None:
                    labels_path = get_labels_path(tracks, i, sub_dirs, labels_dir, batch_name)
                else: 
                    labels_path = None
                kwargs = {
                    'frames' : frames, 
                    'box' : box, 
                    'id_col' : id_col, 
                    'time_col' : time_col, 
                    'scale' : scale, 
                    'array_order' : array_order, 
                    'non_tzyx_col' : None, 
                    'seed' : None, 
                    'weights' : weights, 
                    'max_lost_prop' : None, 
                    'min_track_length' : min_track_length, 
                    'labels_path' : labels_path
                }
                #print(kwargs)
                if sample_type == 'tracks':
                    if repeat_file:
                        sample = sample_tracks(path, image_path, shape, tracks_name, n, **kwargs)
                        sample = get_sample_hypervolumes(sample, img_channel=image_channel)
                        samples_list.append(sample)
                        if save_dir is not None:
                            save_sample(cond_dir, sample)
                    else:
                        print(tracks_name)
                        file_matches = [f for f in existing_samples if f.find(tracks_name) != -1]
                        print(file_matches)
                        if len(file_matches) > 0:
                            # we only want to get samples not yet done
                            p = re.compile(r'id-\d*_t-\d*')
                            for f in file_matches:
                                smpl_dir = os.path.join(cond_dir, f)
                                smpls = [f for f in os.listdir(smpl_dir) if len(p.findall(f)) > 0]
                                new_n = n - len(smpls) # how many to get
                                if new_n <= 0:
                                    sample = None
                                else:
                                    sample = sample_tracks(path, image_path, shape, tracks_name, new_n, **kwargs)
                                    sample = get_sample_hypervolumes(sample, img_channel=image_channel)
                                    if save_dir is not None:
                                            save_sample(cond_dir, sample)
                            samples_list.append(None)
                        else:
                            sample = sample_tracks(path, image_path, shape, tracks_name, n, **kwargs)
                            sample = get_sample_hypervolumes(sample, img_channel=image_channel)
                            if save_dir is not None:
                                save_sample(cond_dir, sample)
                    samples_list.append(sample)
                elif sample_type == 'terminations':
                    sample = sample_track_terminations(path, image_path, shape, tracks_name, n, **kwargs)
                    samples_list.append(sample)
                elif sample_type == 'objects':
                    # this block is very much at draft stage so may or may not work... 
                        # ... a very good guess is no
                    m = 'Please provide dir with labels data to argument: labels_dir'
                    assert labels_dir is not None, m
                    m = 'Please provide correct batch name to arument: batch_name'
                    assert batch_name is not None, m
                    labels_path = get_labels_path(tracks, i, sub_dirs, labels_dir, batch_name)
                    labels = zarr.open(labels)
                    sample = sample_objects(tracks, labels, image_path, tracks_name, n, **kwargs)
                    samples_list.append(sample)
            else:
                samples_list.append(None)
        samples_dict[cond] = samples_list
    return samples_dict
                    

def add_cohort_tx_pairs(df, stratify_by):
    series = df[stratify_by[0]]
    for i in range(1, len(stratify_by)):
        series = series + ' ' + df[stratify_by[i]]
    df['stratify_by'] = series
    return df


def get_labels_path(df, i, sub_dirs, labels_dir, batch_name):
    labels_name = batch_name + '_' + df.loc[i, 'file'] + '_labels.zarr'
    if sub_dirs is not None:
        to_join = [labels_dir, ] + [df[sd][0] for sd in sub_dirs] + [labels_name, ]
    else:
        to_join = [labels_dir, labels_name]
    labels_path = os.path.join(*to_join)
    return labels_path


def make_bash_friendly(s):
    s = s.replace(' ', '_')
    s = s.replace('(', '_')
    s = s.replace(')', '_')
    return s

# ---------------------------------------
# Obtain image and labels data for sample
# ---------------------------------------

def save_sample_data(sample_dict, save_dir, img_channel='channel2'):
    for key in sample_dict.keys():
        cond_dir = os.path.join(save_dir, key)
        #print(cond_dir)
        os.makedirs(cond_dir, exist_ok=True)
        for image_sample in sample_dict[key]:
            image_sample = get_sample_hypervolumes(image_sample, img_channel=img_channel)
            save_sample(cond_dir, image_sample)



if __name__ == '__main__':
    p = '/home/abigail/data/plateseg-training/timeseries_seg/problem-children/210922_185235_problem-children_platelet-tracks/problem-children_metadata.csv'
    meta = pd.read_csv(p)
    print(p)
    tracks_dir = '/home/abigail/data/plateseg-training/timeseries_seg/problem-children/210922_185235_problem-children_platelet-tracks'
    image_dir = '/home/abigail/data/plateseg-training/timeseries_seg/problem-children'
    labels_dir = '/home/abigail/data/plateseg-training/timeseries_seg/problem-children/210922_185235_problem-children_segmentations/timeseries_seg/problem-children'
    save_dir = '/home/abigail/data/plateseg-training/timeseries_seg/problem-children'
    #sample_dict = get_sample_from_metadata(meta, tracks_dir, image_dir, save_dir=save_dir, name='PCs-2', sub_dirs=None, image_channel=None,
                       #                    use_h5=True, batch_name='210922_185235_problem-children', labels_dir=labels_dir)
    sample_dict = get_sample_from_metadata(meta, tracks_dir, image_dir, save_dir=save_dir, name='PC-terms-0', sub_dirs=None, image_channel=None,
                                           use_h5=True, batch_name='210922_185235_problem-children', labels_dir=labels_dir, sample_type='terminations')