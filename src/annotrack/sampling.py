import re
from datetime import date
from itertools import repeat
import json
import numpy as np
import os
import pandas as pd 
from pandas import DataFrame
from pathlib import Path
from skimage.measure import regionprops
import time
from toolz import curry
from typing import Iterable, Union
import zarr
from nd2_dask.nd2_reader import nd2_reader
import dask.array as da
from datetime import datetime


# ------------------------------
# RANDOM SAMPLE & BOUNDING BOXES
# ------------------------------

def random_sample(
                  df: DataFrame,
                  shape: tuple, 
                  name: str,
                  n_samples: int,
                  frames: int =30, 
                  box: int =60,
                  id_col: str ='ID', 
                  time_col: str ='t', 
                  array_order: Iterable[str] =('t', 'x', 'y', 'z'), 
                  scale: Iterable[int] =(1, 1, 1, 4), 
                  non_tzyx_col: Union[Iterable[str], str, None] = None,
                  seed: Union[int, None] =None,
                  weights: Union[str, None] =None,
                  max_lost_prop: Union[float, None] =None,
                  **kwargs
                  ):
    """
    Take a random sample from a 
    """
    #array = single_zarr(image_path)
    #shape = array.shape
    _frames = np.floor_divide(frames, 2)
    _box = np.floor_divide(box, 2)
    # this is the image shape scaled to the data
    scaled_shape = [s * scale[i] # scaled shape in image order
                    for i, s in enumerate(shape)]
    # curried function. Add important info for later
    _add_info = _add_sample_info(
                                 id_col, 
                                 time_col,
                                 _frames, 
                                 max_lost_prop)
    if seed is None:
        seed = np.random.randint(0, 100_000)
    # initalise the sample dict 
    #   Why? The function is recursive (if max_lost_prop < 1.0)
    #       initialising within isnt an option.
    #       and mutable defaults are just silly
    sample = {}
    sample = _sample(
                     sample, 
                     df, 
                     n_samples, 
                     seed, 
                     array_order, 
                     shape, 
                     name, 
                     weights, 
                     _add_info
                     )
    # sample scale is the scale that brings the data to the image
    sample = _estimate_bounding_boxes(
                                      sample, 
                                      shape,
                                      id_col, 
                                      time_col, 
                                      _frames, 
                                      _box,
                                      array_order, 
                                      non_tzyx_col, 
                                      scale
                                      )
    sample['info'] = _tracks_df(df, sample, id_col, time_col, array_order)
    return sample



# Sampling
# --------

def _sample(
            sample, 
            df, 
            n_samples, 
            seed, 
            array_order, 
            shape, 
            name, 
            weights, 
            _add_info):
    """
    select random sample
    """
    for i, col in enumerate(array_order):
        df = df[df[col] < shape[i]].copy()
    if n_samples == 0:
        print(f"Random {name} obtained")
        pairs = list(sample.keys())
        return sample
    elif n_samples < 0:
        # remove excess - print so that I can see if this is ever executed
        #   I don't think it will be, but just in case
        excess = abs(n_samples)
        print(f"Removing {excess} excess {name}")
        pairs = list(sample.keys())
        for i in range(excess):
            del sample[pairs[i]]
        return sample
    # If we don't get enough samples 
    #   (e.g., chooses a time point too near start or end)
    #   the function will call itself (with remainder samples to obtain)
    # NOTE: the max proportion of frames that can be lost in the track can be set 
    else:
        print(f'Sampling {n_samples}...')
        if weights is not None:
            w = df[weights].values
        else:
            w = None
        #iamkitten
        kittens_sample = df.sample(n=n_samples, weights=w, random_state=seed) # kitten's
        num_obtained = _add_info(kittens_sample, df, sample)
        n_samples = n_samples - num_obtained
        # this whole recursion thing doesnt really work when you use the same seed
        seed += 1 
        return _sample(
                       sample, 
                       df, 
                       n_samples, 
                       seed, 
                       array_order, 
                       shape, 
                       name, 
                       weights, 
                       _add_info)


# referenced in random_sample(...)
@curry
def _add_sample_info(id_col, 
                     time_col,
                     _frames, 
                     max_lost_prop,
                     sample_df, 
                     df, 
                     sample
                     ):
    counter = 0
    for i in range(len(sample_df)):
        ID = sample_df[id_col].values[i]
        t = sample_df[time_col].values[i]
        pair = (ID, t)
        wt_df = df[df[id_col] == ID].reset_index(drop=True)
        #print('time: ', t, ', no tracks: ', len(wt_df), ', tmax, tmin: ', 
         #    wt_df[time_col].values.max(), wt_df[time_col].values.min(), ', expected tracks: ', wt_df['track_length'].values.mean())
        lil_df = df.loc[(df[id_col] == ID) &
            (df[time_col] >= t - _frames) & 
            (df[time_col] <= t + _frames)
            ]
        if max_lost_prop is None:
            max_lost_prop = 1.
        right_len = len(lil_df) >= np.floor((1-max_lost_prop) * (_frames * 2 + 1))
        right_len = right_len and len(lil_df) <= _frames * 2 + 1
        new_pair = pair not in sample.keys()
        row = df.loc[(df[id_col]==pair[0]) & 
                     (df[time_col]==pair[1])].copy()
        right_row_len = len(row) == 1
        if right_len & new_pair & right_row_len:
            lil_df = lil_df.reset_index(drop=True)
            try:
                lil_df = lil_df.drop(columns=['Unnamed: 0'])
            except:
                pass
            sample[pair] = {'df': lil_df}
            counter += 1
    return counter


# Bounding Boxes
# --------------

# referenced in random_sample(...)
def _estimate_bounding_boxes(
                             sample, 
                             shape,
                             id_col, 
                             time_col, 
                             _frames, 
                             _box,
                             array_order, 
                             non_tzyx_col, 
                             image_scale
                             ):
    print('Finding bounding boxes...')     
    hc_shape, sample_scale = _box_shape(
                                        array_order, 
                                        time_col, 
                                        non_tzyx_col, 
                                        _frames, 
                                        _box, 
                                        shape, 
                                        image_scale
                                        ) 
    coords_cols = _coords_cols(array_order, non_tzyx_col, time_col)
    pairs = sample.keys()
    for pair in pairs:
        df = sample[pair]['df']
        row = df.loc[(df[id_col]==pair[0]) & 
                     (df[time_col]==pair[1])].copy()
        sample[pair]['corr'] = {}
        # initialise the bounding box info
        sample[pair]['b_box'] = {}
        # TIME SLICE
        # find last time frame index
        for i, col in enumerate(array_order):
            if col == time_col:
                t_lim = shape[i] - 1
        sample = _time_slice(
                             pair, 
                             sample, 
                             df, 
                             t_lim, 
                             time_col, 
                             _frames, 
                             hc_shape
                             )
        # SPATIAL COORDS SLICES
        sample = _coords_slices(
                                array_order, 
                                coords_cols, 
                                hc_shape, 
                                sample_scale, 
                                row, 
                                _box, 
                                image_scale, 
                                sample,
                                shape,
                                df, 
                                pair
                                )
        # NON SPATIAL OR TIME SLICES
        sample = _non_tzyx_slices(sample, pair, non_tzyx_col)
    return sample


# referenced in _estimate_bounding_boxes
def _box_shape(
               array_order, 
               time_col, 
               non_tzyx_col, 
               _frames, 
               _box, 
               shape, 
               image_scale
               ):
    # get the scale that brings the data to the image
    sample_scale = []
    for i, col in enumerate(array_order):
            s = np.divide(1, image_scale[i])
            sample_scale.append(s)
    sample_scale = np.array(sample_scale)
    sample_scale = sample_scale / sample_scale.max()
    for i, col in enumerate(array_order):
        if col == time_col:
            sample_scale[i] = 1
    # if necessary, change configuration of non_tzyx_col
    if not isinstance(non_tzyx_col, Iterable):
        non_tzyx_col = [non_tzyx_col]
    # get the hypercube shape
    hc_shape = []
    for i, col in enumerate(array_order):
        if col == time_col:
            scaled = np.multiply(_frames*2+1, sample_scale[i])
            hc_shape.append(scaled)
        elif col in non_tzyx_col:
            scaled = np.multiply(shape[i]*2+1, sample_scale[i])
            hc_shape.append()
        else:
            scaled = np.multiply(_box*2+1, sample_scale[i])
            hc_shape.append(scaled)
    hc_shape = np.floor(np.array(hc_shape)).astype(int)
    return hc_shape, sample_scale


# referenced in _estimate_bounding_boxes
def _coords_cols(array_order, non_tzyx_col, time_col):
    if isinstance(non_tzyx_col, Iterable):
        coords_cols = non_tzyx_col.copy().append(time_col)
    else:
        coords_cols = [col for col in array_order if col \
                    not in [time_col, non_tzyx_col]]
    return coords_cols


# referenced in _estimate_bounding_boxes
def _time_slice(
                pair, 
                sample, 
                df, 
                t_lim, 
                time_col, 
                _frames, 
                hc_shape
                ):
    # get the estimated time index
    t_min, t_max = pair[1] - _frames, pair[1] + _frames
    # correct the min time index
    if t_min < 0:
        t_min = 0
    # correct the max time index
    if t_max > t_lim:
        t_max = t_lim
    t_max += 1
    # correct the track data for the bounding box
    df = sample[pair]['df']
    df[time_col] = df[time_col] - t_min 
    sample[pair]['b_box'][time_col] = slice(t_min, t_max)
    sample[pair]['df'] = df
    return sample


# referenced in _estimate_bounding_boxes
def _coords_slices(
                   array_order, 
                   coords_cols, 
                   hc_shape, 
                   sample_scale, 
                   row, 
                   _box, 
                   image_scale, 
                   sample,
                   shape,
                   df, 
                   pair
                   ):
    df = sample[pair]['df']
    for i, coord in enumerate(array_order):
        if coord in coords_cols:
            sz = hc_shape[i]
            # scale the tracks value to fit the image
            sample_2_box_scale = sample_scale[i]
            # scale the bounding box for the image coordinate
            box = np.multiply(_box, sample_2_box_scale)
            # get the coord in pixels and the projected bounding box
            value = row[coord].values[0]
            col_min = np.floor(value - box).astype(int)
            col_max = np.floor(value + box).astype(int)
            sz1 = col_max - col_min
            diff = sz1 - sz
            col_min, col_max = _correct_diff(diff, col_min, col_max)
            # slice corrections
            #box_to_tracks_scale = image_scale[i]
            # get the max and min values for track vertices in pixels
            max_coord = df[coord].max()
            min_coord = df[coord].min()
            # correct box shape if the box is too small to capture the 
            # entire track segment
            if min_coord < col_min:
                col_min = np.floor(min_coord).astype(int)
            if max_coord > col_max:
                col_max = np.ceil(max_coord).astype(int)
            if col_min < 0:
                col_min = 0
            if col_max > shape[i]:
                col_max = shape[i]
            sample[pair]['b_box'][coord] = slice(col_min, col_max)
            # need to construct correctional slices for getting 
            # images that are smaller than the selected cube volume
            # into the cube. 
            sz1 = col_max - col_min
            diff = sz - sz1
            # the difference should not be negative 
            #m0 = 'The cube dimensions should not be '
            #m1 = 'greater than the image in any axis: '
            #m2 = f'{col_min}:{col_max} for coord {coord}'
            #assert diff >= 0, m0 + m1 + m2
            # correct the position data for the frame
            #c_min = (col_min * box_to_tracks_scale)
            df[coord] = df[coord] - col_min 
    return sample


# referenced in _coords_slices
def _correct_diff(diff, col_min, col_max):
    odd = diff % 2
    if diff < 0:
        diff = abs(diff)
        a = -1
    else:
        a = 1
    adj = np.floor_divide(diff, 2)
    col_min = col_min - (adj * a)
    if odd:
        adj = adj + 1
        col_max = col_max - (adj * a)
    else:
        col_max = col_max - (adj * a)
    return col_min, col_max


# referenced in _estimate_bounding_boxes
def _non_tzyx_slices(sample, pair, non_tzyx_col):
    if non_tzyx_col is not None:
        for col in non_tzyx_col:
            sample[pair]['b_box'][col] = slice(None)
    return sample


# referenced in random_sample(...)
def _tracks_df(df, sample, id_col, time_col, array_order):
    info = []
    pairs = sample.keys()
    for pair in pairs:
        # get the row of info about the sampled segment
        row = df.loc[(df[id_col]==pair[0]) & 
                     (df[time_col]==pair[1])].copy()
        # add summary stats for the track
        for col in sample[pair]['df'].columns.values:
            if isinstance(col[0], str):
                pass
            else:
                mean = sample[pair]['df'][col].mean()
                sem = sample[pair]['df'][col].sem()
                mean_name = col + '_seg_mean'
                sem_name = col + '_seg_sem'
                row.loc[:, mean_name] = mean
                row.loc[:, sem_name] = sem
        # add bounding box information 
        row = _add_bbox_info(pair, row, sample, array_order, time_col)
        info.append(row)
    info = pd.concat(info)
    info = info.reset_index(drop=True)
    if 'Unnamed: 0' in info.columns.values:
        info = info.drop(columns=['Unnamed: 0'])
    info['annotated'] = [False, ] * len(info)
    return info 


def _add_bbox_info(pair, row, sample, array_order, time_col):
    for coord in array_order:
        s_ = sample[pair]['b_box'][coord]
        s_min = s_.start
        n_min = coord + '_start'
        s_max = s_.stop
        n_max = coord + '_stop'
        row.loc[:, n_min] = [s_min,]
        row.loc[:, n_max] = [s_max,]
    num_frames = sample[pair]['df'][time_col].max() \
                    - sample[pair]['df'][time_col].min()
    row.loc[:, 'frames'] = [num_frames,]
    return row


# -------------
# SAMPLE TRACKS
# -------------

def sample_tracks(tracks_path: str,
                  image_path: str, 
                  shape : tuple,
                  name: str,
                  n_samples: int,
                  labels_path: Union[str, None]=None,
                  frames: int =30, 
                  box: int =60,
                  id_col: str ='ID', 
                  time_col: str ='t', 
                  array_order: Iterable[str] =('t', 'x', 'y', 'z'), 
                  scale: Iterable[int] =(1, 1, 1, 4), 
                  non_tzyx_col: Union[Iterable[str], str, None] = None,
                  seed: Union[int, None] =None,
                  weights: Union[str, None] =None,
                  max_lost_prop: Union[float, None] =None,
                  min_track_length: Union[int, None] =30,
                  **kwargs):
    
    if tracks_path.endswith('.csv'):
        df = pd.read_csv(tracks_path)
    elif tracks_path.endswith('.parquet'):
        df = pd.read_parquet(tracks_path)
    else:
        raise ValueError('please ensure the tracks file is a csv or parquet')
    # calculate weights if required
    coords_cols = _coords_cols(array_order, non_tzyx_col, time_col)
    # well this was lazy (see below)
    df = _add_disp_weights(df, coords_cols, id_col)
    # older code from elsewhere, decided it wasn't hurting anything
    if weights is not None:
        if weights not in df.columns.values.tolist():
            if weights == '2-norm':
                df = _add_disp_weights(df, coords_cols, id_col)
            elif weights == 'log-frequency':
                df = _add_logfreq_weights(df, id_col)
            else: 
                cols = df.columns.values
                if weights not in cols:
                    m = 'Please use a column in the data frame or 2-norm to add distances'
                    raise(KeyError(m))
    # filter for min track length
    df['track_length'] = df['nrtracks']
    print(min_track_length)
    print(len(df))
    if min_track_length is not None:
        #from misc import find_splits, no_tracks_correct
        #splits = find_splits(df, time_col=time_col, id_col=id_col)
        #assert len(splits) == 0
        #no_tracks_correct(df, id_col)
        df = df[df['track_length'] >= min_track_length]
        df = df.reset_index()
        print(df['track_length'].values.min())
        print(len(df))
    # get the sample
    sample = random_sample(
                           df,
                           shape, 
                           name,
                           n_samples,
                           frames, 
                           box,
                           id_col, 
                           time_col, 
                           array_order, 
                           scale, 
                           non_tzyx_col,
                           seed,
                           weights,
                           max_lost_prop,
                           **kwargs)
    # add track arrays ready for input to napari tracks layer 
    sample = _add_track_arrays(sample, id_col, time_col, coords_cols)
    sample['image_path'] = image_path
    sample['labels_path'] = labels_path
    sample['sample_type'] = 'random tracks'
    sample['tracks_path'] = tracks_path
    sample = _add_construction_info(sample, id_col, time_col, 
                                    array_order, non_tzyx_col)
    return sample


# referenced in sample_tracks()
def _add_disp_weights(df, coords_cols, id_col):
        """
        Get L2 norm of finite difference across x,y,z for each track point
        These can be used as weights for random track selection.
        """
        coords = coords_cols
        weights = []
        ids = df[id_col].values
        unique_ids = list(df[id_col].unique())
        for ID in list(df[id_col].unique()):
            id_df = df.loc[(df[id_col] == ID)][coords]
            # get the finite difference for the position vars
            diff = id_df.diff()
            diff = diff.fillna(0)
            # generate L2 norms for the finite difference vectors  
            n2 = list(np.linalg.norm(diff.to_numpy(), 2, axis=1))
            weights.extend(n2)
        v0 = len(weights)
        v1 = len(df)
        m = 'An issue has occured when calculating track displacements'
        m = m + f': the length of the data frame ({v1}) does not equal '
        m = m + f'that of the displacements ({v0})'
        assert v0 == v1, m 
        df['2-norm'] = weights
        return df


# referenced in sample_tracks()
def _add_logfreq_weights(df, id_col):
    bin_count = np.bincount(df[id_col].values)
    id_dict = {i : bin_count[i] for i in range(len(bin_count))}
    df['frequency'] = df[id_col].apply(id_dict.get)
    df['log-frequency'] = np.log(df['frequency'].values)
    max_ = df['log-frequency'].values.max()
    min_ = df['log-frequency'].values.min()
    print(min_, max_)
    return df


# referenced in sample_tracks() 
def _add_track_arrays(sample, id_col, time_col, coords_cols):
    '''
    For each sample, as represented by the ID-time tuple in the saple dict,
    obtain a 'tracks' data frame containing the information for displaying 
    the sampled track in napari (cols: ID, t, z, y, x).
    '''
    cols = [id_col, time_col]
    for c in coords_cols:
        cols.append(c)
    for pair in sample.keys():
        if isinstance(pair, tuple):
            df = sample[pair]['df']
            tracks = df[cols].to_numpy()
            sample[pair]['tracks'] = tracks
    # despite the fact that the sample dict is modified regardless of whether
    # it is returned, the sample = function(sample, arg0, ...) approach
    # is the clearest way to demonstrate logic & info flow
    return sample 

# -------------------------
# SAMPLE TRACK TERMINATIONS
# -------------------------

def sample_track_terminations(tracks_path: str,
                              image_path: str,
                              shape: tuple, 
                              name: str,
                              n_samples: int,
                              labels_path: Union[str, None]=None,
                              frames: int =10, 
                              box: int =60,
                              id_col: str ='ID', 
                              time_col: str ='t', 
                              array_order: Iterable[str] =('t', 'x', 'y', 'z'), 
                              scale: Iterable[int] =(1, 1, 1, 4), 
                              non_tzyx_col: Union[Iterable[str], str, None] = None,
                              seed: Union[int, None] =None,
                              weights: Union[str, None] =None,
                              max_lost_prop: Union[float, None] =None,
                              min_track_length: Union[int, None] =20,
                              **kwargs
                              ):
    # filter data frame for terminations
    #
    # get sample
    sample = sample_tracks(
                           tracks_path,
                           image_path,
                           shape, 
                           name,
                           n_samples,
                           labels_path,
                           frames, 
                           box,
                           id_col, 
                           time_col, 
                           array_order, 
                           scale, 
                           non_tzyx_col,
                           seed,
                           weights,
                           max_lost_prop,
                           min_track_length,
                           **kwargs
                           )
    sample['sample_type'] = 'terminating tracks'
    return sample


# --------------
# SAMPLE OBJECTS
# --------------


def sample_objects(
                   tracks,
                   labels,
                   image_path, 
                   name,
                   n_samples,
                   frames, 
                   box,
                   id_col, 
                   time_col, 
                   array_order, 
                   scale, 
                   non_tzyx_col,
                   seed,
                   max_lost_prop,
                   **kwargs
                   ): 
    #
    array = single_zarr(image_path)
    _frames = np.floor_divide(frames, 2)
    _box = np.floor_divide(box, 2)
    objs = get_objects_without_tracks(
                               labels, 
                               tracks, 
                               id_col, 
                               time_col, 
                               array_order, 
                               scale,
                               _frames, 
                               _box
                               )
    #
    try:
        kittens_sample = objs.sample(n=n_samples, random_state=seed)
    except:
        kittens_sample = objs
        print(len(objs))
    #
    shape = labels.shape
    sample = _get_object_tracks(labels,
                       kittens_sample, 
                       tracks, 
                       shape,
                       _frames, 
                       _box, 
                       scale, 
                       id_col, 
                       time_col, 
                       array_order, 
                       non_tzyx_col
                       )
    sample['image_path'] = image_path
    sample['sample_type'] = 'untracked objects'
    sample = _add_construction_info(sample, id_col, time_col, 
                                    array_order, non_tzyx_col)
    return sample



def _get_object_tracks(labels,
                       df, 
                       tracks, 
                       shape,
                       _frames, 
                       _box, 
                       scale, 
                       id_col, 
                       time_col, 
                       array_order, 
                       non_tzyx_col
                       ):
    # some scaled values for later
    # this is the image shape scaled to the data
    scaled_shape = [np.round(s * scale[i]).astype(int) # scaled shape in image order
                    for i, s in enumerate(shape)]
    inv_scale = np.divide(1, scale)
    hlf_hc = []
    for c in array_order:
        if c == time_col:
            n = _frames
        else:
            n = _box
        hlf_hc.append(n)
    # generate sample dict
    sample = {}
    df = df.reset_index(drop=True)
    df['frames'] = [None] * len(df)
    df['n_objects'] = [None] * len(df)
    df['t_start'] = [None] * len(df)
    # go through the sample and compose bounding boxes
    for idx in df.index:
        point_coords = []
        # get pair info for each sampled object
        # used as key for each object
        pair = (df.loc[idx, id_col], df.loc[idx, time_col])
        sample[pair] = {}
        # generate df
        lil_tracks = tracks.copy()
        b_box = {}
        for i, c in enumerate(array_order):
            coord = df.loc[idx, c]
            max_ = np.round(coord * scale[i] + hlf_hc[i])
            min_ = np.round(coord * scale[i] - hlf_hc[i])
            # correct for edges of image
            if max_ >= scaled_shape[i]:
                max_ = scaled_shape[i]
            if min_ <  0:
                min_ = 0
            # get all tracks that live in this box 
            lil_tracks = lil_tracks.loc[(lil_tracks[c] >= min_) & (lil_tracks[c] < max_)]
            if c == time_col:
                df.loc[i, 't_start'] = min_
            lil_tracks[c] = lil_tracks[c] - min_ 
            # same coordinates in image slicing scale
            b_min = np.round(coord - hlf_hc[i] * inv_scale[i]).astype(int)
            b_max = np.round(coord + hlf_hc[i] * inv_scale[i]).astype(int)
            # correct for edges of image
            if b_max >= shape[i]:
                b_max = shape[i] - 1
            if b_min < 0:
                b_min = 0 
            # add the coordinate slice to the pair info 
            b_box[c] = slice(b_min, b_max)
            # add the coordinate in image scale to the point coords
            point_coords.append(coord - b_min)
        # add the point to the pair
        sample[pair]['point'] = np.array([point_coords])
        # add the hypercube slices to the paie
        sample[pair]['b_box'] = b_box
        # add the tracks info to the pair
        lil_tracks = lil_tracks.reset_index(drop=True)
        sample[pair]['df'] = lil_tracks
        # get the tracks array for input to napari
        cols = [id_col, time_col]
        coord_cols = _coords_cols(array_order, non_tzyx_col, time_col)
        for c in coord_cols:
            cols.append(c)
        only_tracks = lil_tracks[cols].to_numpy()
        sample[pair]['tracks'] = only_tracks
    # add the sample info to the sample dict
    sample['info'] = df
    return sample
        

            
def get_objects_without_tracks(
                               labels, 
                               tracks, 
                               id_col, 
                               time_col, 
                               array_order, 
                               scale,
                               _frames, 
                               _box
                               ):
    """
    The 
    """
    df = {c:[] for c in array_order}
    df[id_col] = []
    df['area'] = []
    coord_cols = [c for c in array_order if c != time_col]
    coord_scale = [scale[i] for i, c in enumerate(array_order) if c != time_col]
    for t in range(labels.shape[0] - 1):
        try:
            frame = np.array(labels[t, ...])
            # get the tracks at this point in time 
            t_trk = tracks.copy()
            t_trk = t_trk.loc[t_trk[time_col] == t]
            # get the properties of objects in the frame
            props = regionprops(frame)
            # go through properties. 
            no_tracks = []
            for p in props:
                label = p['label']
                bbox = p['bbox']
                ct = t_trk.copy()
                # get any tracks in the bounding box for this obj
                for i, c in enumerate(coord_cols):
                    min_ = bbox[i] * coord_scale[i]
                    max_ = bbox[i + len(coord_cols)] * coord_scale[i]
                    ct = ct.loc[(ct[c] >= min_) & (ct[c] < max_)]
                # if there are no tracks in the bbox, add to the list
                if len(ct) == 0:
                    no_tracks.append(label)
            # based on list entries, make dataframe for objects
            for p in props:
                if p['label'] in no_tracks:
                    df[time_col].append(t)
                    for i, c in enumerate(coord_cols):
                        df[c].append(p['centroid'][i])
                    df[id_col].append(p['label'])
                    df['area'].append(p['area'])
        except KeyError:
            print(t)
    df = pd.DataFrame(df)
    print(f'Found {len(df)} untracked objects')
    return df
        

# -------
# Helpers
# -------

def add_track_length(df, id_col, new_col='track_length'):
    bincounts = np.bincount(df[id_col].values)
    ids_dict = {ID : num for ID, num in enumerate(bincounts) if num > 0}
    df[new_col] = [None, ] * len(df)
    for ID in ids_dict.keys():
        idxs = df[df[id_col] == ID].index.values
        df.at[idxs, new_col] = ids_dict[ID]
    return df


def single_zarr(input_path, c=2, idx=0):
    '''
    Parameters
    ----------
    c: int or tuple
        Index of indices to return in array
    idx: int or tuple
        which indicies of the dim to apply c to
    '''
    assert type(c) == type(idx)
    arr = da.from_zarr(input_path)
    slices = [slice(None)] * arr.ndim
    if isinstance(idx, int):
        slices[idx] = c
    elif isinstance(idx, tuple):
        for i, ind in enumerate(idx):
            slices[ind] = c[i]
    else:
        raise TypeError('c and idx must be int or tuple with same type')
    slices  = tuple(slices)
    arr = arr[slices]
    return arr


# referenced in sample_tracks() and sample_objects()
def _add_construction_info(sample, id_col, time_col, array_order, non_tzyx_col):
    '''
    Add information that can be used to (1) distinguish coordinate and ID
    columns of the data frames, (2) reconstruct names of other columns
    containing important information, and (3) determine how said columns
    relate to array dimensions. 
    '''
    sample['coord_info'] = {
        'id_col' : id_col, 
        'time_col' : time_col,
        'array_order' : array_order, 
        'non_tzyx_col' : non_tzyx_col
    }
    return sample


def _guess_channel_axis(shp):
    initial_guess = np.argmin(shp)
    return initial_guess if shp[initial_guess] < 6 else None


def open_with_correct_modality(image_path, channel=None, chan_axis=None):
    if chan_axis is None:
        chan_axis = _guess_channel_axis(image.shape)
    suffix = Path(image_path).suffix
    if suffix == '.zarr':
        image = zarr.open(image_path, 'r')
        if isinstance(image, zarr.hierarchy.Group):  # assume ome-zarr
            image = zarr['/0']
            if not isinstance(image, zarr.core.Array):
                raise ValueError(
                        'Not a zarr array or ome zarr array: {image_path}'
                        )

        if channel is not None and chan_axis is not None:
            # extract only the desired channel
            image = da.array(image)
            ix = [slice(None, None), ] * image.ndim
            ix[chan_axis] = channel
            image = image[tuple(ix)]
    else:
        # only support zarr
        raise ValueError('only zarr images are supported')
    return image


def read_from_h5(h5_path, channel='channel2'):
    '''
    For corrupted nd2 files saved as h5 files using Fiji
    h5 key structure for these files is as follows:
        {'t<index>' : {'channel<index>' : <3d array>, ...}, ...} 
    '''
    import h5py
    #print('imported h5py')
    with h5py.File(h5_path) as f:
        #print('read h5 file')
        t_keys = [int(key[1:]) for key in f.keys() if 't' in key]
        t_keys = ['t' + str(key) for key in sorted(t_keys)]
        #print(t_keys)
        c_keys = [key for key in f[t_keys[0]].keys()]
        #print(c_keys)
        #images = []
        frame = f[t_keys[0]][c_keys[0]]
        #print('Getting channel with key: ', channel)
        for i, c in enumerate(c_keys):
            if c == channel:
                chan_image = np.zeros((len(t_keys), ) + frame.shape, dtype=frame.dtype)
                #print('Generated empty numpy for channel with shape: ', chan_image.shape)
                for j, t in enumerate(t_keys):
                    chan_image[j, :, :, :] = f[t][c]
                #print('Added h5 data to np array')
                image = da.from_array(chan_image)
                #print('converted to dask')
                #images.append(chan_da)
                #image = chan_da
    return image


# ---------------
# GET SAMPLE DATA
# ---------------

def get_sample_hypervolumes(sample, img_channel=None):
    image_path = sample['image_path']
    #print(image_path)
    labels_path = sample['labels_path']
    image = open_with_correct_modality(image_path, img_channel)
    if labels_path is not None:
        labels = open_with_correct_modality(labels_path)
    else:
        labels = None
    array_order = sample['coord_info']['array_order']
    pairs = [key for key in sample.keys() if isinstance(key, tuple)]
    #if labels is not None:
        #print(labels.shape)
        #labels = np.array(labels)
    for pair in pairs:
        l = len(array_order)
        m = f'Image must be of same dimensions ({l}) as in the sample array_order: {array_order}'
        assert image.ndim == l, m
        slice_ = []
        for key in array_order:
            s_ = sample[pair]['b_box'][key]
            slice_.append(s_)
        #print(slice_)
        #print(image)
        img = image[tuple(slice_)]
        if isinstance(img, da.core.Array):
            img = img.compute()
        if labels is not None:
            lab = labels[tuple(slice_)]
            if isinstance(lab, da.core.Array):
                lab = img.compute()
        else:
            lab = None
        sample[pair]['image'] = img
        sample[pair]['labels'] = lab
    return sample




# -----------
# SAVE SAMPLE 
# -----------

def save_sample(save_dir, sample):
    """
    """
    pairs = [key for key in sample.keys() if isinstance(key, tuple)]
    n_samples = len(pairs)
    file_name = Path(sample['tracks_path']).stem
    if sample['sample_type'] == 'random tracks':
        tp = 'rtracks'
    elif sample['sample_type'] == 'track terminations':
        tp = 'tterm'
    elif sample['sample_type'] == 'untracked objects':
        tp = 'uobj'
    else:
        tp = 'ukn'
    now = datetime.now()
    dt = now.strftime("%y%m%d_%H%M%S")
    name = file_name + f'_{tp}_{dt}_n={n_samples}.smpl'
    sample_dir = os.path.join(save_dir, name)
    os.makedirs(sample_dir, exist_ok=True)
    # save base for sample
    sample_json = {
        'image_path' : sample['image_path'],
        'labels_path' : sample['labels_path'],
        'sample_type' : sample['sample_type'], 
        'coord_info' : sample['coord_info'], 
        'tracks_path' : sample['tracks_path']
    }
    #print(sample_json)
    json_name = file_name + '_read-info.json'
    with open(os.path.join(sample_dir, json_name), 'w') as f:
        json.dump(sample_json, f, indent=4)
    # save the info data frame
    info_name = file_name + '_info.csv'
    sample['info'].to_csv(os.path.join(sample_dir, info_name))
    # save the individual samples
    for pair in pairs:
        pair_name = f'id-{pair[0]}_t-{pair[1]}'
        pair_dir = os.path.join(sample_dir, pair_name)
        os.makedirs(pair_dir)
        # save the df
        df_name = pair_name + '_df.csv'
        sample[pair]['df'].to_csv(os.path.join(pair_dir, df_name))
        # save the tracks
        tracks_name = pair_name + '_tracks.zarr'
        t_data = sample[pair]['tracks']
        tracks = zarr.open(os.path.join(pair_dir, tracks_name), mode='w', shape=t_data.shape, chunks=t_data.shape)
        tracks[:, :] = t_data
        # save the bounding box
        bbox_name = pair_name + '_bbox.json'
        new_bbox = sample[pair]['b_box'].copy()
        for key in new_bbox.keys():
            s_ = new_bbox[key]
            new_bbox[key] = (int(s_.start), int(s_.stop), s_.step)
        with open(os.path.join(pair_dir, bbox_name), 'w') as f:
            json.dump(new_bbox, f, indent=4)
        # if there are any corrections, save
        try:
            if len(sample[pair]['corr']) > 0:
                corr_name = pair_name + '_corr.json'
                with open(os.path.join(pair_dir, corr_name), 'w') as f:
                    json.dump(sample[pair]['corr'], f)
        except:
            pass
        # if the image is in the sample, save this
        try:
            img = sample[pair]['image']
            image_path = os.path.join(pair_dir, pair_name + '_image.zarr')
            img_zarr = zarr.open(image_path, mode='w', shape=img.shape, chunks=img.shape, dtype=img.dtype)
            img_zarr[:, :, :, :] = img
        except KeyError:
            pass
        # if the label is in the sample, save this
        try:
            lab = sample[pair]['labels']
            labels_path = os.path.join(pair_dir, pair_name + '_labels.zarr')
            lab_zarr = zarr.open(labels_path, mode='w', shape=lab.shape, chunks=lab.shape, dtype=lab.dtype)
            lab_zarr[:, :, :, :] = lab
        except KeyError:
            pass
    return sample_dir

    

def read_sample(sample_path):
    """
    Parse sample info from .smpl 'file'
    """
    files = os.listdir(sample_path)
    json_path = [f for f in files if f.endswith('_read-info.json')]
    assert len(json_path) == 1, 'There should be exactly 1 *_read-info.json file in the sample (.smpl)'
    # get base for sample
    with open(os.path.join(sample_path, json_path[0]), 'r') as f:
        sample = json.load(f)
    # get sample info
    info_path = [f for f in files if f.endswith('info.csv')]
    assert len(info_path) == 1, 'There should be exactly 1 *_info.csv file in the sample (.smpl)'
    sample['info'] = pd.read_csv(os.path.join(sample_path, info_path[0]))
    info_path = os.path.join(sample_path, info_path[0])
    # get individual sample
    regex = re.compile(r'id-\d*_t-\d*')
    pairs = [f for f in files if len(regex.findall(f)) > 0]
    for str_pair in pairs:
        pair_dir = os.path.join(sample_path, str_pair)
        pair_dict = {}
        # get df
        df_path = os.path.join(pair_dir, str_pair + '_df.csv')
        pair_dict['df'] = pd.read_csv(df_path)
        pair_dict['df_path'] = df_path
        pair_dict['info_path'] = info_path
        # get tracks
        tracks_path = os.path.join(pair_dir, str_pair + '_tracks.zarr')
        pair_dict['tracks'] = zarr.open(tracks_path)
        # get b_box
        bbox_path = os.path.join(pair_dir, str_pair + '_bbox.json')
        with open(bbox_path, 'r') as f:
            bbox = json.load(f)
        new_bbox = {}
        for key in bbox.keys():
            args = list(bbox[key])
            new_bbox[key] = slice(*args)
        pair_dict['b_box'] = new_bbox
        # find out if corrs exist and read if so
        corr_path = [f for f in os.listdir(pair_dir) if f.endswith('_corr.json')]
        if len(corr_path) > 0:
            with open(os.path.join(pair_dir, corr_path[0]), 'r') as f:
                corr = json.load(f)
            pair_dict['corr'] = corr
        # if image exists, read
        img_path = os.path.join(pair_dir, str_pair + '_image.zarr')
        if os.path.exists(img_path):
            img = zarr.open(img_path)
            #print(img, img_path)
            pair_dict['image'] = img
        # if labels exist, read
        lab_path = os.path.join(pair_dir, str_pair + '_labels.zarr')
        if os.path.exists(lab_path):
            lab = zarr.open(lab_path)
            #print(lab.shape)
            pair_dict['labels'] = lab
        # add pair to sample
        id_re = re.compile(r'id-\d*')
        ID = id_re.findall(str_pair)[0][3:]
        t_re = re.compile(r't-\d*')
        t = t_re.findall(str_pair)[0][2:]
        pair = (int(ID), int(t))
        sample[pair] = pair_dict
        #print(str_pair, pair)
    #print(list(sample.keys()))
    #print(list(sample[eval(pairs[0])].keys()))
    return sample




def save_sample_volumes(sample, sample_dir):
    """
    Save image data corresponding to a sample
    """
    pairs = [key for key in sample.keys() if isinstance(key, tuple)]
    for pair in pairs:
        pair_dir = os.path.join(sample_dir, str(pair))
        try:
            img = sample[pair]['image']
            image_path = os.path.join(pair_dir, str(pair) + '_image.zarr')
            zarr.save(image_path, img)
        except KeyError:
            pass
        # if the label is in the sample, save this
        try:
            lab = sample[pair]['labels']
            labels_path = os.path.join(pair_dir, str(pair) + '_labels.zarr')
            zarr.save(labels_path, lab)
        except KeyError:
            pass

