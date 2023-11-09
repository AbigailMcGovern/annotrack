from annotrack import sample_tracks, get_sample_hypervolumes, save_sample
import pandas as pd
from pathlib import Path
import zarr


# ------------
# Choose files
# ------------
image_file = '20201015_Ms2_DMSO_1800.zarr'
labels_file = '230828_MxV1800is_plateseg2c.zarr'
species = 'mouse'
sample_name='20201015_MxV_1800is'
#.................

tracks_name = f'{sample_name}.parquet'

# -----
# Paths
# -----
tracks_path = f'/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/{species}/tracking/{tracks_name}'
image_path = f'/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/{species}/zarrs/{image_file}'
labels_path = f'/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/{species}/segmented_timeseries/{labels_file}'
save_dir = f'/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/{species}/tracking/track_samples'

# ------
# Set up
# ------
#tracks = pd.read_parquet(tracks_path)
name = Path(tracks_path).stem
labels = zarr.open(labels_path)
shape = labels.shape # hardcoded because image files are 5D and can potentially have c @ 0 or @ 1 (I only use 1 channel)
array_order = ('frame', 'z_pixels', 'y_pixels', 'x_pixels') # names of coord columns (in pixels) in tracks df in order of image dims
id_col = 'particle' # name of col with IDs
time_col = 'frame' # name of col with time (in frames)
frames = 30 # how many frames to show around the track segment
box = 60 # how big should the img box be in pixels in the dim/s with smallest pixel size (here x & y)
# box and frames are subject to rounding error by 1... still haven't bothered to fix this
scale = (1, 2, 0.5, 0.5) # scale of dims
min_track_length = 10 # this is the bit that is currently broken
n_samples = 30

# ------
# Sample
# ------
sample = sample_tracks(
    tracks_path,
    image_path, 
    shape,
    name,
    n_samples,
    labels_path=labels_path,
    frames=frames, 
    box=box,
    id_col=id_col, 
    time_col=time_col, 
    array_order=array_order, 
    scale=scale, 
    min_track_length=min_track_length)
    
# ADD IMAGE AND LABELS DATA
img_channel = 2 # which channel... if nd2 uses nd2 metadata to get channel axis
sample = get_sample_hypervolumes(sample, img_channel)

# SAVE SAMPLE
save_sample(save_dir, sample) # uses sample info and name provided above to name the sample



# -------
# Options
# -------

# Human 3000 is
# -------------
# image_file = '201118_hTR4_DMSO_3000s_.zarr'
# labels_file = '230824_HxV3000is_plateseg2c.zarr'
# species = 'human'
# treatment_name='HxV_3000is'
# sample_name='201118_HxV_3000is'

# Human 600 is
# ------------
# image_file = '201118_hTR4_21335_600_.zarr'
# labels_file = '230824_HxV600is_plateseg2c.zarr'
# species = 'human'
# sample_name='201118_HxV_600is'

# Mouse 600 is
# ------------
# image_file = '200917_Ms1_DMSO_600.zarr'
# labels_file = '230831_MxV600_DMSO_is_plateseg2c.zarr'
# species = 'mouse'
# sample_name='200917_MxV_600is'

# Mouse 600 is CMFDA
# ------------------
# image_file = '201104_MsDT87_CMFDA10_600_exp3.zarr'
# labels_file = '230828_MxV600_CMFDA_is_plateseg2c.zarr'
# species = 'mouse'
# sample_name='201104_CMFDA_MxV_600is'

# Mouse 1800 is - 1
# -----------------
#image_file = '20201015_Ms2_DMSO_1800.zarr'
#labels_file = '230828_MxV1800is_plateseg2c.zarr'
#species = 'mouse'
#sample_name='20201015_MxV_1800is'
