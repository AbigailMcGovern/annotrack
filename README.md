# annotrack
Annotate image tracking results from a variety of different conditions

## Installation 

Install the package into your chosen python environment

```bash
git clone <repository https or ssh>
cd annotrack
pip install .
```

## Sampling

To randomly sample track segments from a single tracks file corresponding to an image. To be honest, I don't use this directly but in `sample_from_meta.py` which obtains samples based on the metadata saved with the platelets data (super specific use case). 

```Python
from annotrack import sample_tracks, get_sample_hypervolumes, save_sample
import pandas as pd
from pathlib import Path

# SET UP
tracks_path = 'path/to/tracks.csv'
image_path = 'path/to/image/time/series"
labels_path = 'path/to/labels/time/series" # optionally None (default)
tracks = pd.read_csv(tracks_path)
name = Path(tracks_path).stem
shape = (194, 33, 512, 512) # hardcoded because image files are 5D and can potentially have c @ 0 or @ 1 (I only use 1 channel)
array_order = ('t', 'z_pixels', 'y_pixels', 'x_pixels') # names of coord columns (in pixels) in tracks df in order of image dims
id_col = 'particle' # name of col with IDs
time_col = 't' # name of col with time (in frames)
frames = 30 # how many frames to show around the track segment
box = 60 # how big should the img box be in pixels in the dim/s with smallest pixel size (here x & y)
# box and frames are subject to rounding error by 1... still haven't bothered to fix this
scale = (1, 2, 0.5, 0.5) # scale of dims
min_track_length = 50 # this is the bit that is currently broken
n_samples = 30

# SAMPLE
sample = sample_tracks(
    tracks,
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
    min_track_length=min_track_length,
    
# ADD IMAGE AND LABELS DATA
img_channel = 2 # which channel... if nd2 uses nd2 metadata to get channel axis
sample = get_sample_hypervolumes(sample, img_channel)

# SAVE SAMPLE
save_dir = 'directory/into/which/to/save/sample'
save_sample(save_dir, sample) # uses sample info and name provided above to name the sample
```

## Annotation

In the case that we are annotating multiple conditions to compare, we want to show them in the one session in randomised order with the annotator blinded to where the sample has originated from. We want to be able to annotated unannotated data from the sample without having the burden of having to do this all at once. The annotations are therefore saved into the saved sample. A selected number of samples saved from the various tracking experiments can be annotated using the following code. If you re-execute this code, you will only be shown not yet annotated data, unless you request otherwise. 

```Python
from annotrack import SampleViewer, prepare_sample_for_annotation

# where are the sample files belonging to different conditions that you want to compare
samples_to_view = {
    'condition_0' : ['path/to/condition/0/sample.smpl', ...], # however many
    'condition_1' : ['path/to/condition/1/sample.smpl', ...]
n_each = 30 # how many to look at from each condition
new_samples = prepare_sample_for_annotation(samples_to_view, n_each)

# annotate the samples
save_path = 'path/to/save/data/about/new_samples' # optional (default None) as the data is saved into the samples anyway
sample_viewer = SampleViewer(new_samples, save_path) # I would have de-objectified this... but I couldn't be arsed *sigh*
sample_viewer.annotate()
```

P.S., this hasn't been extensively tested... the main parts work in the conditions they've been tested in, but this probably only means I haven't yet seen where it breaks. 

Keys to navagate and annotate samples
- '2' - move to next sample
- '1' - move to previous sample
- 'y' - annotate as correct
- 'n' - annotate as containing an error
- 'i' - annotate the frame following a ID swap error
- 't' - annotate the fame following an incorrect termination
- 'Shift-t' - annotate the frame containing a false start error
- 's' - annotate an error ('i', 't', or 'Shift-t') as being associated with a segmentation error (merge or split of objects)

When an error is associated the specific frame ('i', 't', 'Shift-t', or 's'), the frame number (within the original image) will be added to a list of errors for the sample within the sample's (.smpl) info data frame. E.g., you may have a list of ID swaps for your sampled track segment (`[108, 111, 112]`) and a corresponding list of segmentation error associations (`[108, 112]`). 
