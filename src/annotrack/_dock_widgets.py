import pandas as pd
import os
from pathlib import Path
import napari
import zarr
from magicgui import magic_factory
from .sampling import sample_tracks, get_sample_hypervolumes, save_sample
from .annotation import SampleViewer
from .sample_management import prepare_sample_for_annotation
from typing import Tuple, Union


@magic_factory(
    path_to_csv={'widget_type': 'FileEdit', 'mode': 'r'},
    output_dir={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    tzyx_cols={'widget_type' : 'LiteralEvalLineEdit'},
    scale={'widget_type' : 'LiteralEvalLineEdit'},
)
def sample_from_csv(
    napari_viewer : napari.Viewer, 
    path_to_csv: str, 
    output_dir: str, 
    output_name: str,
    category_col: str = 'sample_type', 
    n_samples = 30,
    tzyx_cols: Tuple[str] = ('frame', 'z_pixels', 'y_pixels', 'x_pixels'),
    id_col:str = 'particle',
    scale: Tuple[float] = (2, 0.5, 0.5), 
    frames: int = 30,
    box_size: int = 60,
    img_channel: Union[int, None] = 2, 
    min_track_length: int = 1,
    #save_samples: bool = True, 
    annotate_now: bool = True, 
    ):
    """
    Sample track segments from CSV. This sampling works by randomly selecting
    a vertex (object coordinate) from a trajectory, then selecting the vertexes
    up to +/- floor(1/2 frames) frames wayy from the selected track vertex. If 
    the track does not extend this far, only the available track will be taken. 
    The 

    parameters
    ----------
    napari_viewer: napari.Viewer
        The existing napari viewer
    path_to_csv: str
        The path storing the info from which to generate the samples. 
        The CSV should have the columns: image_path, labels_path, tracks_path, <category_col>, 
        You can also add an optional n_samples column if you would like to 
        specify how many samples to take from each individual file. Otherwise, 
        the default "n_samples" you've supplied will be used. 
    output_dir: str
        Where will the output be saved?
    output_name: str
        What will output summary files/directories be called?
    n_samples: int
        How many samples to be obtained from each file. Will be overwritten
        if there is a valid integer number in the n_samples colum of the csv.
    tzyx_cols: tuple of str
        What are the names of the columns denoting time (in frames) and coordinate
        positions (in pixels) in the file containing tracks? The order should be:
        t, z, y, x. 
    id_col: str
        What is the name of the column denoting the specific ID for each tracked
        object?
    scale: tuple of float
        size of pixels (e.g., in um) for the z, y, and x coordinates (in that
        order)
    frames: int
        Approximate maximum number of frames of track segment. 
        Max frames = frames (if even) or frames - 1 (if odd)
    box_size: int
        Approximate size of bounding box (in pixels). 
    min_track_len: int
        You can set a minimum track len to include in the search. 
        This can help to eliminate less useful data. Do not use 
        this if you are interested in shorter tracks. 
    image_channel: 
        This denotes the index of the channel from which to get 
        image data (0: channel 1, 1: channel 2, 2: channel 3, 3: channel 4)

    Notes
    -----
    CSV is in the following format:
    0: image_path, labels_path, tracks_path, <category_col>, n_samples*&, sample_path*
    1: <str>, <str>, <str>, <str>, <int>, <str>
    2: ...

    * will be generated when the samples are created
    & only if not provided intially
    """
    _sample_from_csv(napari_viewer, path_to_csv, output_dir, output_name, 
                     category_col, n_samples, tzyx_cols, id_col, scale, 
                     frames, box_size, img_channel, min_track_length, annotate_now)



def _sample_from_csv(
        napari_viewer : napari.Viewer, 
        path_to_csv: str, 
        output_dir: str, 
        output_name: str,
        category_col: str = 'sample_type', 
        n_samples = 30,
        tzyx_cols: Tuple[str] = ('frame', 'z_pixels', 'y_pixels', 'x_pixels'),
        id_col:str = 'particle',
        scale: Tuple[float] = (2, 0.5, 0.5), 
        frames: int = 30,
        box_size: int = 60,
        img_channel: Union[int, None] = 2, 
        min_track_length: int = 10,
        #save_samples: bool = True, 
        annotate_now: bool = True
    ):

    t_col = tzyx_cols[0]
    scale = (1, ) + scale
    instructions = pd.read_csv(path_to_csv)
    if 'n_samples' not in instructions.columns.values:
            instructions['n_samples'] = None
    if 'sample_path' not in instructions.columns.values:
            instructions['sample_path'] = None
    for idx in instructions.index.values:

        # Prepare
        # -------
        image_path = instructions.loc[idx, 'image_path']
        tracks_path = instructions.loc[idx, 'tracks_path']
        name = Path(tracks_path).stem
        labels_path = instructions.loc[idx, 'labels_path']
        labels = zarr.open(labels_path)
        shape = labels.shape 
        poss_n = instructions.loc[idx, 'n_samples']
        if isinstance(poss_n, int) or isinstance(poss_n, float):
             n = poss_n
        else:
             n = n_samples

        # sample
        # ------
        sample = sample_tracks(tracks_path, image_path, shape, 
                               name, n, labels_path=labels_path,
                               frames=frames, box=box_size, id_col=id_col, 
                               time_col=t_col, array_order=tzyx_cols, 
                               scale=scale, min_track_length=min_track_length)
        sample = get_sample_hypervolumes(sample, img_channel)
        # SAVE SAMPLE
        sample_dir = save_sample(output_dir, sample)
        if instructions.loc[idx, 'sample_path'] is None:
            instructions.loc[idx, 'sample_path'] = sample_dir
        else:
            row = instructions.loc[idx, :].copy()
            row['sample_path'] = sample_dir
            instructions = pd.concat([instructions, row]).reset_index()
    instructions.to_csv(path_to_csv)  
    
    # View
    # ----
    if annotate_now:
        _annotate_samples_from_csv(instructions, category_col, output_dir, 
                               output_name, scale[1:], napari_viewer)



@magic_factory(
    path_to_csv={'widget_type': 'FileEdit'}, 
    output_dir={'widget_type': 'FileEdit', 'mode' : 'd'}, 
    scale={'widget_type' : 'LiteralEvalLineEdit'},
)
def annotate_existing_samples(
    napari_viewer : napari.Viewer, 
    path_to_csv: str, 
    output_dir: str, 
    output_name: str,
    category_col: str = 'sample_type', 
    scale: Tuple[float] = (2, 0.5, 0.5), 
    ):
    instructions = pd.read_csv(path_to_csv)
    _annotate_samples_from_csv(instructions, category_col, output_dir, 
                               output_name, scale, napari_viewer)


def _annotate_samples_from_csv(
          instructions, 
          category_col, 
          output_dir, 
          output_name, 
          scale, 
          napari_viewer
          ):
    categories = pd.unique(instructions[category_col])
    sample_paths = {c : [] for c in categories}
    for idx in instructions.index.values:
         c = instructions.loc[idx, category_col]
         sample_paths[c].append(instructions.loc[idx, 'sample_path'])
    sample_dict = prepare_sample_for_annotation(sample_paths)
    save_path = os.path.join(output_dir, output_name + '.csv')
    sv = SampleViewer(sample_dict, save_path, scale=scale, viewer=napari_viewer)
    sv.annotate()
