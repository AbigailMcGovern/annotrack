import pandas as pd
import numpy as np

def find_splits(df, time_col='t', id_col='particle'):
    '''
    Function for checking if IDs are truely all belonging to the same track

    Parameters
    ----------
    df: pd.DataFrame
    time_col: str
    id_col: str

    Returns
    -------
    splits: dict
        Of the form {id : [(<last t before split>, <first t after split>), ...]}
        However, ideally you want to return an empty dict.
    '''
    bincount = np.bincount(df[id_col].values)
    tracked_ids = [ID for ID, num in enumerate(bincount) if num > 1]
    splits = {}
    for ID in tracked_ids:
        iddf = df[df[id_col] == ID]
        iddf = iddf.reset_index(drop=True)
        for i in range(len(iddf)):
            if i == 0:
                pass
            else:
                diff = iddf.loc[i, time_col] - iddf.loc[i - 1, time_col]
                if diff > 2:
                    if splits.get(ID) is None:
                        splits[ID] = [(iddf.loc[i - 1, time_col], iddf.loc[i, time_col]),]
                    else:
                        splits[ID].append((iddf.loc[i - 1, time_col], iddf.loc[i, time_col]))
                    print(f'ID-{ID}: diff at {i} = {diff}')
    return splits


def no_tracks_correct(df, id_col='particle', tl_col='track_length'):
    for ID in df[id_col].unique():
        tdf = df[df[id_col] == ID]
        n_tracks = len(tdf)
        expected = tdf[tl_col].values[0]
        m = f'The number of tracks ({n_tracks}) identified for {ID} was not as expected ({expected})'
        assert n_tracks == expected, m