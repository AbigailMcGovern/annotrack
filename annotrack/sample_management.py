import numpy as np
from numpy.core.fromnumeric import size
from numpy.random import sample
import pandas as pd
from sampling import read_sample


def prepare_sample_for_annotation(samples, n_each=30, use_annotated=False):
    '''
    Obtain a sample of 
    
    Parameters
    ----------
    samples: dict
        For the form {"<name of condition>" : ['path/to/a/sample.smpl', ...], ...}
    n_each: int
        How many samples from each condition do you want to look at?
    use_annotated: bool
        Should we load already annotated samples?
    '''
    sample_dict = {}
    for key in samples.keys():
        cond_samples = []
        for sample_path in samples[key]:
            sample = read_sample(sample_path)
            cond_samples.append(sample)
        sample = combine_samples(cond_samples)
        if not use_annotated:
            triplets = []
            for idx in sample['info'].index.values:
                if sample['info'].loc[idx, 'annotated'] == False:
                    triplets.append(sample['info'].loc[idx, 'key'])
        else:
            triplets = sample['info']['key'].values
        if n_each is not None and n_each <= len(triplets):    
            #print(len(triplets))        
            # get the samples to remove from the dict 
            poss_idx = sample['info'].index.values
            idxs = np.random.choice(poss_idx, size=(len(triplets) - n_each), replace=False)
            #print(idxs)
            inc_trip = []
            exc_trip = []
            for i, trip in enumerate(triplets):
                if i in idxs:
                    del sample[trip]
                    exc_trip.append(trip)
                else:
                    inc_trip.append(trip)
            drop_idxs = []
            for idx in sample['info'].index.values:
                if sample['info'].loc[idx, 'key'] in exc_trip:
                    drop_idxs.append(idx)
            sample['info'] = sample['info'].drop(list(drop_idxs))
            #print(inc_trip)
            #print(list(sample['info']['key'].values))
            sample['info'] = sample['info'].reset_index(drop=True)
        sample_dict[key] = sample
    return sample_dict
            



def combine_samples(sample_list):
    sample = {
        'image_path' : None, 
        'labels_path' : None, 
        'sample_type' : None,
        'coord_info' : None
    }
    # check the coords info is the same and add
    coords_info = [s['coord_info'] for s in sample_list]
    for i in range(1, len(coords_info)):
        assert coords_info[0] == coords_info[i], 'coord_info must be identical to combine samples'
    sample['coord_info'] = coords_info[0]
    # check the sample type is the same and add
    sample_type = [s['sample_type'] for s in sample_list]
    for i in range(1, len(sample_type)):
        assert sample_type[0] == sample_type[i], 'sample_type must be identical to combine samples'
    sample['sample_type'] = sample_type[0]
    # now get the tuple keys for combined sample. Three this time (<ID>, <time>, <img-idx>).
    triplets = []
    for i, s in enumerate(sample_list):
        pairs = get_pairs_from_df(s)
        for j, pair in enumerate(pairs):
            triplet = pair + (i, )
            #print(pair, triplet)
            sample[triplet] = s[pair]
            triplets.append(triplet)
    # get the info (the contains the 'annotated' column)
    info = [s['info'] for s in sample_list]
    info = pd.concat(info)
    info['key'] = triplets
    info = info.reset_index(drop=True)
    sample['info'] = info
    #print(info.head())
    #print(triplets)
    #print(list(sample.keys()))
    #print(list(sample[triplets[0]].keys()))
    return sample


def get_pairs_from_df(sample):
    id_col = sample['coord_info']['id_col']
    time_col = sample['coord_info']['time_col']
    info = sample['info'].reset_index(drop=True)
    pairs = []
    for i in range(len(info)):
        pair = (info.loc[i, id_col], info.loc[i, time_col])
        pairs.append(pair)
    return pairs


if __name__ == '__main__':
    samples = {
        'img_0' : ['/home/abigail/data/plateseg-training/timeseries_seg/problem-children/timeseries_seg problem-children/210601_IVMTR114_Inj1_ASA_exp3_n=34.smpl', ], 
        'img_1' : ['/home/abigail/data/plateseg-training/timeseries_seg/problem-children/timeseries_seg problem-children/210601_IVMTR114_Inj2_ASA_exp3_n=33.smpl', ]
    }
    new_samples = prepare_sample_for_annotation(samples, n_each=15)
    keys = list(new_samples.keys())
    print(keys)
    print(new_samples[keys[0]]['coord_info'])
    s_keys = list(new_samples[keys[0]].keys())
    print(s_keys)
