from datetime import time
import napari
import numpy as np
import os
from numpy.core.numeric import full
from numpy.random import sample
import pandas as pd
from pathlib import Path



def add_track_data(sv):
    viewer = sv.v
    sample = sv.sample
    pair = sv.pairs[sv._i]
    layer_names = [l.name for l in viewer.layers]
    data = sample[pair]['tracks']
    prop = {'track_id': data[:, 0]}
    if 'tracks' not in layer_names:
        viewer.add_tracks(
                          data, 
                          properties=prop,
                          color_by='track_id',
                          name='Tracks', 
                          colormap='viridis', 
                          tail_width=6, 
                          )
    else:
        #i = [i for i, name in enumerate(layer_names) if name == 'Tracks']
        #viewer.layers.pop(i[0])
        #viewer.add_tracks(
         #                 data, 
          #                properties=prop,
           #               color_by='track_id',
            #              name='Tracks', 
             #             colormap='viridis', 
              #            tail_width=6, 
               #           )
        viewer.layers.properties = prop
        viewer.layers.color_by = 'track_id'
        viewer.layers['Tracks'].data = data
        viewer.layers.color_by = 'track_id'


# ------------------
# SampleViewer Class
# ------------------


class SampleViewer:
    def __init__(self, 
                 sample, 
                 save_path=None, 
                 mode='Track',
                 id_swap_col='id-swaps', 
                 false_term_col='false-terminations',
                 false_start_col='false-starts',
                 t_start='frame_start', 
                 scale=(2, .5, .5), 
                 viewer=None,
                 ):
        '''
        View and annotate a sample

        Parameters
        ----------
        sample: dict
            OPTION 1:
                Sample information dict the form 
                {
                    'info' : pd.DataFrame,
                    'coord_info' : {
                        'array_order' : (<col name corresponding to coords>, ...)
                        'time_col' : str, 
                        'id_col' : str
                }
                    (ID, t) : {
                        'df' : pd.DataFrame,
                        'df_path' : 'path/to/info/df.csv', 
                        'info_path' : 'path/to/info/df.csv', 
                        'b_box': {
                            '<coord name>' : slice, 
                            ... 
                            }
                        'image' : <array like>, 
                        'labels' : <array like>
                        },
                ...}
            OPTION 2:
                dict of sample dicts (as above) with keys corresponding to 
                different conditions that you wish to get the data to compare

        '''
        if 'coord_info' in list(sample.keys()):
            sample = {0 : sample}

        # Prepare the image and or label data
        self.scale = scale
        self.array_order = {key: sample[key]['coord_info']['array_order'] for key in sample.keys()}
        self.time_col = {key : sample[key]['coord_info']['time_col'] for key in sample.keys()}
        self.id_col = {key: sample[key]['coord_info']['id_col'] for key in sample.keys()}
        self.keys = list(sample.keys())

        # sample info
        self.sample = sample
        self.info = {key : sample[key]['info'] for key in sample.keys()}
        self.id_swap_col = id_swap_col # add FPs
        self.false_term_col = false_term_col # add FNs
        self.false_start_col = false_start_col
        self.seg_err_col = 'seg-error'
        self.t_start = t_start
        
        # get the order of conditions of samples 
        key_list = []
        for key in sample.keys():
            n = len(sample[key]['info'])
            keys = [key, ] * n
            key_list.extend(keys)
        key_list = np.array(key_list)
        key_list = np.random.choice(key_list, size=(len(key_list), ), replace=False)

        # get the order of samples in a condition
        sample_order = {}
        for key in sample.keys():
            tup_keys = sample[key]['info']['key'].values
            idxs = sample[key]['info'].index.values
            idxs = np.random.choice(idxs, size=(len(idxs), ), replace=False)
            sample_order[key] = [tup_keys[i] for i in idxs]

        # get the list with which to obtain samples in randomised order
        samples_list = []
        for k in key_list:
            tk = sample_order[k].pop(0)
            samples_list.append((k, tk))
        self.samples_list = samples_list

        # set the index to zero to start at the start of this list
        self._i = 0

        # add annotation cols
        self._add_annotation_cols()
        
        # initialise viewer attr
        self.v = viewer
        self.v_source = 'self'
        if viewer is not None:
            message = (
            """Mark samples as correct or incorrect by pressing y or n,
            respectively.
            
            To indicate that an ID swap has occurred, navigate to the first
            frame of the new track and press i.
            
            To indicate that a false termination or start has occurred,
            navigate to the frame where it occurs and press t or Shift-t.
            
            To indicate that a tracking error (any of the above) is caused by
            a segmentation error, press s.
            
            These error types will be saved to the info.csv file in the
            sample.smpl directory.
            
            To navigate to the next sample, use 2, and use 1 to navigate to the
            previous sample.
            """
            )
            print(m)
            self.v_source = 'preexising'

        # this lil' guys not hurting anything... or helping... so sue me
        self.mode = mode

        # will save dataframe here if specified
        self.save_path = save_path

        # get the info data frame with correct indexes
        info = self._get_full_info()
        self.info = info


    def _add_annotation_cols(self):
        for key in self.keys:
            add_empty_col(self.info[key], 'correct', None)
            add_empty_col(self.info[key], self.id_swap_col, [])
            add_empty_col(self.info[key], self.false_term_col, [])
            add_empty_col(self.info[key], self.false_start_col, [])
            add_empty_col(self.info[key], self.seg_err_col, [])


    def _get_full_info(self):
        full_info = []
        for keys in self.samples_list:
            info = self.info[keys[0]]
            if 'condition_key' not in info.columns.values:
                info['condition_key'] = [keys[0], ] * len(info)
            row = info[info['key'] == keys[1]]
            full_info.append(row)
        full_info = pd.concat(full_info)
        if 'Unnamed: 0' in full_info.columns.values:
            full_info = full_info.drop(columns=['Unnamed: 0', ])
        if 'level_0' in full_info.columns.values:
            full_info = full_info.drop(columns=['level_0', ])
        full_info = full_info.reset_index(drop=True)
        full_info.to_csv(self.save_path)
        full_info = pd.read_csv(self.save_path)
        if 'Unnamed: 0' in full_info.columns.values:
            full_info = full_info.drop(columns=['Unnamed: 0', ])
            full_info['key'] =  full_info['key'].apply(eval)
            full_info[self.id_swap_col] =  full_info[self.id_swap_col].apply(eval)
            full_info[self.false_start_col] =  full_info[self.false_start_col].apply(eval)
            full_info[self.false_term_col] =  full_info[self.false_term_col].apply(eval)
            full_info[self.seg_err_col] =  full_info[self.seg_err_col].apply(eval)
        return full_info

    
    def save_info(self):
        if self.save_path is not None:
            self.info.to_csv(self.save_path)


    @property
    def i(self):
        return self._i


    @i.setter
    def i(self, i):
        if i < len(self.samples_list) or i > 0:
            self._i = i
        else:
            print('invalid sample index')


    def annotate(self):
        self._show_sample()
        self.v.bind_key('y', self.yes)
        self.v.bind_key('n', self.no)
        self.v.bind_key('2', self.next)
        self.v.bind_key('1', self.previous)
        # on the frame after a swap assign the frame number
            # to the ID swap list
        self.v.bind_key('i', self.id_swap)
        self.v.bind_key('t', self.false_termination)
        self.v.bind_key('Shift-t', self.false_start)
        self.v.bind_key('s', self.segmentation_error)
        if self.v_source == 'self':
            napari.run()


    def _show_sample(self):
        # initialise the viewer if it doesnt exist
        if self.v is None:
            self.v = napari.Viewer()
            print('---------------------------------------------------------')
            print(f"Showing sample 1/{len(self.samples_list)}")
            print('---------------------------------------------------------')
            m = 'Mark samples as correct or incorrect by pressing y or n, \n'
            m = m + 'repspectively. \n'
            m = m + '---------------------------------------------------------\n'
            m = m + 'To indicate that an ID swap has has occured, at a given \n'
            m = m + 'point in time press i.\n'
            m = m + '---------------------------------------------------------\n'
            m = m + 'To indicate that a false termination or start has occured\n'
            m = m + 'at a given point in time, use t or Shift-t, respectively.\n'
            m = m + '---------------------------------------------------------\n'
            m = m + 'To indicate that a tracking error at a given time point\n'
            m = m + 'is associated with a segmentation error, press s.\n'
            m = m + '---------------------------------------------------------\n'
            m = m + 'Pressing i, t or Shift-t will record the frame number at\n'
            m = m + 'which the error occured. This will be saved into the \n'
            m = m + 'info csv in the sample.smpl directory\n'
            m = m + '---------------------------------------------------------\n'
            m = m + 'To navagate to the next sample, use 2. To move to the \n'
            m = m + 'previous sample, use 1. (a keyboard agnostic approach)\n'
            m = m + '---------------------------------------------------------\n'
            print(m)
        # get the names of layers currently attached to the viewer
        layer_names = [l.name for l in self.v.layers]
        prop = {}
        # get keys with which to obtain image, labels, and tracks
        keys = self.samples_list[self.i]
        # add volumes
        scale = self.scale
        for name in ['image', 'labels']:
            data = np.array(self.sample[keys[0]][keys[1]][name])
            #print(data.shape)
            if name not in layer_names:
                if name == 'image':
                    self.v.add_image(data, 
                                     name=name, 
                                     scale=scale, 
                                     colormap='bop orange')
                elif name == 'labels':
                    self.v.add_labels(data, 
                                      name=name, 
                                      scale=scale)
            else:
                self.v.layers[name].data = data
        # add tracks
        tracks = np.array(self.sample[keys[0]][keys[1]]['tracks'])
        prop = self.sample[keys[0]][keys[1]]['df'].to_dict(orient='list')
        #print(tracks.shape)
        if 'tracks' not in layer_names:
            #prop = {'track_id': tracks[:, 0]}
            self.v.add_tracks(tracks, properties=prop, color_by=self.time_col[keys[0]], scale=scale,
                              name='tracks', colormap='viridis', tail_width=6)
        else:
            self.v.layers.color_by = 'track_id'
            self.v.layers.properties = None
            self.v.layers['tracks'].data = tracks
            self.v.layers['tracks'].properties = prop
            self.v.layers['tracks'].color_by = self.time_col[keys[0]]
            #self.v.layers.pop('tracks')
            #self.v.add_tracks(tracks, properties=prop, color_by=self.time_col[keys[0]], scale=scale,
                    #          name='tracks', colormap='viridis', tail_width=6)
        #self.v.layers['tracks'].display_id = True 
        


    # For Key Binding
    #----------------

    def yes(self, viewer):
        """
        Annotate as correct. Will be bound to key 'y'
        """
        self._annotate(1)
        self._check_ann_status()
    
    def no(self, viewer):
        """
        Annotate as incorrect. Will be bound to key 'n'
        """
        self._annotate(0)
        self._check_ann_status()

    def next(self, viewer):
        """
        Move to next pair. Will be bound to key '1'
        """
        print(f'next')
        self._next()

    def previous(self, viewer):
        """
        Move to previous pair. Will be bound to key '2'
        """
        print(f'previous')
        self._previous()
    

    def id_swap(self, viewer):
        """
        Get the time frame at which ID swap happended
        """
        t = self._get_current()
        keys = self.samples_list[self._i]
        row = self.info[(self.info['condition_key'] == keys[0]) & (self.info['key'] == keys[1])]
        idx = row.index.values[0]
        row.loc[idx, self.id_swap_col] = row.loc[idx, self.id_swap_col].append(t)
        print(f'ID swap recorded at time {t}')
        self.save_info()
        # save to original
        #self._add_data_to_orig_info(t, self.id_swap_col, [])

    
    def false_termination(self, viewer):
        """
        Get the time frame at which false termination occured
        """
        t = self._get_current()
        keys = self.samples_list[self._i]
        idx = self.info[(self.info['condition_key'] == keys[0]) & (self.info['key'] == keys[1])].index.values[0]
        self.info.loc[idx, self.false_term_col].append(t)
        print(f'False termination recorded at time {t}')
        self.save_info()
        # save to original
        #self._add_data_to_orig_info(t, self.false_term_col, [])


    def false_start(self, viewer):
        """
        Get the time frame at which false termination occured
        """
        t = self._get_current()
        keys = self.samples_list[self._i]
        idx = self.info[(self.info['condition_key'] == keys[0]) & (self.info['key'] == keys[1])].index.values[0]
        self.info.loc[idx, self.false_start_col].append(t)
        print(f'False start recorded at time {t}')
        self.save_info()
        # save to original
        #self._add_data_to_orig_info(t, self.false_start_col, [])


    def segmentation_error(self, viewer):
        '''
        Get the time frame at which there was a segmentation error
        '''
        t = self._get_current()
        keys = self.samples_list[self._i]
        idx = self.info[(self.info['condition_key'] == keys[0]) & (self.info['key'] == keys[1])].index.values[0]
        self.info.loc[idx, self.seg_err_col].append(t)
        print(f'Segmentation-based error recorded at time {t}')
        self.save_info()
        # save to original
        #self._add_data_to_orig_info(t, self.seg_err_col, [])


    # Key binding helpers
    # -------------------

    def _next(self):
        penultimate = len(self.samples_list) - 2
        if self._i <= penultimate:
            self._i += 1
            self._show_sample()
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.samples_list)}")
        else:
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.samples_list)}")
            self._check_ann_status()
            print("To navagate to prior samples press the 1 key")

    
    def _previous(self):
        if self._i >= 1:
            self._i -= 1
            self._show_sample()
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.samples_list)}")
        else:
            print('---------------------------------------------------------')
            print(f"Showing sample {self._i + 1}/{len(self.samples_list)}")
            self._check_ann_status()
            print("To navagate to the next sample press the 2 key")


    def _annotate(self, ann):
        word = lambda a : 'correct' if a == 1 else 'incorrect'
        # save to overall data frame
        self.info.at[self.i, 'correct'] = ann
        print(f'sample at index {self.i} was marked {word(ann)}')
        self._get_score()
        self.save_info()
        # save to original info dataframe for the sample (image sample .smpl)
        #self._add_data_to_orig_info(ann, 'correct')
        # record that this track has been annotated 
        #self._add_data_to_orig_info(True, 'annotated')
        # (it won't be obtained when prepping data with sample_management.py unless otherwise specifed)
        self._next()


    def _add_data_to_orig_info(self, data, col, empty=None):
        keys = self.samples_list[self.i]
        info_path = self.sample[keys[0]][keys[1]]['info_path']
        print(info_path)
        info = pd.read_csv(info_path)
        cols = info.columns.values
        if col not in cols:
            add_empty_col(info, col, empty)
            pls_eval = False
        else:
            pls_eval = True
        #print(cols)
        #info['key'] = info['key'].apply(eval)
        #row = info[(info['condition_key'] == keys[0]) & (info['key'] == keys[1])]
        row = info[(info[self.id_col[keys[0]]] == keys[1][0]) & (info[self.time_col[keys[0]]] == keys[1][1])]
        assert len(row) == 1
        idx = row.index[0]
        if empty is None:
            info.loc[idx, col] = data
        elif isinstance(empty, list):
            if pls_eval:
                l = row[col].values[0]
                #print('l', l, type(l))
                l = eval(l)
            else:
                l = []
            l.append(data)
            info.loc[idx, col] = l
            #print(l)
            #print(info.loc[idx, col])
            #print(info.loc[0, col])
        to_drop = [col for col in info.columns.values if col.startswith('Unnamed:')]
        info = info.drop(to_drop, axis=1)
        info.to_csv(info_path)


    def _get_current(self):
        current = self.v.dims.current_step[0]
        t0 = self.info[self.t_start][self._i]
        t = current + t0
        return t


    def _check_ann_status(self):
        out = self.info['correct'].values
        if None not in out:
        #    print('---------------------------------------------------------')
        #    print('All tracks have been annotated')
        #    print(f'Final score is {self.score * 100} %')
            pass
        elif None in out and self._i + 1 > len(out):
            not_done = []
            for i, o in enumerate(out):
                if o == None:
                    not_done.append(i)
            if len(not_done) == 1:
                print('---------------------------------------------------------')
                print(f'Track {not_done[0]} of {len(out)} has not yet been annotated')
            if len(not_done) > 1:
                print('---------------------------------------------------------')
                print(f'Tracks {not_done} of {len(out)} have not yet been annotated')


    def _get_score(self):
            """
            Get the proportion of correctly scored tracks
            """
            self.score = np.divide(self.info['correct'].sum(), len(self.info))



def add_empty_col(df, col, empty):
    df[col] = [empty, ] * len(df)


if __name__ == '__main__':
    from sample_management import prepare_sample_for_annotation
    samples = {
        'img_0' : ['/home/abigail/data/plateseg-training/timeseries_seg/problem-children/timeseries_seg problem-children/210601_IVMTR114_Inj1_ASA_exp3_rtracks_211013_125430_n=34.smpl', 
        '/home/abigail/data/plateseg-training/timeseries_seg/problem-children/timeseries_seg problem-children/210601_IVMTR114_Inj3_ASA_exp3_rtracks_211013_125448_n=33.smpl'], 
        'img_1' : ['/home/abigail/data/plateseg-training/timeseries_seg/problem-children/timeseries_seg problem-children/210601_IVMTR114_Inj2_ASA_exp3_rtracks_211013_125439_n=33.smpl', 
         '/home/abigail/data/plateseg-training/timeseries_seg/problem-children/timeseries_seg problem-children/210601_IVMTR114_Inj3_ASA_exp3_rtracks_211013_125448_n=33.smpl']
    }
    new_samples = prepare_sample_for_annotation(samples, n_each=15)
    keys = list(new_samples.keys())
    #print(keys)
    #print(new_samples[keys[0]]['coord_info'])
    s_keys = list(new_samples[keys[0]].keys())
    #print(s_keys)
    tups = [key for key in s_keys if isinstance(key, tuple)]
    s = new_samples[keys[0]][tups[0]]
    #print(list(s.keys()))
    #print(s['image'].shape)
    #print(s['labels'].shape)
    save_path = '/home/abigail/data/plateseg-training/timeseries_seg/problem-children/timeseries_seg problem-children/annotations.csv'
    sample_viewer = SampleViewer(new_samples, save_path)
    sample_viewer.annotate()

