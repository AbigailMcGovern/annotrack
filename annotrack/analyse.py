from datetime import time
import pandas as pd
import matplotlib.pyplot as plt
from annotrack.sampling import read_sample
import ptitprince as pt
import seaborn as sns

# -------------------
# Collect Annotations
# -------------------

def colate_data(
    samples, 
    save_path=None, 
    annotated_col='annotated', 
    columns=('id-swaps', 'false-terminations', 'false-starts', 'seg-error'), 
    frames_col='frames'
    ):
    '''
    Parameters
    ----------
    samples: list of str or str
        paths to the sample (.smpl) directories in which the data can be found. 
    save_path: str
        optional path at which to save data (.csv)
    annotated_col: str
        name of the column that denotes if the sample has been annotated. 
        The dtype of the column should be bool. 
    time_col: str
        name of the 
    '''
    if isinstance(samples, str):
        samples = [samples, ]
    dfs = []
    for path in samples:
        sample = read_sample(path)
        info = sample['info']
        # get only the annotated info
        info = info[info[annotated_col] ==  True]
        info = info.reset_index(drop=True)
        dfs.append(info)
        #for i in info.index.values:
         #   pair = (info.loc[i, id_col], info.loc[i, time_col])
          #  df = sample[pair]['df']
    cols_list = []
    for df in dfs:
        cols_list.extend(list(df.columns.values))
    cols = list(set(cols_list))
    # make sure every df has every column
    for df in dfs:
        these_cols = list(df.columns.values)
        new_cols = [c for c in cols if c not in these_cols]
        for c in new_cols:
            df[c] = [None, ] * len(df)
    dfs = pd.concat(dfs).reset_index(drop=True)
    if save_path is not None:
        dfs.to_csv(save_path)
    dfs = summarise_list_vars(dfs, columns, frames_col)
    return dfs
    

# ------------
# Process Data
# ------------

def summarise_list_vars(
    df, 
    columns=('id-swaps', 'false-terminations', 'false-starts', 'seg-error'), 
    frames_col='frames'
    ):
    '''
    Provide counts and frequencies based on list variables. These list variables
    denote the frame at which a particular error occured. 
    '''
    df = df.reset_index(drop=True) # now uses range(0, n) indexing
    for col in columns:
        try:
            df = convert_to_list(df, col)
        except TypeError:
            print(df[col])
            raise TypeError
        n_name = 'n_' + col
        f_name = col + '_per-frame'
        df[n_name] = [len(set(ls)) for ls in df[col].values]
        df[f_name] = [df.loc[i, n_name] / df.loc[i, frames_col] for i in df.index.values]
    return df



def convert_to_list(df, col):
    new_vals = []
    for v in df[col].values:
        if isinstance(v, str):
            new_vals.append(eval(v))
        elif isinstance(v, list):
            new_vals.append(v)
        else:
            new_vals.append([])
    df[col] = new_vals
    return df


# ------------
# Analyse Data
# ------------

def raincloud_plot(
    df, 
    title, 
    values_col, 
    group_col, 
    save_path=None, 
    show=True, 
    figsize=(10, 10), 
    fontsize=25
    ):
    plt.rcParams.update({'font.size': fontsize})
    data = {
        group_col : df[group_col].values,  #astype('category').cat.codes, 
        values_col :df[values_col].values, 
    }
    data = pd.DataFrame(data)
    o = 'h'
    pal = 'Set2'
    sigma = .4
    f, ax = plt.subplots(figsize=figsize)
    pt.RainCloud(x=group_col, y=values_col, data=data, palette=pal, bw=sigma,
                 width_viol=.7, ax=ax, orient=o)
    #ax=pt.half_violinplot( x = group_col, y = values_col, data = data, palette = pal, bw = .2, cut = 0.,
                    #  scale = "area", width = .6, inner = None, orient = o)
    #ax=sns.stripplot( x = group_col, y = values_col, data = df, palette = pal, edgecolor = "white",
             #    size = 3, jitter = 1, zorder = 0, orient = o)
    #ax=sns.boxplot( x = group_col, y = values_col, data = df, color = "black", width = .15, zorder = 10,\
          #  showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
           # showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
            #   saturation = 1, orient = o)
    plt.title(title)
    f.savefig(save_path)
    if show:
        plt.show()


# 
if __name__ == '__main__':
    import os
    from scipy.stats import ttest_ind
    #from scipy.stats.t import interval
    # get the paths to the samples we wish to analyse
    p = '/home/abigail/data/tracking_accuracy'
    cond_dirs = [d for d in os.listdir(p) if not d.endswith('.csv')]
    cang_dmso = ['min-len>1,wt=lf_Inhibitor_cohort_2020_Cangrelor', 'min-len>1,wt=lf_Inhibitor_cohort_2020_DMSO'] # check the directory list
    samples = [os.path.join(os.path.join(p, cang_dmso[0]), f) \
                for f in os.listdir(os.path.join(p, cang_dmso[0]))] + \
              [os.path.join(os.path.join(p, cang_dmso[1]), f) \
                for f in os.listdir(os.path.join(p, cang_dmso[1]))]
    samples = [s for s in samples if s.endswith('.smpl')]
    # get the annotated data
    save_path = '/home/abigail/GitRepos/annotrack/untracked/201111_IC20_cangrelor-dmso.csv'
    df = colate_data(samples)
    df.to_csv(save_path)
    f = open('201111_IC20_cangrelor-dmso_ttests.txt', 'w')

    # --------
    # ID SWAPS
    # --------
    # variables for output plot
    save_path = '/home/abigail/GitRepos/annotrack/untracked/201111_IC20_cangrelor-dmso_id-swaps.png'
    title = 'Inhibitor Cohort - 2020: ID Swap Errors'
    values_col = 'id-swaps_per-frame'
    group_col = 'treatment'
    # save and show plot
    raincloud_plot(df, title, values_col, group_col, save_path)
    # do a ttest
    a = df[df['treatment'] == 'DMSO'][values_col].values
    b = df[df['treatment'] == 'Cangrelor'][values_col].values
    res = ttest_ind(a, b)
    print(title)
    print(res)
    f.writelines([title, str(res)])

    # ------------------
    # False Terminations
    # ------------------
    # variables for output plot
    save_path = '/home/abigail/GitRepos/annotrack/untracked/201111_IC20_cangrelor-dmso_false-terminations.png'
    title = 'Inhibitor Cohort - 2020: False Termination Errors'
    values_col = 'false-terminations_per-frame'
    group_col = 'treatment'
    # save and show plot
    raincloud_plot(df, title, values_col, group_col, save_path)
    # do a ttest
    a = df[df['treatment'] == 'DMSO'][values_col].values
    b = df[df['treatment'] == 'Cangrelor'][values_col].values
    res = ttest_ind(a, b)
    print(title)
    print(res)
    f.writelines([title, str(res)])

    # ------------
    # False Starts
    # ------------
    # variables for output plot
    save_path = '/home/abigail/GitRepos/annotrack/untracked/201111_IC20_cangrelor-dmso_false-starts.png'
    title = 'Inhibitor Cohort - 2020: False Start Errors'
    values_col = 'false-starts_per-frame'
    group_col = 'treatment'
    # save and show plot
    raincloud_plot(df, title, values_col, group_col, save_path)
    # do a ttest
    a = df[df['treatment'] == 'DMSO'][values_col].values
    b = df[df['treatment'] == 'Cangrelor'][values_col].values
    res = ttest_ind(a, b)
    print(title)
    print(res)
    f.writelines([title, str(res)])
    f.close()