import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt



# ---------
# Functions
# ---------

def plot_track_errors(
        df, 
        save_data_dir, 
        save_name, 
        order,
        condition_col='condition_key',
        id_swap_col='id-swaps', 
        false_start_col='false-starts', 
        false_term_col='false-terminations', 
        ):
    plt.rcParams['svg.fonttype'] = 'none' # so the text will be saved with the svg - not curves
    df, id_err_name, dis_err_name = get_error_rates(df, id_swap_col, false_start_col, false_term_col)
    fig, axs = plt.subplots(1, 2, sharey=True)
    sns.barplot(data=df, x=condition_col, y=id_err_name, ax=axs[0], capsize=0.1, order=order)
    sns.barplot(data=df, x=condition_col, y=dis_err_name, ax=axs[1], capsize=0.1, order=order)
    sns.despine(ax=axs[0])
    sns.despine(ax=axs[1])
    fig.set_size_inches(5, 3)
    sp_data = os.path.join(save_data_dir, save_name + '_data.csv')
    df.to_csv(sp_data)
    sp_plots = os.path.join(save_data_dir, save_name + '_plots.svg')
    fig.savefig(sp_plots)
    plt.show()


def get_error_rates(
        df, 
        id_swap_col, 
        false_start_col, 
        false_term_col):
    # convert string lists to python lists
    df[id_swap_col] = df[id_swap_col].apply(eval)
    df[false_start_col] = df[false_start_col].apply(eval)
    df[false_term_col] = df[false_term_col].apply(eval)
    # determine ID swap error rate
    df['no_ID_swap'] = df[id_swap_col].apply(count_unique)
    df['ID swap rate (error/frame)'] = df['no_ID_swap'] / df['frames'] 
    # determine discontinuation error rate
    df['no_false_start'] = df[false_start_col].apply(count_unique)
    df['no_false_term'] = df[false_term_col].apply(count_unique)
    df['no_discont'] = df['no_false_start'] + df['no_false_term']
    df['discontinuation rate (error/frame)'] = df['no_discont'] / df['frames'] 
    return df, 'ID swap rate (error/frame)', 'discontinuation rate (error/frame)'

def count_unique(l):
    l = list(set(l))
    return len(l)

def remove_sudden_z_shift(df):
    sml_df = df[(df['treatment'] == 'MxV_1800is') & (df['frame_start'] < 71) & (df['frame_stop'] > 71)]
    idxs = sml_df.index.values
    df = df.drop(idxs)
    return df

def remove_less_than_10(df):
    df = df[df['track_no_frames'] > 10]
    return df


# --------------
# Paths & config
# --------------
annotation_path = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/tracking_accuracy/200904_in_vivo_annotations_1.csv'
save_data_dir = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/tracking_accuracy'
save_name = '230905_in-vivo_track-accuracy'



# -------
# Compute
# -------
df = pd.read_csv(annotation_path)
#df = remove_sudden_z_shift(df)
df = remove_less_than_10(df)
plot_track_errors(df, save_data_dir, save_name, order=('small', 'medium', 'large'))



# -----------
# Old options
# -----------
#annotation_path = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/tracking_accuracy/200904_annotations.csv'
#save_data_dir = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/tracking_accuracy'
#save_name = '230904_ex-vivo_track-accuracy'

