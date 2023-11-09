import os
from annotrack import SampleViewer, prepare_sample_for_annotation

small_dir = '/Users/abigailmcgovern/Data/platelet-analysis/track-accuracy/min-len>1,wt=lf_Inhibitor_cohort_2020_Cangrelor'
medium_dir = '/Users/abigailmcgovern/Data/platelet-analysis/track-accuracy/min-len>1,wt=lf_Inhibitor_cohort_2020_Saline'
large_dir = '/Users/abigailmcgovern/Data/platelet-analysis/track-accuracy/min-len>1,wt=lf_Inhibitor_cohort_2020_DMSO'
save_dir = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/tracking_accuracy'

sample_paths = {
    'small' : [os.path.join(small_dir, f) for f in os.listdir(small_dir) if f.endswith('.smpl')], 
    'medium' : [os.path.join(medium_dir, f) for f in os.listdir(medium_dir) if f.endswith('.smpl')], 
    'large' : [os.path.join(large_dir, f) for f in os.listdir(large_dir) if f.endswith('.smpl')]
}

sample_dict = prepare_sample_for_annotation(sample_paths)

save_path = os.path.join(save_dir, '200904_in_vivo_annotations_1.csv')

sv = SampleViewer(sample_dict, save_path, t_start='t_start')
sv.annotate()
