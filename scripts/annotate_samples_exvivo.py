import os
from annotrack import SampleViewer, prepare_sample_for_annotation

mouse_dir = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/tracking/track_samples'
human_dir = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/human/tracking/track_samples'
save_dir = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/tracking_accuracy'

sample_paths = {
    'mouse' : [os.path.join(mouse_dir, f) for f in os.listdir(mouse_dir) if f.endswith('.smpl')], 
    'human' : [os.path.join(human_dir, f) for f in os.listdir(human_dir) if f.endswith('.smpl')]
}

sample_dict = prepare_sample_for_annotation(sample_paths)

save_path = os.path.join(save_dir, '200904_annotations.csv')

sv = SampleViewer(sample_dict, save_path)
sv.annotate()
