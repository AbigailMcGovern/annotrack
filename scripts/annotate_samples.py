import os
from annotrack import SampleViewer, prepare_sample_for_annotation

p = '/home/abigail/data/tracking_accuracy'
cond_dirs = [d for d in os.listdir(p) if not d.endswith('.csv')]
cang_dmso = cond_dirs[-2:] # check the directory list

sample_paths = {
    'cangrelor' : [os.path.join(os.path.join(p, cang_dmso[0]), f) for f in os.listdir(os.path.join(p, cang_dmso[0]))], 
    'DMSO' : [os.path.join(os.path.join(p, cang_dmso[1]), f) for f in os.listdir(os.path.join(p, cang_dmso[1]))]
}

sample_dict = prepare_sample_for_annotation(sample_paths)

save_path = os.path.join(p, 'output')

sv = SampleViewer(sample_dict, save_path)
sv.annotate()

