import annotrack
import os
from importlib import reload

p = '/home/abigail/data/tracking_accuracy'
cond_dirs = [d for d in os.listdir(p) if not d.endswith('.csv')]
cang_dmso = ['min-len>1,wt=lf_Inhibitor_cohort_2020_Cangrelor', 'min-len>1,wt=lf_Inhibitor_cohort_2020_DMSO'] # check the directory list

samples = [os.path.join(os.path.join(p, cang_dmso[0]), f) \
            for f in os.listdir(os.path.join(p, cang_dmso[0]))] + \
          [os.path.join(os.path.join(p, cang_dmso[1]), f) \
            for f in os.listdir(os.path.join(p, cang_dmso[1]))]
samples = [s for s in samples if s.endswith('.smpl')]

df = annotrack.analyse.colate_data(samples)