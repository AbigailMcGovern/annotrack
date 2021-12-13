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

save_path = os.path.join(p, 'output', 'cangrelor_IC2020_vs_DMSO_IC2020.csv')

sv = SampleViewer(sample_dict, save_path)
sv.annotate()

Nov17_0 = [
    'min-len>1,wt=lf_Inhibitor_cohort_2021__ASA__Aspirin_DMSO__ASAD_', 
    'min-len>1,wt=lf_MIPS_analysis_DMSO_20ul', 
    'min-len>1,wt=lf_PAR4_Cohort_Control', 
    'min-len>1,wt=lf_PAR4_Cohort_PAR4--'
]

sample_paths = {
    'asprin_DMSO' : [os.path.join(os.path.join(p, Nov17_0[0]), f) for f in os.listdir(os.path.join(p, Nov17_0[0]))], 
    'MIPS_DMSO' : [os.path.join(os.path.join(p, Nov17_0[1]), f) for f in os.listdir(os.path.join(p, Nov17_0[1]))], 
    'PAR4_control' : [os.path.join(os.path.join(p, Nov17_0[2]), f) for f in os.listdir(os.path.join(p, Nov17_0[2]))], 
    'PAR4--' : [os.path.join(os.path.join(p, Nov17_0[3]), f) for f in os.listdir(os.path.join(p, Nov17_0[3]))], 
}