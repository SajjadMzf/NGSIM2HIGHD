from NGSIM2HighD import NGSIM2HighD
import HighD_Columns as HC 
import NGSIM_Columns as NC 
ngsim_dataset_dir =  "../../Dataset/NGSIM/Traj_data/"
ngsim_dataset_files = ['trajectories-0400-0415.csv', 
            'trajectories-0500-0515.csv',
            'trajectories-0515-0530.csv',
            'trajectories-0750am-0805am.csv',
            'trajectories-0805am-0820am.csv',
            'trajectories-0820am-0835am.csv']

converter = NGSIM2HighD(ngsim_dataset_dir, ngsim_dataset_files)
converter.convert_tracks_info()
converter.convert_meta_info()
converter.convert_static_info()