from NGSIM2HighD import NGSIM2HighD

ngsim_dataset = "./ngsim/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv"
export_dir = "./exported_ngsim/"
locations = ['us-101', 'i-80']
#TODO: Test with other locations
converter = NGSIM2HighD(ngsim_dataset, export_dir, locations)
converter.convert_tracks_info()