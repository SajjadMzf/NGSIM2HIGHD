import os
import pandas
import HighD_Columns as HC 
import NGSIM_Columns as NC 
class NGSIM2HighD:
    def __init__(self,ngsim_csv_file_dir, ngsim_export_dir, locations):
        self.ngsim_csv_file_dir = ngsim_csv_file_dir
        self.ngsim_export_dir = ngsim_export_dir
        self.locations = locations
        df = pandas.read_csv(self.ngsim_csv_file_dir)
        # TODO: Assert the loaded dataset has the required columns
        self.ngsim = []
        for location in self.locations:
            self.ngsim.append(df[df[NC.LOCATION]==location])

    def convert_tracks_info(self):
        """ This method applies following changes:
            1. Delete Unneccessary Coloumns:          
            2. Modify Existing Coloumns:
            3. Compute New Coloumns: 
        """
        for i, location in enumerate(self.locations):
            
            
            # 1. Delete
            self.ngsim[i] = self.ngsim[i].drop(
                columns = [
                    NC.O_ZONE, 
                    NC.D_ZONE, 
                    NC.INT_ID, 
                    NC.SECTION_ID, 
                    NC.DIRECTION, 
                    NC.MOVEMENT, 
                    NC.GLOBAL_X, 
                    NC.GLOBAL_Y, 
                    NC.GLOBAL_TIME,
                    NC.LOCATION, 
                    NC.TOTAL_FRAME 
                    ])
            # 2. Modify 
            self.ngsim[i] = self.ngsim[i].sort_values([NC.ID, NC.FRAME], ascending=[1,1])
            #self.ngsim[i].to_csv(location + '_data.csv')
            duplicate_rows = self.ngsim[i].duplicated()
            duplicate_rows.to_csv(location + '_duplicate.csv')
            # 3. Compute

    def convert_static_info(self):
        # TODO: Export static info from NGSIM
        return 0
    def convert_meta_info(self):
        # TODO: Export meta info from NGSIM
        return 0
    def save(self):
        return 0
