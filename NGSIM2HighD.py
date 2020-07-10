import os
import pandas
import numpy as np
import HighD_Columns as HC 
import NGSIM_Columns as NC 
class NGSIM2HighD:
    def __init__(self,ngsim_csv_file_dir, ngsim_export_dir, files):
        self.ngsim_csv_file_dir = ngsim_csv_file_dir
        self.ngsim_export_dir = ngsim_export_dir
        self.files = files
        self.ngsim = []
        

    def save_locations(self):
        # TODO: Assert the loaded dataset has the required columns
        df = pandas.read_csv(self.ngsim_csv_file_dir)
        for i, location in enumerate(self.files):
            self.ngsim.append(df[df[NC.LOCATION]==location])
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
                                NC.FRAME,
                                NC.LOCATION, 
                                NC.TOTAL_FRAME,
                                #NC.PRECEDING_ID,
                                #NC.FOLLOWING_ID,
                                ])         
            # NGSIM has some duplicate rows need to be dropped
            self.ngsim[i] = self.ngsim[i].drop_duplicates()
            self.ngsim[i] = self.ngsim[i].sort_values([NC.ID, NC.GLOBAL_TIME], ascending=[1,1])
            self.ngsim[i].to_csv(location+ ".csv", index = False)
    def load_locations(self):
        self.ngsim = []
        for location in self.files:
            self.ngsim.append(pandas.read_csv(location+'.csv'))

    def convert_tracks_info(self):
        """ This method applies following changes:
            1. Delete Unneccessary Coloumns:          
            2. Modify Existing Coloumns:
            3. Compute New Coloumns: 
        """
        for i, location in enumerate(self.files):  
            ngsim_columns = self.ngsim[i].columns
            ngsim_array = self.ngsim[i].to_numpy()
            NC_dict = {}
            for i,c in enumerate(ngsim_columns):
                NC_dict[c] = i
            assert(len(ngsim_columns)==14)
            
            # Transformations
            ngsim_array = self.transform_track_features(ngsim_array, NC_dict)
            ngsim_array, SVC_dict = self.transform_frame_features(ngsim_array, NC_dict)
            
            highD_columns = [None]* (len(ngsim_columns) + len(SVC_dict))
            # Untransformed Columns
            highD_columns[NC_dict[NC.CLASS]] = NC.CLASS
            highD_columns[NC_dict[NC.VELOCITY]] = NC.VELOCITY # Note: Velocity is changed from feet/s to m/s
            highD_columns[NC_dict[NC.ACCELERATION]] = NC.ACCELERATION # Note: Acceleration is changed from feet/s^2 to m/s^2 
            highD_columns[NC_dict[NC.PRECEDING_ID]] = NC.PRECEDING_ID 
            highD_columns[NC_dict[NC.FOLLOWING_ID]] = NC.FOLLOWING_ID 
            # Transformed Columns
            highD_columns[NC_dict[NC.ID]] = HC.TRACK_ID
            highD_columns[NC_dict[NC.GLOBAL_TIME]] = HC.FRAME
            highD_columns[NC_dict[NC.X]] = HC.Y # NC.X = HC.Y
            highD_columns[NC_dict[NC.Y]] = HC.X # NC.Y = HC.X
            highD_columns[NC_dict[NC.LENGTH]] = HC.WIDTH # NC.LENGTH = HC.WIDTH
            highD_columns[NC_dict[NC.WIDTH]] = HC.HEIGHT # NC.WIDTH = HC.HEIGHT
            highD_columns[NC_dict[NC.DHW]] = HC.DHW
            highD_columns[NC_dict[NC.THW]] = HC.THW
            highD_columns[NC_dict[NC.LANE_ID]] = HC.LANE_ID
            
            # Added Columns
            highD_columns[len(ngsim_columns) + SVC_dict[HC.PRECEDING_ID]] = HC.PRECEDING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.FOLLOWING_ID]] = HC.FOLLOWING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.LEFT_PRECEDING_ID]] = HC.LEFT_PRECEDING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.LEFT_ALONGSIDE_ID]] = HC.LEFT_ALONGSIDE_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.LEFT_FOLLOWING_ID]] = HC.LEFT_FOLLOWING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.RIGHT_PRECEDING_ID]] = HC.RIGHT_PRECEDING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.RIGHT_ALONGSIDE_ID]] = HC.RIGHT_ALONGSIDE_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.RIGHT_FOLLOWING_ID]] = HC.RIGHT_FOLLOWING_ID
            
            # To dataframe
            transformed_ngsim = pandas.DataFrame(data = ngsim_array, columns = highD_columns)
            transformed_ngsim = transformed_ngsim.sort_values([HC.TRACK_ID, HC.FRAME], ascending=[1,1])
            transformed_ngsim.to_csv(location+'_transformed.csv', index=False)

    def transform_frame_features(self, ngsim_data, NC_dict, logging = True):
        """
        1. Divide Global Time by 100
        2. transform from feet to meter. 
        3. Reverse the order of Lane IDs
        4. Extract vehicle IDs of surrounding vehicles.
        """
        SVC_dict = {
            HC.PRECEDING_ID:0,
            HC.FOLLOWING_ID:1,
            HC.LEFT_PRECEDING_ID:2,
            HC.LEFT_ALONGSIDE_ID:3,
            HC.LEFT_FOLLOWING_ID:4,
            HC.RIGHT_PRECEDING_ID:5,
            HC.RIGHT_ALONGSIDE_ID:6,
            HC.RIGHT_FOLLOWING_ID:7
        }

        sorted_ind = np.argsort(ngsim_data[:,NC_dict[NC.GLOBAL_TIME]])
        ngsim_data = ngsim_data[sorted_ind]
        
        # Global Time => Frame
        ngsim_data[:,NC_dict[NC.GLOBAL_TIME]] = ngsim_data[:,NC_dict[NC.GLOBAL_TIME]]/100
        ngsim_data[:,NC_dict[NC.GLOBAL_TIME]] = ngsim_data[:,NC_dict[NC.GLOBAL_TIME]] - min(ngsim_data[:,NC_dict[NC.GLOBAL_TIME]])+1
        # Feet => meter
        ngsim_data[:,NC_dict[NC.X]] = 0.3048 * ngsim_data[:,NC_dict[NC.X]]
        ngsim_data[:,NC_dict[NC.Y]] = 0.3048 * ngsim_data[:,NC_dict[NC.Y]]
        ngsim_data[:,NC_dict[NC.LENGTH]] = 0.3048 * ngsim_data[:,NC_dict[NC.LENGTH]]
        ngsim_data[:,NC_dict[NC.WIDTH]] = 0.3048 * ngsim_data[:,NC_dict[NC.WIDTH]]
        ngsim_data[:,NC_dict[NC.VELOCITY]] = 0.3048 * ngsim_data[:,NC_dict[NC.VELOCITY]]
        ngsim_data[:,NC_dict[NC.ACCELERATION]] = 0.3048 * ngsim_data[:,NC_dict[NC.ACCELERATION]]
        ngsim_data[:,NC_dict[NC.DHW]] = 0.3048 * ngsim_data[:,NC_dict[NC.DHW]]
        # Change order of lane numbers
        ngsim_data[:,NC_dict[NC.LANE_ID]] = max(ngsim_data[:,NC_dict[NC.LANE_ID]])+1- ngsim_data[:,NC_dict[NC.LANE_ID]]

        augmented_features = np.zeros((ngsim_data.shape[0], 8))
        all_times = sorted(list(set(ngsim_data[:,NC_dict[NC.GLOBAL_TIME]])))
        max_itr = len(all_times)
        for itr, g_time in enumerate(all_times):
            if logging and itr%100 == 0:
                print('Processing: ', itr, 'out_of: ', max_itr)
            
            
            selected_ind = ngsim_data[:,NC_dict[NC.GLOBAL_TIME]] == g_time
            cur_data = ngsim_data[selected_ind]
            #print("Current Vehicles:{} at: {}".format(cur_data[:,NC_dict[NC.ID]], g_time))
            #exit()
            cur_aug_features = augmented_features[selected_ind]
            num_veh = cur_data.shape[0]
            
            for veh_itr in range(num_veh):
                cur_lane = cur_data[veh_itr, NC_dict[NC.LANE_ID]]
                cur_y = cur_data[veh_itr, NC_dict[NC.Y]]
                cur_length = cur_data[veh_itr, NC_dict[NC.LENGTH]]
                #print("ID: {}, Y: {}, Lane: {}".format(cur_data[veh_itr, NC_dict[NC.ID]], cur_y, cur_lane))
                mask = [True]* num_veh
                mask[veh_itr] = False
                cur_data_minus_ev = cur_data[mask]
                
                cur_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == cur_lane)
                left_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == (cur_lane+1))
                right_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == (cur_lane-1))
                preceding_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.Y]]- cur_data_minus_ev[:,NC_dict[NC.LENGTH]] > cur_y)
                following_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.Y]] < cur_y-cur_length)
                alongside_sv_ind = \
                np.logical_and((cur_data_minus_ev[:,NC_dict[NC.Y]] >= (cur_y-cur_length)),
                 ((cur_data_minus_ev[:,NC_dict[NC.Y]]-cur_data_minus_ev[:,NC_dict[NC.LENGTH]]) <= cur_y))

                #pv_id
                pv_cand_data = cur_data_minus_ev[np.logical_and(preceding_sv_ind, cur_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.PRECEDING_ID]] = \
                pv_cand_data[np.argmin(pv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(preceding_sv_ind, cur_lane_sv_ind)) == True else 0
                
                #fv_id
                fv_cand_data = cur_data_minus_ev[np.logical_and(following_sv_ind, cur_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.FOLLOWING_ID]] = \
                fv_cand_data[np.argmax(fv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(following_sv_ind, cur_lane_sv_ind)) == True else 0

                #rpv_id
                rpv_cand_data = cur_data_minus_ev[np.logical_and(preceding_sv_ind, right_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.RIGHT_PRECEDING_ID]] = \
                rpv_cand_data[np.argmin(rpv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(preceding_sv_ind, right_lane_sv_ind)) == True else 0
                
                #rfv_id
                rfv_cand_data = cur_data_minus_ev[np.logical_and(following_sv_ind, right_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.RIGHT_FOLLOWING_ID]] = \
                rfv_cand_data[np.argmax(rfv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(following_sv_ind, right_lane_sv_ind)) == True else 0

                #lpv_id
                lpv_cand_data = cur_data_minus_ev[np.logical_and(preceding_sv_ind, left_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.LEFT_PRECEDING_ID]] = \
                lpv_cand_data[np.argmin(lpv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(preceding_sv_ind, left_lane_sv_ind)) == True else 0
                
                #lfv_id
                lfv_cand_data = cur_data_minus_ev[np.logical_and(following_sv_ind, left_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.LEFT_FOLLOWING_ID]] = \
                lfv_cand_data[np.argmax(lfv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]]\
                if np.any(np.logical_and(following_sv_ind, left_lane_sv_ind)) == True else 0
                
                #rav_id
                rav_cand_data = cur_data_minus_ev[np.logical_and(alongside_sv_ind, right_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.RIGHT_ALONGSIDE_ID]] = \
                rav_cand_data[np.argmax(rav_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(alongside_sv_ind, right_lane_sv_ind)) == True else 0
                
                #lav_id
                lav_cand_data = cur_data_minus_ev[np.logical_and(alongside_sv_ind, left_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.LEFT_ALONGSIDE_ID]] = \
                lav_cand_data[np.argmax(lav_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(alongside_sv_ind, left_lane_sv_ind)) == True else 0
                #print('SVs: ',cur_aug_features[veh_itr])

            augmented_features[selected_ind] = cur_aug_features

        ngsim_data = np.concatenate((ngsim_data, augmented_features), axis = 1)
        
        
        return ngsim_data, SVC_dict

    def transform_track_features(self, ngsim_data, NC_dict, logging = False):
        """
        This method iterate on all rows of ngsim dataset sorted by vehicle ID and:
            1. Correct vehicle ID duplication problem.
            2. TODO:Re-calc X-/Y- Velocity/Acceleration using X and Y (all using NGSIM coordinate) after applying a filtering algorithm. 
        """
        current_veh_id = ngsim_data[0, NC_dict[NC.ID]] 
        correct_veh_id = ngsim_data[0,NC_dict[NC.ID]] 
        cur_t = ngsim_data[0,NC_dict[NC.GLOBAL_TIME]]
        empty_veh_id = max(list(set(ngsim_data[:,NC_dict[NC.ID]])))+1
        num_rows = ngsim_data.shape[0]
        #augmented_features = np.zeros((num_rows, 4))
        fr = 0
        for row_itr in range(num_rows):
            if current_veh_id != ngsim_data[row_itr, NC_dict[NC.ID]]:
                current_veh_id = ngsim_data[row_itr, NC_dict[NC.ID]]
                correct_veh_id = ngsim_data[row_itr, NC_dict[NC.ID]]
                cur_t = ngsim_data[row_itr, NC_dict[NC.GLOBAL_TIME]] + 100
                fr = 0 + 1
                continue
            if cur_t != ngsim_data[row_itr, NC_dict[NC.GLOBAL_TIME]]:
                correct_veh_id = empty_veh_id
                if logging:
                    print("Duplicatation found in id: ", current_veh_id, ". The id is changed to: ", correct_veh_id)
                empty_veh_id += 1
                ngsim_data[row_itr, NC_dict[NC.ID]] = correct_veh_id
                cur_t = ngsim_data[row_itr, NC_dict[NC.GLOBAL_TIME]] + 100
                fr = 0 + 1
                continue
            
            ngsim_data[row_itr, NC_dict[NC.ID]] = correct_veh_id
            
            cur_t += 100
            fr += 1
        return ngsim_data

    def convert_static_info(self):
        # TODO: Export static info from NGSIM
        return 0
    def convert_meta_info(self):
        # TODO: Export following meta features from NGSIM:
        #  SPEED_LIMIT, MONTH, WEEKDAY, START_TIME, DURATION, TOTAL_DRIVEN_DISTANCE, TOTAL_DRIVEN_TIME, N_CARS, N_TRUCKS
        for i,location in enumeratE(self.files):
            ngsim_transformed = pandas.read_csv(location+'_transformed.csv')
            meta_columns = [HC.ID, HC.FRAME_RATE, HC.LOCATION_ID, HC.N_VEHICLES, HC.UPPER_LANE_MARKINGS, HC.LOWER_LANE_MARKINGS]
            # Note: Upper lanes are not recorded in NGSIM, we arbitrary set some values to them.
            meta_data = [i, 10, i, ngsim_transformed[HC.TRACK_ID].max(), ]
            meta = pandas.DataFrame(meta_data, columns = meta_columns)
        return 0
    def save(self):
        return 0e