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
        


    def convert_tracks_info(self):
        """ This method applies following changes:
            1. Delete Unneccessary Coloumns:          
            2. Modify Existing Coloumns:
            3. Compute New Coloumns: 
        """
        for i, traj_file in enumerate(self.files):  
            self.ngsim.append(pandas.read_csv(self.ngsim_csv_file_dir+ traj_file))
            self.ngsim[i] = self.ngsim[i].drop(
                            columns = [
                                NC.GLOBAL_X, 
                                NC.GLOBAL_Y,
                                NC.GLOBAL_TIME
                                #NC.PRECEDING_ID,
                                #NC.FOLLOWING_ID,
                                ])
            ngsim_columns = self.ngsim[i].columns
            ngsim_array = self.ngsim[i].to_numpy()
            NC_dict = {}
            for i,c in enumerate(ngsim_columns):
                NC_dict[c] = i
            
            ngsim_array, SVC_dict = self.transform_frame_features(ngsim_array, NC_dict)
            
            highD_columns = [None]* (len(ngsim_columns) + len(SVC_dict))
            # Untransformed Columns
            highD_columns[NC_dict[NC.CLASS]] = NC.CLASS
            highD_columns[NC_dict[NC.VELOCITY]] = NC.VELOCITY # Note: Velocity is changed from feet/s to m/s
            highD_columns[NC_dict[NC.ACCELERATION]] = NC.ACCELERATION # Note: Acceleration is changed from feet/s^2 to m/s^2 
            highD_columns[NC_dict[NC.PRECEDING_ID]] = NC.PRECEDING_ID 
            highD_columns[NC_dict[NC.FOLLOWING_ID]] = NC.FOLLOWING_ID
            highD_columns[NC_dict[NC.TOTAL_FRAME]] = NC.TOTAL_FRAME
            
            # Transformed Columns
            highD_columns[NC_dict[NC.ID]] = HC.TRACK_ID
            highD_columns[NC_dict[NC.FRAME]] = HC.FRAME
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
            transformed_ngsim.to_csv(self.ngsim_csv_file_dir+traj_file+'_transformed.csv', index=False)

    def transform_frame_features(self, ngsim_data, NC_dict, logging = True):
        """
        * Transform from feet to meter. 
        * Reverse the order of Lane IDs
        * Extract vehicle IDs of surrounding vehicles.
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

        sorted_ind = np.argsort(ngsim_data[:,NC_dict[NC.FRAME]])
        ngsim_data = ngsim_data[sorted_ind]
        
        # Feet => meter
        ngsim_data[:,NC_dict[NC.X]] = 0.3048 * ngsim_data[:,NC_dict[NC.X]]
        ngsim_data[:,NC_dict[NC.Y]] = 0.3048 * ngsim_data[:,NC_dict[NC.Y]]
        ngsim_data[:,NC_dict[NC.LENGTH]] = 0.3048 * ngsim_data[:,NC_dict[NC.LENGTH]]
        ngsim_data[:,NC_dict[NC.WIDTH]] = 0.3048 * ngsim_data[:,NC_dict[NC.WIDTH]]
        ngsim_data[:,NC_dict[NC.VELOCITY]] = 0.3048 * ngsim_data[:,NC_dict[NC.VELOCITY]]
        ngsim_data[:,NC_dict[NC.ACCELERATION]] = 0.3048 * ngsim_data[:,NC_dict[NC.ACCELERATION]]
        ngsim_data[:,NC_dict[NC.DHW]] = 0.3048 * ngsim_data[:,NC_dict[NC.DHW]]
        # Change order of lane numbers

        augmented_features = np.zeros((ngsim_data.shape[0], 8))
        all_frames = sorted(list(set(ngsim_data[:,NC_dict[NC.FRAME]])))
        max_itr = len(all_frames)
        for itr, frame in enumerate(all_frames):
            if logging and itr%100 == 0:
                print('Processing: ', itr, 'out_of: ', max_itr)
            
            
            selected_ind = ngsim_data[:,NC_dict[NC.FRAME]] == frame
            cur_data = ngsim_data[selected_ind]
            #print("Current Vehicles:{} at: {}".format(cur_data[:,NC_dict[NC.ID]], frame))
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
                left_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == (cur_lane-1))
                right_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == (cur_lane+1))
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

    
    def convert_static_info(self):
        # TODO:  Export following meta features from NGSIM:
        #  TRAVELED_DISTANCE, MIN_X_VELOCITY, MAX_X_VELOCITY, MEAN_X_VELOCITY, MIN_DHW, MIN_THW, MIN_TTC, NUMBER_LANE_CHANGES
        for i,traj_file in enumerate(self.files):
            ngsim_transformed = pandas.read_csv(self.ngsim_csv_file_dir + traj_file + '_transformed.csv')
            static_columns = [HC.INITIAL_FRAME, HC.FINAL_FRAME, HC.NUM_FRAMES, NC.CLASS, HC.DRIVING_DIRECTION]
            ngsim_transformed = ngsim_transformed.sort_values(by=[HC.TRACK_ID])
            max_track_id = int(ngsim_transformed[HC.TRACK_ID].max())
            ngsim_columns = ngsim_transformed.columns
            ngsim_array = ngsim_transformed.to_numpy()
            HC_dict = {}
            for i,c in enumerate(ngsim_columns):
                HC_dict[c] = i
            static_data = np.zeros((max_track_id, len(static_columns)))
            
            for itr in range(max_track_id):
                track_id = itr+1
                cur_track_data = ngsim_array[ngsim_array[:, HC_dict[HC.TRACK_ID]==track_id]] 
                initial_frame = min(cur_track_data[:,HC_dict[HC.FRAME]])
                final_frame = max(cur_track_data[:,HC_dict[HC.FRAME]])
                num_frame = final_frame - initial_frame
                v_class = cur_track_data[0, HC_dict[NC.CLASS]]
                driving_dir = 2
                static_data[itr,:] = [initial_frame, final_frame, num_frame, v_class, driving_dir]

            static = pandas.DataFrame(data = static_data, columns = static_columns)
            static.to_csv(self.ngsim_csv_file_dir + 'static_'+traj_file, index = False)
    def convert_meta_info(self):
        # TODO: Export following meta features from NGSIM:
        #  SPEED_LIMIT, MONTH, WEEKDAY, START_TIME, DURATION, TOTAL_DRIVEN_DISTANCE, TOTAL_DRIVEN_TIME, N_CARS, N_TRUCKS
        for i,traj_file in enumerate(self.files):
            ngsim_transformed = pandas.read_csv(self.ngsim_csv_file_dir + traj_file + '_transformed.csv')
            meta_columns = [HC.ID, HC.FRAME_RATE, HC.LOCATION_ID, HC.N_VEHICLES, HC.UPPER_LANE_MARKINGS, HC.LOWER_LANE_MARKINGS]
            ngsim_transformed = ngsim_transformed.sort_values(by=[HC.LANE_ID])
            max_lane = int(ngsim_transformed[HC.LANE_ID].max())
            ngsim_columns = ngsim_transformed.columns
            ngsim_array = ngsim_transformed.to_numpy()
            HC_dict = {}
            for i,c in enumerate(ngsim_columns):
                HC_dict[c] = i
            
            lower_lanes = np.zeros((max_lane+1))
            average_y = np.zeros((max_lane))
            for lane in range(max_lane):
                lane_id = lane+1
                average_y[lane] = np.mean(ngsim_array[ngsim_array[:,HC_dict[HC.LANE_ID]] == lane_id, HC_dict[HC.Y]])
            
            for lane in range(max_lane+1):
                lane_id = lane+1
                if lane_id ==1 or lane_id == max_lane+1:
                    continue
                lower_lanes[lane] = average_y[lane-1] + (average_y[lane] - average_y[lane-1])/2
            lower_lanes[0] = lower_lanes[1] - 2*(lower_lanes[1] - average_y[0])
            lower_lanes[-1] = lower_lanes[-2] + 2*(average_y[-1] - lower_lanes[-2])
            upper_lane = np.array([lower_lanes[-1]])
            print("Estimated Lower Lane Markings: {}".format(lower_lanes))
            # Note: Upper lanes are not recorded in NGSIM, we arbitrary set some values to them.
            meta_data = [i, 10, i, ngsim_transformed[HC.TRACK_ID].max(), upper_lane.tostring(), lower_lanes.tostring()]
            print(meta_data)
            meta = pandas.DataFrame(data = meta_data, columns = meta_columns)
            meta.to_csv(self.ngsim_csv_file_dir + 'meta_'+traj_file, index = False)
        
    