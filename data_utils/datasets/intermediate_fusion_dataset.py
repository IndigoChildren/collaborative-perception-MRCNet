# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json


def getIntermediateFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)

            # motion context fusion schemes:
            # i) next frame  supervision for prediction 
            # ii)use current frame supervision of the ego agent's enhance motion context fusion
            self.prediction = True if ('prediction' in params['model']['args'] and params['model']['args']['prediction']) \
                                        else False    
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            
            # project the raw data
            # self.proj_first = False if 'proj_first' not in params['fusion']['args'] else params['fusion']['args']['proj_first']
            self.proj_first = True
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.kd_flag = params.get('kd_flag', False)

            #加入时序信息 
            self.temporal_model = True if ('temporal_setting' in params and params['temporal_setting']) \
                                        else False
            #记录总的frames取多少
            self.frames_nums_total = int(params['temporal_setting']['frames_adj']) + 1 if self.temporal_model else None
            self.store_pre_frames = params['temporal_setting']['store_pre_frames'] if self.temporal_model else False


        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_cav_base : dict
                The dictionary contains the ego agent's raw information.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix)
                if self.proj_first:   #set to True
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar

                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})

            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )
            selected_cav_processed.update({
                                "single_label_dict": label_dict,
                                "single_object_bbx_center": object_bbx_center,
                                "single_object_bbx_mask": object_bbx_mask})

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                        ego_pose_clean)

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )


            return selected_cav_processed
    
        def get_item_his_info(self, selected_cav_base, ego_cav_base):
            """
            Process the ego agent' temporal information for the train/test pipeline.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_cav_base : dict
                The dictionary contains the ego agent's raw information.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            #和get_item_single_car一样，但是只做特征处理，不保存label
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']
        
            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            
            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix)
                if self.proj_first:   # set to True in our scheme
                    lidar_np[:, :3] = projected_lidar
                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})
            return selected_cav_processed
        
        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            
            # add pose noise
            if self.supervise_single:
                supervise_dict = base_data_dict['supervise']
                base_data_dict.pop('supervise')
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}
            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    #lidar_pose是加噪的，liar_clean_pose是没有加噪的
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []

            too_far = []
            lidar_pose_list = []
            cav_id_list = []
            projected_lidar_clean_list = [] # disconet
            #记录历史信息
            his_processed_features = []
            #记录监督信息
            next_frame_processed_features = []

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                #这里有问题，是不是应该换成'lidar_pose_clean'
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                # 防止去掉next_frame信息
                com_agent_flag = 'frame' not in cav_id and 'his' not in cav_id
                if distance > self.params['comm_range'] and com_agent_flag:
                    too_far.append(cav_id)
                    continue

                #append selected cars
                # lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            # filter out the agents beyond the communication range
            for cav_id in too_far:
                base_data_dict.pop(cav_id)


            # if self.project_first == True, then pairwise_t_matrix == eye(4)
            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            
            # merge preprocessed features from different cavs into the same dict
            # cav_num = len(cav_id_list)

            # store historical BEV features in his, not labels
            for _i, cav_id in enumerate(cav_id_list):

                # process the historical BEV features
                if 'his' in cav_id:
                    selected_cav_base = base_data_dict[cav_id]
                    his_info_processed = self.get_item_his_info(
                        selected_cav_base, ego_cav_base)
                    his_processed_features.append(his_info_processed['processed_features'])
                    continue
                # provess the info of next frame
                if 'frame' in cav_id:
                    selected_cav_base = base_data_dict[cav_id]
                    frame_info_processed = self.get_item_his_info(
                    selected_cav_base, ego_cav_base)
                    eval(f'{cav_id}_processed_features').append(frame_info_processed['processed_features'])
                    continue                  
                selected_cav_base = base_data_dict[cav_id]
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)
                    
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                if self.load_lidar_file:
                    processed_features.append(
                        selected_cav_processed['processed_features'])

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                    
            # supervise the current frame
            if self.supervise_single:
                supervise_object_bbx_center, supervise_mask, supervise_object_ids = \
                    self.generate_object_center([supervise_dict],
                                                        ego_lidar_pose)
            if self.kd_flag:
                stack_lidar_np = np.vstack(projected_lidar_stack)
                stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                            self.params['preprocess'][
                                                'cav_lidar_range'])
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({'teacher_processed_lidar':
                stack_feature_processed})

            
            # exclude all repetitive objects    
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            cav_num = len(processed_features)

            if self.supervise_single:
                supervise_label_dict = \
                    self.post_processor.generate_label(
                        gt_box_center = supervise_object_bbx_center,
                        anchors = self.anchor_box,
                        mask = supervise_mask) 
                
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})

            if self.temporal_model:
                # merge the historical features
                his_frames = len(his_processed_features)
                his_merged_feature_dict = merge_features_to_dict(his_processed_features)
            if self.prediction:
                # merge the next frame's feature
                next_frame_merged_feature_dict = merge_features_to_dict(next_frame_processed_features)
            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)
            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,   #self.proj_first=True, then be eyes(4)
                'lidar_poses': lidar_poses})
            
            # add supervision
            if self.supervise_single:
                processed_data_dict['ego'].update(
                        {'supervise_label_dict': supervise_label_dict})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})
            # add temporal information
            if self.temporal_model:
                processed_data_dict['ego'].update({'his_processed_lidar': his_merged_feature_dict,
                                                   'his_frames': his_frames})
       
            if self.prediction:
                has_future_frame = base_data_dict['next_frame']['has_future_frame']
                processed_data_dict['ego'].update({'next_frame_processed_lidar': next_frame_merged_feature_dict,
                                                   'has_future_frame': has_future_frame
                                                    })
            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})
            
                

            return processed_data_dict


        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            # lidar_pose_clean_list = []
            
            # store the historical features
            his_processed_lidar_list = []
            his_frames_lens = []
            # supervised features
            next_frame_processed_lidar_list = []
            future_frame_mask = []   # mask in the loss without next frame
            # the ego agent's supervision
            supervise_label_dict_list = []  
            # all CAVs' processed lidar
            all_processed_lidar_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []
        
            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                # add supervision
                if self.prediction:
                    next_frame_processed_lidar_list.append(ego_dict['next_frame_processed_lidar'])
                    future_frame_mask.append(ego_dict['has_future_frame'])
                if self.supervise_single:
                    supervise_label_dict_list.append(ego_dict['supervise_label_dict'])
                # temporal fusion
                if self.temporal_model:  
                    his_processed_lidar_list.append(ego_dict['his_processed_lidar'])
                    # V2V lidar and historical lidar
                    # features' rank: [v2v features, historical features]
                    all_processed_lidar_list = processed_lidar_list[:]+his_processed_lidar_list[:] + \
                                                next_frame_processed_lidar_list[:]
                    
                    # merge historical information into the batch to facilitate parallel computing
                    all_merged_feature_dict = merge_features_to_dict(all_processed_lidar_list)
                    all_processed_lidar_torch_dict = \
                        self.pre_processor.collate_batch(all_merged_feature_dict)
                    if 'voxel_features' in ego_dict['his_processed_lidar'].keys():
                        his_frames_lens.append(ego_dict['his_frames']) 
                    else:
                        his_frames_lens.append(0)
                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})


            if self.supervise_single:
                supervise_label_torch_dict = \
                    self.post_processor.collate_batch(supervise_label_dict_list)
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                     'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len
            

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose': lidar_pose,
                                    'anchor_box': self.anchor_box_torch})

            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                teacher_processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(teacher_processed_lidar_list)
                output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})
            # with supervision
            if self.prediction:
                output_dict['ego'].update({'future_frame_mask': future_frame_mask})
            if self.supervise_single:
                output_dict['ego'].update({'supervise_label_dict': supervise_label_torch_dict})

            # temporal information
            if self.temporal_model:
                output_dict['ego']['processed_lidar'] = all_processed_lidar_torch_dict
                output_dict['ego'].update({'his_frames_lens': his_frames_lens}) # the length of his_frame within the batch
            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()
            
            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                        'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list']
            })

            return output_dict


        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor


    return IntermediateFusionDataset


