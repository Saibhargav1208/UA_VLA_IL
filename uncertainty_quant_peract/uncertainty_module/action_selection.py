import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
from itertools import product
from tqdm import tqdm

class ActionSelection:
    def __init__(self, device, 
                rotation_resolution, 
                batch_size, 
                num_rotation_classes, 
                voxel_size,
                temperature: float = 1.0,
                alpha1: float = 1.0,
                alpha2: float = 1.0,
                alpha3: float = 1000,
                alpha4: float = 1000,
                tau: float = 3,
                trans_conf_thresh: float = 1e-5,
                rot_conf_thresh: float = 1/72,
                search_size: int = 20,
                search_step: int = 2,
                log_dir: str = None,
                enabled: bool = True
                ):
        self.device = device
        self._temperature = temperature

        self._rotation_resolution = rotation_resolution
        self._batch_size = batch_size
        self.bs = batch_size
        self._num_rotation_classes = num_rotation_classes
        self._voxel_size = voxel_size
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._alpha3 = alpha3
        self._alpha4 = alpha4
        self._tau = tau
        self._trans_conf_thresh = trans_conf_thresh
        self._rot_conf_thresh = rot_conf_thresh
        self._search_size = search_size
        self._search_step = search_step
        self.log_dir = log_dir
        self.enabled = enabled
        
        self._trans_search_size = 0 #2*2 # 5*2
        self._rot_search_size = 0 # 2*2 # 6*2
        self._tran_size = 100-1
        self._rot_size = 72-1
        
    def get_action_diff(self, actions, target_actions):
        for i, t in enumerate(actions):
            if t.dtype != torch.float32:
                actions[i] = t.float()
        for i, t in enumerate(target_actions):
            if t.dtype != torch.float32:
                target_actions[i] = t.float()
        # pdb.set_trace()
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions_action = actions

        action_trans, \
        target_rot_and_grip_indicies, \
        target_ignore_collision_indicies = target_actions
        
        # TODO: add support when target_actions are in (n, 8)
        # print(action_trans)
        if action_trans.ndim == 1:
            diff_trans = self._alpha1 * torch.norm(coords - action_trans)
            diff_rot = self._alpha2 * torch.norm(rot_and_grip_indicies[:-1] - target_rot_and_grip_indicies[:-1])
            diff_grip = self._alpha3 * torch.norm(rot_and_grip_indicies[-1] - target_rot_and_grip_indicies[-1])
            diff_collision = self._alpha4 * torch.norm(ignore_collisions_action - target_ignore_collision_indicies)
        else:
            diff_trans = self._alpha1 * torch.norm(coords - action_trans, dim=1)
            diff_rot = self._alpha2 * torch.norm(rot_and_grip_indicies[:,:-1] - target_rot_and_grip_indicies[:,:-1], dim=1)
            diff_grip = self._alpha3 * torch.norm(rot_and_grip_indicies[:,-1].reshape(-1,1) - target_rot_and_grip_indicies[:,-1].reshape(-1,1), dim=1)
            diff_collision = self._alpha4 * torch.norm(ignore_collisions_action - target_ignore_collision_indicies, dim=1)
        # print(diff_trans)
        # print(diff_rot)
        # print(diff_grip)
        # print(rot_and_grip_indicies)
        # print(target_rot_and_grip_indicies)
        # print(diff_collision)
        # pdb.set_trace()
        total_diff = diff_trans + diff_rot + diff_grip + diff_collision
        return total_diff
    
    def get_trans_rot_diff(self, actions, target_actions):
        for i, t in enumerate(actions):
            if t.dtype != torch.float32:
                actions[i] = t.float()
        for i, t in enumerate(target_actions):
            if t.dtype != torch.float32:
                target_actions[i] = t.float()
        # pdb.set_trace()
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions_action = actions

        action_trans, \
        target_rot_and_grip_indicies, \
        target_ignore_collision_indicies = target_actions
        
        # TODO: add support when target_actions are in (n, 8)
        # print(action_trans)
        if action_trans.ndim == 1:
            diff_trans = self._alpha1 * torch.norm(coords - action_trans)
            diff_rot = self._alpha2 * torch.norm(rot_and_grip_indicies[:-1] - target_rot_and_grip_indicies[:-1])
            # diff_grip = self._alpha3 * torch.norm(rot_and_grip_indicies[-1] - target_rot_and_grip_indicies[-1])
            # diff_collision = self._alpha4 * torch.norm(ignore_collisions_action - target_ignore_collision_indicies)
        else:
            diff_trans = self._alpha1 * torch.norm(coords - action_trans, dim=1)
            diff_rot = self._alpha2 * torch.norm(rot_and_grip_indicies[:,:-1] - target_rot_and_grip_indicies[:,:-1], dim=1)
            # diff_grip = self._alpha3 * torch.norm(rot_and_grip_indicies[:,-1].reshape(-1,1) - target_rot_and_grip_indicies[:,-1].reshape(-1,1), dim=1)
            # diff_collision = self._alpha4 * torch.norm(ignore_collisions_action - target_ignore_collision_indicies, dim=1)
        # print(diff_trans)
        # print(diff_rot)
        # print(diff_grip)
        # print(rot_and_grip_indicies)
        # print(target_rot_and_grip_indicies)
        # print(diff_collision)
        # pdb.set_trace()
        total_diff = diff_trans + diff_rot
        return total_diff
    

    def equal_action(self, actions, target_actions):
        return self.get_action_diff(actions, target_actions) < self._tau
    
    def encode_rot(self, q_rot_grip):
        # transfer from [1, 216] to [idx, idx, idx, 1/0]
        # to
        # [1, 3, 72]
        q_rot = torch.stack(torch.split(
            q_rot_grip[:, :-2],
            int(360 // self._rotation_resolution),
            dim=1), dim=1)
        return q_rot
        
    def decode_rot(self, q_rot):
        # transfer [idx, idx, idx, 1/0] to [1, 3, 72]
        return q_rot.reshape(q_rot.size(0), 3, -1)

    # self.generate_grid(coords[0], self._trans_search_size)
    # self.generate_grid(rot_and_grip_indicies[0][:-1], self._rot_search_size)
    def generate_grid(self, indices, size):
        # Calculate the half size of the grid
        half_size = size // 2

        # Calculate min and max boundaries
        min_boundary = indices - half_size
        max_boundary = indices + half_size

        # Adjust boundaries to fit within [0, 99]
        for i in range(3):  # For x, y, z dimensions
            if min_boundary[i] < 0:
                offset = 0 - min_boundary[i]
                min_boundary[i] += offset
                max_boundary[i] += offset
            elif max_boundary[i] > self._tran_size:
                offset = max_boundary[i] -self._tran_size
                min_boundary[i] -= offset
                max_boundary[i] -= offset
                
        x_values = torch.linspace(min_boundary[0], max_boundary[0], size+1)
        y_values = torch.linspace(min_boundary[1], max_boundary[1], size+1)
        z_values = torch.linspace(min_boundary[2], max_boundary[2], size+1)

        return [x_values.int(), y_values.int(), z_values.int()]

    def compute_target_score_vectorized(self, action_permutations, thresholded_indices, thresholded_scores):

        n, m = action_permutations.shape[0], thresholded_indices.shape[0]
        accm_scores = torch.zeros(n)
        for i in tqdm(range(n)):
           
            safe_action = action_permutations[i,:].reshape(1,-1).float()
            # pdb.set_trace()
            safe_action = [safe_action[:,:3],
                           safe_action[:,3:7],
                           safe_action[:,7:]]
            goal_action = thresholded_indices.float()
            goal_action = [goal_action[:,:3],
                           goal_action[:,3:7],
                           goal_action[:,7:]]
            # pdb.set_trace()
            action_dists = self.get_action_diff(safe_action, goal_action)
            mask = (action_dists < self._tau)
            # print('max dist',action_dists.max())
            # print('threshed max dist', thresholded_scores[mask].max())
            accm_scores[i] = torch.sum(thresholded_scores[mask])
        return accm_scores

    def get_safest_action_search_around_max(self, confidences, best_actions):
        # TODO: safe_action sometimes have non-max grip/collision actions
        # first do the thresholding for the actions to get the subset of actions
        # 1. if the subset is empty, just take the action with highest confidence
        # 2. otherwise, 
            # 2.1 take the (min, max) of all three axis of translation and rotation as the search space
                # or
            # 2.1 take the fixed space around the highest conf action as the search space ---> implementing this
            # 2.2 take the max confidence for the grip openess and collision avoidance
        trans_conf_softmax, rot_conf_softmax, grip_conf_softmax, collision_conf_softmax = confidences
        
        # get confidence map for each action
        trans_conf_softmax = trans_conf_softmax[0,0,:,:,:]
        rot_conf_softmax = rot_conf_softmax[0,:,:]
        grip_conf_softmax = grip_conf_softmax[0,:]
        collision_conf_softmax = collision_conf_softmax[0,:]
        with torch.no_grad():
            # Generate indices to define a cubic space around the best_actions
            coords, rot_and_grip_indicies, ignore_collision_indicies = best_actions
            trans_indices = self.generate_grid(coords[0], self._trans_search_size) # self.generate_trans_grid(coords)
            rot_indices = self.generate_grid(rot_and_grip_indicies[0][:-1], self._rot_search_size) # self.generate_rot_grid(rot_and_grip_indicies)
            grip_indices = torch.tensor([0, 1], device=trans_indices[0].device).int()
            collide_indices = torch.tensor([0, 1], device=trans_indices[0].device).int()
            
            # Generate permutation for the action spaces
            grid = torch.meshgrid(trans_indices[0],
                                  trans_indices[1].int(), 
                                  trans_indices[2].int(),
                                  rot_indices[0].int(),
                                  rot_indices[1].int(),
                                  rot_indices[2].int(), 
                                  grip_indices, 
                                  collide_indices)
            
            # Reshape the grid to have all permutations
            permutations = [g.reshape(-1) for g in grid]
            
            # indices(action) is the neighboring space around the max_q
            # is of size ((self._trans_search_size+1)*(self._rot_search_size+1)*4, 8)
            indices = torch.stack(permutations, dim=1).long() 
            
            # calculate the scores within the neighboring region
            trans_scores = trans_conf_softmax[indices[:, 0], indices[:, 1], indices[:, 2]]
            rot_scores = rot_conf_softmax[0,indices[:,3+0]] \
                        *rot_conf_softmax[1,indices[:,3+1]] \
                        *rot_conf_softmax[2,indices[:,3+2]]
            grid_scores = grip_conf_softmax[indices[:,6]]
            collision_scores = collision_conf_softmax[indices[:,7]]
            
            score_map = trans_scores * rot_scores * grid_scores * collision_scores
            
            # thresholding the score_map by beta(confidence threshold)
            mask = score_map > self._conf_thresh
            thresholded_score_map = score_map[mask]
            thresholded_indices = indices[mask]

            # generate the scores for all indices
            accum_scores = self.compute_target_score_vectorized(indices, # m x 7
                                                                thresholded_indices, # n x 7
                                                                thresholded_score_map) # n
            # take the action (indices) with the max accumulative scores
            safest_action = indices[accum_scores.max(0).indices.item()].to(coords[0].device)

            safest_action = [
                safest_action[:3],
                safest_action[3:-1],
                safest_action[-1]
            ]

            return safest_action
    
    def get_safest_action_from_each_space_select_from_high_score(self, confidences):
        
        def get_safe_action(conf_softmax, top_k=4000, tau=5, conf_thresh = 1/(100**3)):
            # trans_top_k = 4000 #2000*5 # trans_thresh = 1/(100**3) # beta
            # trans_tau = 3
            with torch.no_grad():
                space_size = conf_softmax.shape[0]
                
                _range = torch.arange(space_size)
                xx, yy, zz = torch.meshgrid(_range,_range,_range)
                actions = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
                
                # reshape score map to (100^3, 1)
                score_map = conf_softmax.view(-1,1)
                # get score and indices for score_map > trans_thresh
                score_mask = score_map > conf_thresh
                filtered_score_map = score_map[score_mask]
                filtered_actions = actions[score_mask.flatten()]
                # pdb.set_trace()
                if filtered_score_map.shape[0] > top_k:
                    # generate the masks for the topk
                    _, top_indices = torch.topk(filtered_score_map.squeeze(), top_k)
                    score_mask = torch.zeros_like(filtered_score_map, dtype=torch.bool)
                    score_mask[top_indices] = True
                    # filter the score_map and actions again
                    filtered_score_map = filtered_score_map[score_mask]
                    filtered_actions = filtered_actions[score_mask.flatten()]
                
                n = filtered_score_map.shape[0]
                print(n)
                
                dist_matrix = torch.norm(filtered_actions[:,None].float()-filtered_actions.float(),dim=2)
                dist_mask = (dist_matrix < tau).to(filtered_score_map.device)
                
                square_filtered_score_map = filtered_score_map.repeat(n, 1)
                accum_scores = (square_filtered_score_map*dist_mask).sum(dim=1,keepdim=True)
                
                num_max_scores = (accum_scores == accum_scores.max()).sum()
                if num_max_scores > 1: # if multiple max scores, just take the q_max
                    actions_index = filtered_score_map.max(0).indices.item()
                else: 
                    actions_index = accum_scores.max(0).indices.item()
                # pdb.set_trace()
                safe_action = filtered_actions[actions_index,:]
                # accum_scores = torch.sum(square_filtered_score_map[dist_mask], dim=1)
                #TODO: if all ones, just take the q_max
                return safe_action         
                
    
        trans_conf_softmax, rot_conf_softmax, grip_conf_softmax, collision_conf_softmax = confidences
        
        # get confidence map for each action
        trans_conf_softmax = trans_conf_softmax[0,0,:,:,:] # (100,100,100)
        rot_conf_softmax = rot_conf_softmax[0,:,:] # (3, 72)
        grip_conf_softmax = grip_conf_softmax[0,:] # (2)
        collision_conf_softmax = collision_conf_softmax[0,:] # (2)
        
        with torch.no_grad():
            _device = trans_conf_softmax.device
            safe_trans = get_safe_action(trans_conf_softmax, top_k=4000, tau=3, trans_thresh = 1/(100**3)).to(_device)
            # pdb.set_trace()
            rot_conf_softmax0 = rot_conf_softmax[0,:].view(72, 1, 1)
            rot_conf_softmax1 = rot_conf_softmax[1,:].view(1, 72, 1)
            rot_conf_softmax2 = rot_conf_softmax[2,:].view(1, 1, 72)

            # Use broadcasting to generate the score space
            rot_conf_softmax_grid = rot_conf_softmax0 * rot_conf_softmax1 * rot_conf_softmax2
            safe_rot = get_safe_action(rot_conf_softmax_grid, top_k=4000, tau=2, trans_thresh = 1/(72**3))
            
            safe_rot_and_grip = torch.cat([safe_rot.to(_device), grip_conf_softmax.max(0).indices.unsqueeze(0).to(_device)])
            safe_collision = collision_conf_softmax.max(0).indices.unsqueeze(0).to(_device)
            return [safe_trans, safe_rot_and_grip, safe_collision]
        
            
            

    def get_safest_action_from_each_space_select_around_mean(self, confidences, best_trans):
        def generate_grid(indices, size, step_size):
            # Calculate the half size of the grid
            half_size = size // 2

            # Calculate min and max boundaries
            min_boundary = indices - half_size
            max_boundary = indices + half_size

            # Adjust boundaries to fit within [0, 99]
            for i in range(3):  # For x, y, z dimensions
                if min_boundary[i] < 0:
                    offset = 0 - min_boundary[i]
                    min_boundary[i] += offset
                    max_boundary[i] += offset
                elif max_boundary[i] > self._tran_size:
                    offset = max_boundary[i] -self._tran_size
                    min_boundary[i] -= offset
                    max_boundary[i] -= offset
            # Calculate number of steps for each dimension
            x_steps = int((max_boundary[0] - min_boundary[0]) / step_size)
            y_steps = int((max_boundary[1] - min_boundary[1]) / step_size)
            z_steps = int((max_boundary[2] - min_boundary[2]) / step_size)
            
            x_values = torch.linspace(min_boundary[0], max_boundary[0], x_steps+1)
            y_values = torch.linspace(min_boundary[1], max_boundary[1], y_steps+1)
            z_values = torch.linspace(min_boundary[2], max_boundary[2], z_steps+1)

            return [x_values.int(), y_values.int(), z_values.int()]

        def get_action_permutation(actions):
            # Generate permutation for the action spaces
            grid = torch.meshgrid(
                                actions[0].int(),
                                actions[1].int(),
                                actions[2].int())
        
            # Reshape the grid to have all permutations
            permutations = [g.reshape(-1) for g in grid]
            
            # indices(action) is the neighboring space around the max_q
            # is of size ((self._trans_search_size+1)*(self._rot_search_size+1)*4, 8)
            action_perm = torch.stack(permutations, dim=1).long() 
            return action_perm
        
        def get_safe_action(conf_softmax, 
                            top_k=4000, 
                            tau=3, 
                            conf_thresh = 1/(100**3), 
                            search_size=20, 
                            search_step=2,
                            best_action=None):
            # trans_top_k = 4000 #2000*5 # trans_thresh = 1/(100**3) # beta
            # trans_tau = 3
            with torch.no_grad():
                space_size = conf_softmax.shape[0]
                
                _range = torch.arange(space_size)
                xx, yy, zz = torch.meshgrid(_range,_range,_range)
                actions = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=1)
                
                # reshape score map to (100^3, 1)
                score_map = conf_softmax.view(-1,1)
                # get score and indices for score_map > trans_thresh
                score_mask = score_map > conf_thresh
                filtered_score_map = score_map[score_mask]
                filtered_actions = actions[score_mask.flatten()]
                # pdb.set_trace()
                if filtered_score_map.shape[0] > top_k:
                    # generate the masks for the topk
                    _, top_indices = torch.topk(filtered_score_map.squeeze(), top_k)
                    score_mask = torch.zeros_like(filtered_score_map, dtype=torch.bool)
                    score_mask[top_indices] = True
                    # filter the score_map and actions again
                    filtered_score_map = filtered_score_map[score_mask]
                    filtered_actions = filtered_actions[score_mask.flatten()]
                
                # get the mean value for each axis
                center_action = torch.mean(filtered_actions.float(),axis=0).int()
                # print('center_action:', center_action)
                # center_action = best_action[0].cpu()
                # pdb.set_trace()
                search_space = generate_grid(center_action, search_size, search_step)
                # search_space = torch.stack(search_space).T
                
                
                search_actions = get_action_permutation(search_space)
                
                diffs = (search_actions.unsqueeze(1) - filtered_actions.unsqueeze(0)).float()
                dist_matrix = torch.norm(diffs, dim=-1)
                dist_mask = (dist_matrix < tau).to(filtered_score_map.device)
                n = dist_matrix.shape[0]
                
                
                square_filtered_score_map = filtered_score_map.repeat(n, 1)
                accum_scores = (square_filtered_score_map*dist_mask).sum(dim=1,keepdim=True)
                # pdb.set_trace()
                
                num_max_scores = (accum_scores == accum_scores.max()).sum()
                if num_max_scores > 1: # if multiple max scores, just take the q_max
                    actions_index = filtered_score_map.max(0).indices.item()
                    safe_action = filtered_actions[actions_index,:]
                else: 
                    actions_index = accum_scores.max(0).indices.item()
                    safe_action = search_actions[actions_index,:]
                # pdb.set_trace()
                
                # accum_scores = torch.sum(square_filtered_score_map[dist_mask], dim=1)
                #TODO: if all ones, just take the q_max
                return safe_action, center_action       
                
        def get_safe_rot(conf_softmax, 
                            top_k=4000, 
                            tau=3, 
                            conf_thresh = 1/72, 
                            search_size=20, 
                            search_step=2,
                            best_action=None):
            with torch.no_grad():
                space_size = conf_softmax.shape[0]
                
                # Generate 1D actions
                actions = torch.arange(space_size).view(-1, 1)
                
                # Reshape score map to (N, 1), where N is the space size
                score_map = conf_softmax.view(-1, 1)
                
                # Get score and indices for score_map > conf_thresh
                score_mask = score_map > conf_thresh
                filtered_score_map = score_map[score_mask]
                filtered_actions = actions[score_mask.flatten()]
                
                if filtered_score_map.shape[0] > top_k:
                    _, top_indices = torch.topk(filtered_score_map.squeeze(), top_k)
                    score_mask = torch.zeros_like(filtered_score_map, dtype=torch.bool)
                    score_mask[top_indices] = True
                    filtered_score_map = filtered_score_map[score_mask]
                    filtered_actions = filtered_actions[score_mask.flatten()]
                
                # Get the mean value for the action
                center_action = torch.mean(filtered_actions.float()).int()
                
                # Generate search space around the center action
                search_space = torch.arange(center_action - search_size, center_action + search_size, search_step).view(-1, 1)
                
                # Calculate distances and scores
                diffs = (search_space - filtered_actions).float()
                dist_matrix = torch.norm(diffs, dim=-1)
                dist_mask = (dist_matrix < tau).to(filtered_score_map.device)
                
                square_filtered_score_map = filtered_score_map.repeat(dist_matrix.shape[0], 1)
                accum_scores = (square_filtered_score_map * dist_mask).sum(dim=1, keepdim=True)
                
                num_max_scores = (accum_scores == accum_scores.max()).sum()
                if num_max_scores > 1:
                    actions_index = filtered_score_map.max(0).indices.item()
                    safe_action = filtered_actions[actions_index]
                else: 
                    actions_index = accum_scores.max(0).indices.item()
                    safe_action = search_space[actions_index]
                
                return safe_action, center_action

        trans_conf_softmax, rot_conf_softmax, grip_conf_softmax, collision_conf_softmax = confidences
        
        # get confidence map for each action
        trans_conf_softmax = trans_conf_softmax[0,0,:,:,:] # (100,100,100)
        rot_conf_softmax = rot_conf_softmax[0,:,:] # (3, 72)
        grip_conf_softmax = grip_conf_softmax[0,:] # (2)
        collision_conf_softmax = collision_conf_softmax[0,:] # (2)
        
        with torch.no_grad():
            _device = trans_conf_softmax.device
            safe_trans, center_trans = get_safe_action(trans_conf_softmax, 
                                                       top_k=4000, 
                                                       tau=self._tau, 
                                                       conf_thresh=self._trans_conf_thresh, 
                                                       search_size=self._search_size, 
                                                       search_step=self._search_step)#, 
                                                       #best_action=best_trans)
            safe_trans = safe_trans.to(_device)
            center_trans = center_trans.to(_device)
            # pdb.set_trace()
            rot_conf_softmax0 = rot_conf_softmax[0,:].view(72, 1, 1)
            rot_conf_softmax1 = rot_conf_softmax[1,:].view(1, 72, 1)
            rot_conf_softmax2 = rot_conf_softmax[2,:].view(1, 1, 72)

            # Use broadcasting to generate the score space
            # rot_conf_softmax_grid = rot_conf_softmax0 * rot_conf_softmax1 * rot_conf_softmax2
            # safe_rot = get_safe_action(rot_conf_softmax_grid, top_k=4000, tau=3, conf_thresh = 1/(72**3))
            
            # use best_rot as safe_rot
            # pdb.set_trace()
            safe_rot = torch.tensor(
                [rot_conf_softmax[0,:].argmax(-1).item(),
                 rot_conf_softmax[1,:].argmax(-1).item(),
                 rot_conf_softmax[2,:].argmax(-1).item()])
            
            # safe_rot_0 = get_safe_rot(rot_conf_softmax[0,:], 
            #                              top_k=4000, 
            #                              tau=self._tau, 
            #                              conf_thresh=self._rot_conf_thresh,
            #                              search_size=self._search_size, 
            #                              search_step=self._search_step)
            # safe_rot_1 = get_safe_rot(rot_conf_softmax[1,:], 
            #                              top_k=4000, 
            #                              tau=self._tau, 
            #                              conf_thresh=self._rot_conf_thresh,
            #                              search_size=self._search_size, 
            #                              search_step=self._search_step)            
            # safe_rot_2 = get_safe_rot(rot_conf_softmax[2,:], 
            #                              top_k=4000, 
            #                              tau=self._tau, 
            #                              conf_thresh=self._rot_conf_thresh,
            #                              search_size=self._search_size, 
            #                              search_step=self._search_step)  
            # safe_rot = torch.tensor([
            #     safe_rot_0,
            #     safe_rot_1,
            #     safe_rot_2
            # ])
            safe_rot_and_grip = torch.cat([safe_rot.to(_device), grip_conf_softmax.max(0).indices.unsqueeze(0).to(_device)])
            safe_collision = collision_conf_softmax.max(0).indices.unsqueeze(0).to(_device)
            return [safe_trans, safe_rot_and_grip, safe_collision], center_trans












    def tensor_permutations(self, min_indices, max_indices):
        pdb.set_trace()
        # Generate tensor grids for each index range
        grids = [torch.arange(min_idx.item(), max_idx.item() + 1) for min_idx, max_idx in zip(min_indices, max_indices)]
        
        # Create a meshgrid of indices, unpack the grids with star
        mesh = torch.meshgrid(*grids)
        
        matrix_permutations = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        
        del grids
        del mesh
        # Reshape and stack the meshgrid to get the permutations
        return matrix_permutations
    

    
    def get_safest_action_search_all(self, confidences):
        # first do the thresholding for the actions to get the subset of actions
        # 1. if the subset is empty, just take the action with highest confidence
        # 2. otherwise, 
            # 2.1 take the (min, max) of all three axis of translation and rotation as the search space
                # or
            # 2.1 take the fixed space around the highest conf action as the search space
            # 2.2 take the max confidence for the grip openess and collision avoidance
        trans_conf_softmax, rot_conf_softmax, grip_conf_softmax, collision_conf_softmax = confidences
        with torch.no_grad():
            trans_conf_softmax = trans_conf_softmax.clone()[0,0,:,:,:]
            rot_conf_softmax = rot_conf_softmax.clone()[0,:,:]
            grip_conf_softmax = grip_conf_softmax.clone()[0,:]
            collision_conf_softmax = collision_conf_softmax.clone()[0,:]
            
            # trans_indices = torch.stack(torch.where(trans_conf_softmax>1e-5)).shape
            
            # Expand dimensions                                                                                                                                                    
            trans_conf_softmax = trans_conf_softmax.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            rot_conf_softmax = rot_conf_softmax.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            grip_conf_softmax = grip_conf_softmax.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            collision_conf_softmax = collision_conf_softmax.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # pdb.set_trace()
            score_space = trans_conf_softmax * rot_conf_softmax * grip_conf_softmax * collision_conf_softmax
            # score_space = trans_conf_softmax * rot_conf_softmax
            # del score_space
            # TODO: set all scores in tensor 1 to be zero
            # accum_score = torch.zeros_like(score_space)
            
            thresholded_indices = torch.stack(torch.where(score_space > self._conf_thresh)).T
            thresholded_scores = score_space[score_space > self._conf_thresh]
            
            # free up score_space immediately to save memory
            del score_space
            torch.cuda.empty_cache()
            

        

            ## list based permutation generation
            # # Generate lists of indices for each column based on min and max indices
            # index_ranges = [list(range(min_idx, max_idx + 1)) for min_idx, max_idx in zip(min_indices, max_indices)]
            # # Calculate all possible permutations of indices
            # permutations = list(product(*index_ranges))
            # # Convert the permutations to a tensor
            # matrix_permutations = torch.tensor(permutations, dtype=torch.long)
            
            ## generate all permutations within [min, max]
            # the actual maximum size is (100^3 * 72^3 * 2 * 2)
            max_indices = torch.max(thresholded_indices, dim=0).values.cpu().numpy()
            min_indices = torch.min(thresholded_indices, dim=0).values.cpu().numpy()
            action_permutations = self.tensor_permutations(min_indices, max_indices)
            pdb.set_trace()
            
            accum_scores = self.compute_target_score_vectorized(action_permutations, # m x 7
                                                                thresholded_indices, # n x 7
                                                                thresholded_scores) # n

            del trans_conf_softmax
            del rot_conf_softmax
            del grip_conf_softmax
            del collision_conf_softmax
            del thresholded_indices
            del thresholded_scores
            del action_permutations
            torch.cuda.empty_cache()




        # ## generate all action space
        # all_indices_accum_score = torch.stack(torch.meshgrid([torch.arange(s) for s in accum_score.shape]), dim=-1).reshape(-1, 7)


    
        

