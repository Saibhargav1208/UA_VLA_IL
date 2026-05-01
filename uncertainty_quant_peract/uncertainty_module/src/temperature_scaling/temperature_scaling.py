import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import lr_scheduler

# from utils.temp_scale import softmax_q_trans, softmax_q_rot_grip, softmax_ignore_collision, choose_highest_action

class TemperatureScaler:
    def __init__(self, calib_type,
                 device, 
                 rotation_resolution, 
                 batch_size, 
                 num_rotation_classes, 
                 voxel_size,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 training: bool = False,
                 use_hard_temp: bool = False,
                 hard_temp: float = 1.0,
                 training_iter: int = 10000,
                 scaler_log_root: str = None
                 ):
        self.calib_type = calib_type
        self.device = device
        self.use_hard_temp = use_hard_temp
        self.training = training
        self.training_iter = training_iter
        self.scaler_log_root = scaler_log_root
        
        print('temp logging dir:', self.scaler_log_root)
        print('am I useing hard_temp?', use_hard_temp)
        print('hard_temp', hard_temp)
        if not use_hard_temp:
            self.temperature = torch.nn.Parameter(torch.ones(1, device=self.device))
        else:
            self.training = False
            self.temperature = hard_temp * torch.nn.Parameter(torch.ones(1, device=self.device))
        
        if self.training:
            self.optimizer = torch.optim.Adam([self.temperature], lr=0.1)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                            step_size=self.training_iter//10, 
                                            gamma=0.1)
        else:
            self.optimizer = None

        self._rotation_resolution = rotation_resolution
        self._batch_size = batch_size
        self.bs = batch_size
        self._num_rotation_classes = num_rotation_classes
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._voxel_size = voxel_size
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        
        # one-hot zero tensors
        self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                        1,
                                                        self._voxel_size,
                                                        self._voxel_size,
                                                        self._voxel_size),
                                                        dtype=int,
                                                        device=device)
        self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                        dtype=int,
                                                        device=device)
        self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                        dtype=int,
                                                        device=device)
        self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                        dtype=int,
                                                        device=device)
        self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                        2),
                                                        dtype=int,
                                                        device=device)
        self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                    2),
                                                                    dtype=int,
                                                                    device=device)

        
    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision


    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def compute_loss(self, logits, labels):
        q_trans, q_rot_grip, q_collision = logits 
        
        action_trans, action_rot_grip, action_ignore_collisions = labels
        
        coords, \
        rot_and_grip_indicies, \
        ignore_collision_indicies = self.choose_highest_action(q_trans, q_rot_grip, q_collision)
        
        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # translation one-hot
        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(self.bs):
            gt_coord = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # translation loss
        q_trans_flat = q_trans.view(self.bs, -1)
        action_trans_one_hot_flat = action_trans_one_hot.view(self.bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(self.bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat =  q_rot_grip[:, 3*self._num_rotation_classes:]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat, action_ignore_collisions_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()
        return total_loss

    def scale_logits(self, logits):
        if self.temperature is None:
            raise ValueError("Temperature not set. First run the calibration process.")
        print('hard_temp', self.temperature)
        q_trans, q_rot_grip, q_collision = logits
        return [q_trans/self.temperature, q_rot_grip/self.temperature, q_collision/self.temperature]

    def set_temperature(self, T: float) -> None:
        """UA-VLA-IL: Override temperature with VLM-predicted adaptive value."""
        self.temperature.data.fill_(T)

    def get_calibrated_probs(self, logits):
        scaled_logits = self.scale_logits(logits)
        [q_trans, q_rot_grip, q_collision] = scaled_logits
        
        trans_conf = self._softmax_q_trans(q_trans).max().detach().cpu().item()
        rot_grip_conf = self._softmax_q_rot_grip(q_rot_grip)
        rot_conf = torch.stack(torch.split(rot_grip_conf[:, :-2],int(360 // self._rotation_resolution),dim=1), dim=1)
        rot_conf = torch.prod(rot_conf.detach().cpu()[0].max(dim=1)[0]).item()

        grip_conf = rot_grip_conf[:, -2:].detach().cpu().max().item()
        collision_conf = self._softmax_ignore_collision(q_collision).max().detach().cpu().item()
        total_conf = trans_conf * rot_conf * grip_conf * collision_conf
        
        return total_conf
    
    def get_val(self):
        return self.temperature
    def save_parameter(self, task_name=None, savedir=None):
        savedir = self.scaler_log_root
        
        if not task_name:
            temp_file = "temperature.pth"
        else:
            savedir = os.path.join(savedir, task_name)
            temp_file = task_name + "_temperature.pth"
        full_path = os.path.join(savedir, temp_file)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        torch.save(self.temperature, full_path)
    
    def load_parameter(self, task_name=None, savedir=None):
        savedir = self.scaler_log_root
        # if using hard temp, don't load
        if not self.use_hard_temp:
            if not task_name:
                temp_file = "temperature.pth"
            else:
                savedir = os.path.join(savedir, task_name)
                temp_file = task_name + "_temperature.pth"
            full_path = os.path.join(savedir, temp_file)
                
            if os.path.exists(full_path):
                loaded_temperature = torch.load(full_path)
                self.temperature.data = loaded_temperature.data
            # TODO: if it does not exist, don't load it
            else:
                print(f"Error: No weights found at {full_path}")
                print("Initializing temperature to 1.0")
                self.temperature.data.fill_(1.0)
        else:
            # print("Initializing temperature to hard coded temperature")
            # self.temperature.data.fill_(self.temperature)
            print("Using hardcoded temperature:", self.temperature)
            pass
