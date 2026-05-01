import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import lr_scheduler

# from utils.temp_scale import softmax_q_trans, softmax_q_rot_grip, softmax_ignore_collision, choose_highest_action

class TemperatureScaler:
    def __init__(self, calib_type,
                 device, 
                 batch_size,
                 add_rgc_loss,
                 voxel_size: int = 220,
                 num_rotation_classes: int = 72, 
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
        self.device = 'cuda:0' #if device==0 else 'cpu'
        self.use_hard_temp = use_hard_temp
        self.training = training
        self.training_iter = training_iter
        self.scaler_log_root = scaler_log_root
        self.add_rgc_loss = add_rgc_loss
        
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        
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
            # self.scheduler = lr_scheduler.StepLR(self.optimizer, 
            #                                 step_size=self.training_iter//10, 
            #                                 gamma=0.1)
            def lr_lambda(epoch):
                return 1 / ((epoch + 1) ** 0.5) 
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.optimizer = None

        # self._rotation_resolution = rotation_resolution
        self._batch_size = batch_size
        self.bs = batch_size
        self._num_rotation_classes = num_rotation_classes
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._voxel_size = voxel_size
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    
    def compute_loss(self, logits, labels):
        q_trans, rot_q, grip_q, collision_q = logits
        
        (action_trans, 
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot) = labels

        # cross-entropy loss
        trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
        rot_loss_x = rot_loss_y = rot_loss_z = 0.0
        grip_loss = 0.0
        collision_loss = 0.0
        if self.add_rgc_loss:
            rot_loss_x = self._cross_entropy_loss(
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ],
                action_rot_x_one_hot.argmax(-1),
            ).mean()

            rot_loss_y = self._cross_entropy_loss(
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ],
                action_rot_y_one_hot.argmax(-1),
            ).mean()

            rot_loss_z = self._cross_entropy_loss(
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ],
                action_rot_z_one_hot.argmax(-1),
            ).mean()

            grip_loss = self._cross_entropy_loss(
                grip_q,
                action_grip_one_hot.argmax(-1),
            ).mean()

            collision_loss = self._cross_entropy_loss(
                collision_q, action_collision_one_hot.argmax(-1)
            ).mean()

        total_loss = (
            trans_loss
            + rot_loss_x
            + rot_loss_y
            + rot_loss_z
            + grip_loss
            + collision_loss
        )
        return total_loss

    def scale_logits(self, logits):
        if self.temperature is None:
            raise ValueError("Temperature not set. First run the calibration process.")
        # print('hard_temp', self.temperature)
        q_trans, rot_q, grip_q, collision_q = logits
        return [q_trans/self.temperature, rot_q/self.temperature, grip_q/self.temperature, collision_q/self.temperature]
    
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
