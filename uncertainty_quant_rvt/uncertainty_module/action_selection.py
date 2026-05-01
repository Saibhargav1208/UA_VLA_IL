import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
from itertools import product
from tqdm import tqdm
import numpy as np

class ActionSelection:
    def __init__(self, device, 
                batch_size, 
                img_num: int = 5,
                tau: float = 3,
                trans_conf_thresh: float = 1e-5,
                rot_conf_thresh: float = 1/72,
                search_size: int = 20,
                search_step: int = 2,
                log_dir: str = None,
                enabled: bool = True,
                action_type = None,
                use_ua_rot: bool = False,
                tau_rot: float = 3,
                sigma: float = 3,
                uncertainty_measure: bool = False
                ):
        self.device = device

        self._batch_size = batch_size
        self.bs = batch_size

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        self._img_num = img_num
        self._tau = tau
        self._trans_conf_thresh = trans_conf_thresh
        self._rot_conf_thresh = rot_conf_thresh
        self._search_size = search_size
        self._search_step = search_step
        self.log_dir = log_dir
        self.enabled = enabled
        self.action_type = action_type
        self.use_ua_rot = use_ua_rot
        self._tau_rot = tau_rot
        self._sigma = sigma
        self.uncertainty_measure = uncertainty_measure
        # self._trans_search_size = 0 #2*2 # 5*2
        # self._rot_search_size = 0 # 2*2 # 6*2
        # self._tran_size = 100-1
        # self._rot_size = 72-1
        self.conv = nn.Conv2d(in_channels=self._img_num, 
                    out_channels=self._img_num, 
                    kernel_size=self._tau, 
                    stride=1, 
                    padding=self._tau//2, 
                    groups=self._img_num, 
                    bias=False,
                    device=self.device)
        with torch.no_grad():
            if action_type == 'gaussian':
                print('using gaussian')
                self._init_with_2Dgaussian() # use 2D for translation
                print(self.conv.weight.detach().cpu().numpy()[0,0,:,:])
            else:
                print('using accumulation')
                self.conv.weight.fill_(1.0)
        
        if use_ua_rot:
            # self.conv_rot = torch.ones(1, 1, self._tau_rot)
            self.conv_rot = nn.Conv1d(
                                in_channels=3,
                                out_channels=3,
                                kernel_size=self._tau_rot, 
                                stride=1, 
                                padding=self._tau_rot//2, 
                                groups=3,
                                bias=False).to(self.device)
            with torch.no_grad():
                if action_type == 'gaussian':
                    print('using gaussian')
                    self._init_with_1Dgaussian() # use 1D for rotation
                    print(self.conv_rot.weight.detach().cpu().numpy())
                else:
                    print('using accumulation')
                    self.conv_rot.weight.fill_(1.0)
            
    def get_uncertainty_heatmap(self, hm):
        # mask = hm <= 1/220**2
        # hm[mask] = 0
        return self.conv(hm) #/self._tau**2
    
    # Define the 2D Gaussian function
    def _init_with_2Dgaussian(self):
        kernel = self._create_standard_2d_gaussian(self._tau, self._sigma)
        with torch.no_grad():
            self.conv.weight.data.copy_(torch.tensor(kernel)[None, None, ...].expand(self._img_num, -1, -1, -1))
    
    def _init_with_1Dgaussian(self):
        kernel = self._create_standard_1d_gaussian(self._tau_rot, self._sigma)
        with torch.no_grad():
            self.conv_rot.weight.data.copy_(torch.tensor(kernel)[None, ...].expand(3, -1, self._tau_rot))

    @staticmethod
    def _create_standard_1d_gaussian(kernel_size, sigma):
        half_size = kernel_size // 2
        x = np.linspace(-half_size, half_size, kernel_size)
        kernel = np.exp(-(x**2 / (2 * sigma**2)))
        kernel /= kernel[half_size]
        return kernel

    @staticmethod
    def _create_standard_2d_gaussian(kernel_size, sigma):
        # Generate a grid of x and y values
        half_size = kernel_size // 2
        x = np.linspace(-half_size, half_size, kernel_size)
        y = np.linspace(-half_size, half_size, kernel_size)
        xx, yy = np.meshgrid(x, y)
        # Create the 2D Gaussian pattern
        kernel = np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
        # Normalize so the center has a value of 1
        kernel /= kernel[half_size, half_size]
        return kernel

    
    def get_uncertainty_rot(self, rot_q, num_rotation_classes):
        self._num_rotation_classes = num_rotation_classes
        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )

        axis_rot = torch.vstack([
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ],
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ],
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ],
            ]
        )
        rot_ua = self.conv_rot(axis_rot.unsqueeze(0)).squeeze(0)
        rot_ua_reconstructed = rot_ua.reshape(3, -1, self._num_rotation_classes).transpose(0, 1).reshape(-1, 3 * self._num_rotation_classes)
        # pred_rot_out = torch.cat(
        #     (
        #         rot_q_reconstructed[
        #             :,
        #             0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
        #         ].argmax(1, keepdim=True),
        #         rot_q_reconstructed[
        #             :,
        #             1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
        #         ].argmax(1, keepdim=True),
        #         rot_q_reconstructed[
        #             :,
        #             2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
        #         ].argmax(1, keepdim=True),
        #     ),
        #     dim=-1,
        # )

        # print('pred_rot', pred_rot)
        # print('pred_rot_out', pred_rot_out)
        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # # Create the heatmap visualization
        # def combined_plot_heatmap(input_data, output_data, title):
        #     # # Stack the input and output tensors vertically
        #     # combined_data = torch.cat((input_data, output_data), dim=0)
            
        #     # plt.figure(figsize=(15, 10))
            
        #     # # Use the aspect parameter to ensure uniform cell size
        #     # ax = sns.heatmap(combined_data, cmap='YlGnBu', aspect="auto")
            
        #     # plt.title(title)
        #     # plt.ylabel('Rows')
        #     # plt.xlabel('Columns')
            
        #     # # Use the ax object to create the colorbar
        #     # plt.colorbar(ax.collections[0])
        #     # plt.tight_layout()
            
        #     # Stack the input and output tensors vertically
        #     combined_data = torch.cat((input_data, torch.ones(input_data.shape), output_data), dim=0)
            
        #     plt.figure(figsize=(15, 10))
            
        #     # Plot the heatmap without the aspect parameter
        #     ax = sns.heatmap(combined_data, cmap='YlGnBu')
            
        #     # Adjust the aspect of the axes object directly
        #     ax.set_aspect('equal')
            
        #     plt.title(title)
        #     plt.ylabel('Rows')
        #     plt.xlabel('Columns')
            
        #     # Use the ax object to create the colorbar
        #     plt.colorbar(ax.collections[0])
        #     plt.tight_layout()
        
        # # Plot the input_tensor
        # combined_plot_heatmap(axis_rot.cpu(), rot_out.cpu(), 'input/output Tensor')
        
        # # Save the visualization to the specified path
        # save_path = "/home/bobwu/shared/rvt/debug/rot_debug_heatmap.png"
        # plt.savefig(save_path)
        # plt.close()
        # import pdb
        # pdb.set_trace()
        return rot_ua_reconstructed


    #     self.conv = nn.Conv2d(in_channels=self._img_num, 
    #                           out_channels=self._img_num, 
    #                           kernel_size=self._tau, 
    #                           stride=1, 
    #                           padding=self._tau//2, 
    #                           groups=self._img_num, 
    #                           bias=False,
    #                           device=self.device)
        
    #     # Adjust the convolution weights by dividing by tau^2
    #     with torch.no_grad():
    #         self.conv.weight.fill_(1.0 / self._tau**2)
            
    # def get_uncertainty_heatmap(self, hm):
    #     convolved_hm = self.conv(hm)
        
    #     # Normalize the output to be between 0 and 1
    #     max_val = convolved_hm.max()
    #     min_val = convolved_hm.min()
    #     normalized_hm = (convolved_hm - min_val) / (max_val - min_val)
        
    #     return normalized_hm