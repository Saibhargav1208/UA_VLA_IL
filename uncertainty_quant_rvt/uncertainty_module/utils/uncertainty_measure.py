

class UncertaintyMeasurement:
    def __init__(self):
        # uncertainty_measure initialization with tasks
        self.trans_task_total_max_var = []
        self.trans_task_total_max_entropy = []
        self.trans_task_total_avg_var = []
        self.trans_task_total_avg_entropy = []
        
        self.rot_task_total_max_var = []
        self.rot_task_total_max_entropy = []
        self.rot_task_total_avg_var = []
        self.rot_task_total_avg_entropy = []    
        
        self.grip_task_total_max_var = []
        self.grip_task_total_max_entropy = []
        self.grip_task_total_avg_var = []
        self.grip_task_total_avg_entropy = []
        
        self.coll_task_total_max_var = []
        self.coll_task_total_max_entropy = []
        self.coll_task_total_avg_var = []
        self.coll_task_total_avg_entropy = []
        
        self.trans_task_total_min_conf = []
        self.trans_task_total_avg_conf = []
        self.rot_task_total_min_conf = []
        self.rot_task_total_avg_conf = []
        self.grip_task_total_min_conf = []
        self.grip_task_total_avg_conf = []
        self.coll_task_total_min_conf = []
        self.coll_task_total_avg_conf = []
        
        self.task_total_success = []
        
    def episode_reset(self):
        self.trans_episode_max_var = -1
        self.trans_episode_max_entropy = -1
        self.trans_episode_total_var = 0
        self.trans_episode_total_entropy = 0
        
        self.rot_episode_max_var = -1
        self.rot_episode_max_entropy = -1
        self.rot_episode_total_var = 0
        self.rot_episode_total_entropy = 0
        
        self.grip_episode_max_var = -1
        self.grip_episode_max_entropy = -1
        self.grip_episode_total_var = 0
        self.grip_episode_total_entropy = 0

        self.coll_episode_max_var = -1
        self.coll_episode_max_entropy = -1
        self.coll_episode_total_var = 0
        self.coll_episode_total_entropy = 0
        
        self.trans_episode_min_conf = 1
        self.trans_episode_total_conf = 0
        self.rot_episode_min_conf = 1
        self.rot_episode_total_conf = 0
        self.grip_episode_min_conf = 1
        self.grip_episode_total_conf = 0
        self.coll_episode_min_conf = 1
        self.coll_episode_total_conf = 0

        self.num_steps = 0

    def episode_update(self, info):
        self.trans_episode_max_var = max(self.trans_episode_max_var, info['hm_var'])
        self.trans_episode_max_entropy = max(self.trans_episode_max_entropy, info['hm_entropies'])
        self.trans_episode_total_var += info['hm_var']
        self.trans_episode_total_entropy += info['hm_entropies']
            
        self.rot_episode_max_var = max(self.rot_episode_max_var, info['rot_var'])
        self.rot_episode_max_entropy = max(self.rot_episode_max_entropy, info['rot_entropies'])
        self.rot_episode_total_var += info['rot_var']
        self.rot_episode_total_entropy += info['rot_entropies']
        
        self.grip_episode_max_var = max(self.grip_episode_max_var, info['grip_var'])
        self.grip_episode_max_entropy = max(self.grip_episode_max_entropy, info['grip_entropy'])
        self.grip_episode_total_var += info['grip_var']
        self.grip_episode_total_entropy += info['grip_entropy']

        self.coll_episode_max_var = max(self.coll_episode_max_var, info['collision_var'])
        self.coll_episode_max_entropy = max(self.coll_episode_max_entropy, info['collision_entropy'])
        self.coll_episode_total_var += info['collision_var']
        self.coll_episode_total_entropy += info['collision_entropy'] 
        
        self.trans_episode_min_conf = min(self.trans_episode_min_conf, info['pred_wpt_conf'])
        self.trans_episode_total_conf += info['pred_wpt_conf']
        self.rot_episode_min_conf = min(self.rot_episode_min_conf, info['pred_rot_conf'])
        self.rot_episode_total_conf += info['pred_rot_conf']
        self.grip_episode_min_conf = min(self.grip_episode_min_conf, info['pred_grip_conf'])
        self.grip_episode_total_conf += info['pred_grip_conf']
        self.coll_episode_min_conf = min(self.coll_episode_min_conf, info['pred_coll_conf'])
        self.coll_episode_total_conf += info['pred_coll_conf']
        
        print('trans_episode_min_conf', self.trans_episode_min_conf)
        self.num_steps += 1
    
    
    # def save_results(self, file_path, _file_name):
    #     def compute_avg(values, mask):
    #         filtered_values = [values[i] for i in range(len(values)) if mask[i]]
    #         return sum(filtered_values) / len(filtered_values) if filtered_values else 0

    def compute_avg(self, values, mask):
        filtered_values = [v for v, m in zip(values, mask) if m]
        return sum(filtered_values) / len(filtered_values) if filtered_values else 0

    # Updated method to include min_conf and avg_conf
    def print_and_format_averages(self, file, prefix, max_var, max_entropy, avg_var, avg_entropy, min_conf, avg_conf, mask):
        avg_max_var = self.compute_avg(max_var, mask)
        avg_max_entropy = self.compute_avg(max_entropy, mask)
        avg_avg_var = self.compute_avg(avg_var, mask)
        avg_avg_entropy = self.compute_avg(avg_entropy, mask)
        avg_min_conf = self.compute_avg(min_conf, mask)
        avg_avg_conf = self.compute_avg(avg_conf, mask)

        print(f"{prefix}_avg_max_var {avg_max_var}")
        print(f"{prefix}_avg_max_entropy {avg_max_entropy:.2f}")
        print(f"{prefix}_avg_avg_var {avg_avg_var}")
        print(f"{prefix}_avg_avg_entropy {avg_avg_entropy:.2f}")
        print(f"{prefix}_avg_min_conf {avg_min_conf}")
        print(f"{prefix}_avg_avg_conf {avg_avg_conf}")

        file.write(f"{prefix}_avg_max_var {avg_max_var}\n")
        file.write(f"{prefix}_avg_max_entropy {avg_max_entropy:.2f}\n")
        file.write(f"{prefix}_avg_avg_var {avg_avg_var}\n")
        file.write(f"{prefix}_avg_avg_entropy {avg_avg_entropy:.2f}\n")
        file.write(f"{prefix}_avg_min_conf {avg_min_conf}\n")
        file.write(f"{prefix}_avg_avg_conf {avg_avg_conf}\n")

    # Updated save_results method
    def save_results(self, file_path, _file_name, eval_task_name):
        file_name = f"uncertainty_measure_{_file_name}"
        with open(file_path + file_name, "w") as file:
            file.write(f"Task: {eval_task_name}\n")

            action_types = ['trans', 'rot', 'grip', 'coll']
            conditions = ['task', 'success', 'failed']
            masks = {
                'task': [1] * len(self.task_total_success),
                'success': self.task_total_success,
                'failed': [1 - s for s in self.task_total_success]
            }

            for action_type in action_types:
                for condition in conditions:
                    prefix = f"{action_type}_{condition}"
                    max_var = getattr(self, f"{action_type}_task_total_max_var")
                    max_entropy = getattr(self, f"{action_type}_task_total_max_entropy")
                    avg_var = getattr(self, f"{action_type}_task_total_avg_var")
                    avg_entropy = getattr(self, f"{action_type}_task_total_avg_entropy")
                    min_conf = getattr(self, f"{action_type}_task_total_min_conf")
                    avg_conf = getattr(self, f"{action_type}_task_total_avg_conf")
                    mask = masks[condition]
                    
                    self.print_and_format_averages(file, prefix, max_var, max_entropy, avg_var, avg_entropy, min_conf, avg_conf, mask)

    def task_update(self, reward):
        self.trans_task_total_max_var.append(self.trans_episode_max_var)
        self.trans_task_total_max_entropy.append(self.trans_episode_max_entropy)
        self.trans_task_total_avg_var.append(self.trans_episode_total_var / self.num_steps)
        self.trans_task_total_avg_entropy.append(self.trans_episode_total_entropy / self.num_steps)
        
        
        self.rot_task_total_max_var.append(self.rot_episode_max_var)
        self.rot_task_total_max_entropy.append(self.rot_episode_max_entropy)
        self.rot_task_total_avg_var.append(self.rot_episode_total_var / self.num_steps)
        self.rot_task_total_avg_entropy.append(self.rot_episode_total_entropy / self.num_steps)
        
        self.grip_task_total_max_var.append(self.grip_episode_max_var)
        self.grip_task_total_max_entropy.append(self.grip_episode_max_entropy)
        self.grip_task_total_avg_var.append(self.grip_episode_total_var / self.num_steps)
        self.grip_task_total_avg_entropy.append(self.grip_episode_total_entropy / self.num_steps)
        
        self.coll_task_total_max_var.append(self.coll_episode_max_var)
        self.coll_task_total_max_entropy.append(self.coll_episode_max_entropy)
        self.coll_task_total_avg_var.append(self.coll_episode_total_var / self.num_steps)
        self.coll_task_total_avg_entropy.append(self.coll_episode_total_entropy / self.num_steps)
        
        self.trans_task_total_min_conf.append(self.trans_episode_min_conf)
        self.trans_task_total_avg_conf.append(self.trans_episode_total_conf / self.num_steps)
        self.rot_task_total_min_conf.append(self.rot_episode_min_conf)
        self.rot_task_total_avg_conf.append(self.rot_episode_total_conf / self.num_steps)
        self.grip_task_total_min_conf.append(self.grip_episode_min_conf)
        self.grip_task_total_avg_conf.append(self.grip_episode_total_conf / self.num_steps)
        self.coll_task_total_min_conf.append(self.coll_episode_min_conf)
        self.coll_task_total_avg_conf.append(self.coll_episode_total_conf / self.num_steps)
        
        task_success = 1 if reward > 50.0 else 0
        self.task_total_success.append(task_success)