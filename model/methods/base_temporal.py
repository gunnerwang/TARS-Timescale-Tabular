from model.methods.base import Method
import time
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
import json
import os
from model.utils import (
    Averager
)

from model.lib.data import (
    Dataset_TS,
    data_nan_process,
    data_enc_process,
    num_enc_process,
    data_norm_process,
    data_label_process,
    data_loader_process_TS,
    get_categories
)

from model.soft_resets_optimizer import SoftResetsOptimizer, create_soft_resets_optimizer

class Method_Temporal(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.enable_timestamp, "requires timestamp"


    def data_format(self, is_train = True, N = None, C = None, M = None, y = None):
        """
        Format the data for training or testing.

        :param is_train: bool, whether the data is for training or testing
        :param N: dict, numerical data
        :param C: dict, categorical data
        :param y: dict, labels
        """
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N,self.num_encoder = num_enc_process(self.N,num_policy = self.args.num_policy, n_bins = self.args.config['training']['n_bins'],y_train=self.y['train'],is_regression=self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy, self.y['train'])
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)
            
            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y['train']))
            self.d_in = 0 if self.N is None else self.N['train'].shape[1]
            self.categories = get_categories(self.C)
            self.N, self.C, self.M, self.y, self.train_loader, self.val_loader, self.criterion = data_loader_process_TS(self.is_regression, (self.N, self.C), self.M, self.y, self.y_info, self.args.device, self.args.batch_size, is_train = True)

        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            N_test, _ = num_enc_process(N_test, num_policy=self.args.num_policy, n_bins=self.args.config['training']['n_bins'],y_train=None, encoder=self.num_encoder)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            _, _, _, _, self.test_loader, _ =  data_loader_process_TS(self.is_regression, (N_test, C_test), M, y_test, self.y_info, self.args.device, self.args.batch_size, is_train = False)
            if N_test is not None and C_test is not None:
                self.N_test, self.C_test = N_test['test'], C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test, self.C_test = None, C_test['test']
            else:
                self.N_test, self.C_test = N_test['test'], None
            self.M_test = M['test']
            self.y_test = y_test['test']
    
    
    def fit(self, data, info, train = True, config = None, best_epoch = None):
        """
        Fit the method to the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param train: bool, whether to train the method
        :param config: dict, configuration for the method
        :return: float, time cost
        """
        N, C, M, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset_TS(N=N, C=C, M=M, y=y, info=info)
        self.N, self.C, self.M, self.y = self.D.N, self.D.C, self.D.M, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.args.t_mean = self.D.t_mean
        self.args.t_std = self.D.t_std
        self.construct_model()
        
        # Initialize optimizer based on configuration
        optimizer_type = getattr(self.args, 'optimizer_type', 'adamw')
        
        if optimizer_type.lower() == 'soft_resets':
            # Use SoftResetsOptimizer with improved settings
            soft_resets_config = getattr(self.args, 'soft_resets_config', 'minimal')  # FIXED: Default to minimal
            self.optimizer = create_soft_resets_optimizer(
                self.model.parameters(),
                lr=self.args.config['training']['lr'],
                config_type=soft_resets_config
            )
            self.use_soft_resets = True
            print(f"Using SoftResets optimizer with config: {soft_resets_config}")
        else:
            # Use default AdamW optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.args.config['training']['lr'], 
                weight_decay=self.args.config['training']['weight_decay']
            )
            self.use_soft_resets = False
            print("Using AdamW optimizer")
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return

        time_cost = 0
        
        if best_epoch is None:
            for epoch in range(self.args.max_epoch):
                tic = time.time()
                self.train_epoch(epoch)
                self.validate(epoch)
                elapsed = time.time() - tic
                time_cost += elapsed
                print(f'Epoch: {epoch}, Time cost: {elapsed}')
                if not self.continue_training:
                    break
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
            )
        else:
            for epoch in range(best_epoch + 1):
                tic = time.time()
                self.train_epoch(epoch)
                elapsed = time.time() - tic
                time_cost += elapsed
                print(f'Epoch: {epoch}, Time cost: {elapsed}')
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
        return time_cost


    def predict(self, data, info, model_name):
        """
        Predict the results of the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param model_name: str, name of the model
        :return: tuple, (loss, metric, metric_name, predictions)
        """
        N, C, M, y = data
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()

        self.data_format(False, N, C, M, y)

        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, M, y) in tqdm(enumerate(self.test_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None  
                        
                pred = self.model(X_num, X_cat, M)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()     

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        
        return vl, vres, metric_name, test_logit


    def train_epoch(self, epoch):
        """
        Train the model for one epoch.

        :param epoch: int, the current epoch
        """
        self.model.train()
        tl = Averager()
        for i, (X, M, y) in enumerate(self.train_loader, 1):
            self.train_step = self.train_step + 1
            if self.N is not None and self.C is not None:
                X_num, X_cat = X[0], X[1]
            elif self.C is not None and self.N is None:
                X_num, X_cat = None, X
            else:
                X_num, X_cat = X, None

            loss = self.criterion(self.model(X_num, X_cat, M), y)

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i-1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        tl = tl.item()
        self.trlog['train_loss'].append(tl)    
        
        # Update gamma parameters for SoftResetsOptimizer if enabled
        if hasattr(self, 'use_soft_resets') and self.use_soft_resets:
            # FIXED: Much more conservative gamma update schedule
            update_gamma = getattr(self.args, 'update_gamma', False)  # FIXED: Default to False
            gamma_update_frequency = getattr(self.args, 'gamma_update_frequency', 20)  # FIXED: Update every 20 epochs by default
            
            if update_gamma and epoch > 0 and epoch % gamma_update_frequency == 0:
                try:
                    print(f'Attempting gamma update at epoch {epoch}...')
                    # Update gamma based on training performance
                    self.optimizer.update_gamma(self.model, self.train_loader, self.criterion, self.args.device)
                    
                    # Log gamma values for monitoring
                    if hasattr(self.optimizer, 'get_gamma_statistics'):
                        gamma_stats = self.optimizer.get_gamma_statistics()
                        if gamma_stats and gamma_stats.get('gamma_updates_enabled', False):
                            print(f'Epoch {epoch} - Gamma updated: Mean={gamma_stats["global_mean"]:.6f}, '
                                  f'Std={gamma_stats["global_std"]:.6f}, Range=[{gamma_stats["global_min"]:.6f}, {gamma_stats["global_max"]:.6f}]')
                        else:
                            print(f'Epoch {epoch} - Gamma updates disabled or no parameters found')
                    else:
                        # Fallback to old method
                        gamma_values = self.optimizer.get_gamma_values()
                        if gamma_values:
                            gamma_mean = float(np.mean(list(gamma_values.values())))
                            gamma_std = float(np.std(list(gamma_values.values())))
                            print(f'Epoch {epoch} - Gamma updated: Mean={gamma_mean:.6f}, Std={gamma_std:.6f}')
                except Exception as e:
                    print(f'Warning: Gamma update failed: {e}')
            else:
                # Just log current gamma values without updating (every 10 epochs for monitoring)
                if epoch % 10 == 0 and hasattr(self.optimizer, 'get_gamma_statistics'):
                    try:
                        gamma_stats = self.optimizer.get_gamma_statistics()
                        if gamma_stats:
                            enabled_status = "enabled" if gamma_stats.get('gamma_updates_enabled', False) else "disabled"
                            print(f'Epoch {epoch} - Gamma (no update, {enabled_status}): Mean={gamma_stats["global_mean"]:.6f}, '
                                  f'Std={gamma_stats["global_std"]:.6f}')
                    except Exception as e:
                        pass


    def validate(self, epoch):
        """
        Validate the model.

        :param epoch: int, the current epoch
        """
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, M, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None                            

                pred = self.model(X_num, X_cat, M)

                test_logit.append(pred)
                test_label.append(y)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        
        vl = self.criterion(test_logit, test_label).item()   

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        
        # Determine if we're in tuning or training phase and use appropriate metrics file
        is_tuning = hasattr(self.args, 'tune') and self.args.tune
        metrics_file = osp.join(self.args.save_path, 'metrics_tuning.json' if is_tuning else 'metrics_training.json')
        
        # Create a dictionary with all args attributes
        config_params = {}
        for key in vars(self.args):
            value = getattr(self.args, key)
            # Skip complex objects that aren't JSON serializable and functions
            if not isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))) or callable(value):
                continue
                
            # For dictionaries, ensure they're serializable (avoid nested complex objects)
            if isinstance(value, dict):
                try:
                    json.dumps(value)
                    config_params[key] = value
                except:
                    # If the dict can't be serialized, skip it
                    continue
            else:
                config_params[key] = value
        
        # Get a hash of the config to identify it consistently
        config_hash = hash(json.dumps(config_params, sort_keys=True))
        
        # Format metrics dictionary for this epoch
        metrics_dict = {name: float(value) for name, value in zip(metric_name, vres)}
        metrics_dict['epoch'] = epoch
        metrics_dict['loss'] = float(vl)
        
        # Log gamma values if using SoftResets optimizer (for monitoring, not updating)
        if hasattr(self, 'use_soft_resets') and self.use_soft_resets:
            try:
                # Use improved gamma statistics
                if hasattr(self.optimizer, 'get_gamma_statistics'):
                    gamma_stats = self.optimizer.get_gamma_statistics()
                    if gamma_stats:
                        metrics_dict['gamma_mean'] = gamma_stats['global_mean']
                        metrics_dict['gamma_std'] = gamma_stats['global_std']
                        metrics_dict['gamma_min'] = gamma_stats['global_min']
                        metrics_dict['gamma_max'] = gamma_stats['global_max']
                        
                        # Log explanation if std is 0
                        if gamma_stats['global_std'] < 1e-10:
                            print(f"  Note: Gamma std=0.0 because all layers have identical gamma values")
                            print(f"  This is expected with per_layer_gamma=False or when gamma updates are uniform")
                else:
                    # Fallback to old method
                    gamma_values = self.optimizer.get_gamma_values()
                    if gamma_values:
                        values_list = list(gamma_values.values())
                        metrics_dict['gamma_mean'] = float(np.mean(values_list))
                        metrics_dict['gamma_std'] = float(np.std(values_list))
            except Exception as e:
                print(f'Warning: Gamma logging failed: {e}')
        
        # Load existing metrics or create new structure
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
                
            # Check if we have a config_map to map hashes to sequential numbers
            if 'config_map' not in all_metrics:
                all_metrics['config_map'] = {}
                
            # Check if this config already has a number assigned
            str_hash = str(config_hash)
            if str_hash not in all_metrics['config_map']:
                # Assign the next sequential number
                next_num = len(all_metrics['config_map']) + 1
                all_metrics['config_map'][str_hash] = next_num
                
            config_key = str(all_metrics['config_map'][str_hash])
        else:
            # Initialize new metrics structure
            all_metrics = {
                'config_map': {str(config_hash): 1}
            }
            config_key = "1"  # First config
            
        # Initialize this config's entry if it doesn't exist
        if config_key not in all_metrics:
            all_metrics[config_key] = {
                'config': config_params,
                'epochs': [],
                'best_metrics': None,
                'best_epoch': None
            }
        
        # Add this epoch's metrics
        all_metrics[config_key]['epochs'].append(metrics_dict)
        
        # Update best metrics if applicable
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            all_metrics[config_key]['best_metrics'] = metrics_dict
            all_metrics[config_key]['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            if self.val_count == self.args.early_stopping:
                self.continue_training = False
            self.val_count += 1
        
        # Save metrics to JSON file
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
            
        # Still save trlog for model training state
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))