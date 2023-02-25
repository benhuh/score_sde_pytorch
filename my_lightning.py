
import torch
# import functools
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
# import tqdm

# import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningDataModule
from typing import Tuple, List, Dict, Any, Union, Optional



from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # Callback, 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from argparse import ArgumentParser, Namespace

from my_model import ScoreNet
from my_utils import ssim_fn, psnr_fn #, CustomProgressBar


class LITmodel(LightningModule):
    def __init__(self, hparams: Namespace): #, device) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        assert hparams['model_type'] == 'ScoreNet'
        self.model = ScoreNet(**hparams) 

        # if isinstance(hparams,dict):
        #     hparams = Namespace(**hparams)

    def forward(self, x, t, *args, **kwargs) -> Any:
        score_std = self.model(x, t, *args, **kwargs) 
        # std = self.marginal_prob_std(t)[:, None, None, None]
        # score = score_std / std  # Normalize output    ##  score = -grad_log_P \approx  - noise_pred/std**2 =  -(x_noisy - x) / std**2  = - z /std   
        return score_std

    @property
    def current_step(self) -> int:
        return self.trainer.fit_loop.total_batch_idx + 1
    # @property
    # def current_epoch(self) -> int:
    #     return int(super().current_epoch/2)  # Hacky fix for factor of 2 in epoch ...                      

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict]]:
        param_groups = [{'params': self.model.parameters()},]

        # depth = self.model.depth  #self.hparams.depth 
        # gam = 1 - 2/depth
        weight_decay = self.hparams.weight_decay
        # weight_decay = gam**(gam) * (weight_decay/(1+gam))**(1+gam)

        # if self.hparams.optim == 'AdamW':
        optimizer = AdamW(param_groups, lr=self.hparams.lr, weight_decay=weight_decay) 
        schedulers = []
        # schedulers = [ {"scheduler": wd_scheduler.MultiStepWeightDecay(optimizer, milestones=self.hparams.step_schedule, gamma=0.5), "interval": "step", "frequency": 1, },    #  'name': 'monitor/weight decay', 
        #                {"scheduler": lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.step_schedule, gamma=0.5),  "interval": "step", "frequency": 1, },  #  , 'name': 'monitor/lr' ]

        return [optimizer], schedulers

    # def on_train_end(self):
    #     torch.save(self.model, "model.pt"))
    #     torch.save(self.model, self.hparams.logger_dir+ "/final.pt")
    #     pass
    
    def marginal_prob_std(self, t):         
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.  """  
        return torch.sqrt((self.hparams.sigma**(2 * t) - 1.) / 2. / np.log(self.hparams.sigma))

    # def diffusion_coeff(t):               """Compute the diffusion coefficient of our SDE.   """
    #     return tensorize(self.hparams.sigma**t, device=device)
    
    def loss_fn(self, score, noise):
        # return torch.mean(torch.sum((score - noise)**2, dim=(1,2,3))) 
        return F.mse_loss(score, noise, reduction="mean") / 2
    
    def _step(self, data, batch_idx):
        eps=1e-5        # eps: A tolerance value for numerical stability.
        images, labels = data
        
        t = torch.rand(images.shape[0], device=images.device) * (1. - eps) + eps  # choose a random time point
        noise_std = self.marginal_prob_std(t)[:, None, None, None]
        noise = noise_std * torch.randn_like(images)
        noisy_images = images + noise
        out = self.model(noisy_images, t)   # score * std   \approx - noise_pred/std =  -(x_noisy - x) /std  = - z    # score = - grad_log_P 

        noise_pred = out                         # = scores * std**2  = - grad_log_P * std**2 
        loss = self.loss_fn(noise_pred, noise)   # works best: model output directly predicts the noise

        ## original version: out = scores * std 
        # noise_pred = out * noise_std           # = scores * std**2  = - grad_log_P * std**2 
        # loss = self.loss_fn(noise_pred/noise_std, noise/noise_std)

        denoised_images = noisy_images - noise_pred  # ideally denoised image? 
        
        batch = images.size(0) 
        psnr = psnr_fn(denoised_images, images)
        ssim = ssim_fn(denoised_images, images)

        info = {"batch": batch,
                "loss":  loss,
                "psnr": torch.Tensor([psnr]),
                "ssim": torch.Tensor([ssim]),
                }      
        return info, noisy_images, denoised_images
    
    

    def logging_step(self, info, added_key='', prog_bar=False):
        for k, v in info.items():
            if k != 'batch':
                self.log(k+added_key, v, on_step=True, on_epoch=False, prog_bar=prog_bar)

    def training_step(self, data, batch_idx):
        info, _, _ = self._step(data, batch_idx=batch_idx) #, mode = 'train')
        
        self.logging_step(info, added_key='/train') #, prog_bar=True)
        return info

    def validation_step(self, data, batch_idx):
        with torch.no_grad():
            info, noisy_images, denoised_images = self._step(data, batch_idx=batch_idx) #, mode = 'val') #'S')
            images = torch.stack([data[0], noisy_images, denoised_images], dim=1)
            images = images[:10]

        self.logging_step(info, added_key='/val', prog_bar=True)

        for id, image in enumerate(images):
            image = make_grid(image.clamp(0, 1), nrow=3, normalize=False)
            self.logger.experiment.add_image(f"valid_samples/[{id}]", image, self.current_step, dataformats='CHW')        

        return info


class DataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for k,v in kwargs.items():
            setattr(self, k, v)
        
    def prepare_data(self):  # download image dataset
        if self.task == 'MNIST':
            datasets = MNIST
            
        datasets(self.data_dir, train=True, download=True)
        datasets(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):          # Assign train/val datasets for use in dataloaders
        self.setup_helper(stage)
        self.setup_done=True
    
    def setup_helper(self, stage):
        if self.task == 'MNIST':
            datasets = MNIST
            transform = transforms.Compose([transforms.ToTensor()]) #,  transforms.Normalize((0.1307,), (0.3081,)), ])
                                            
        self.train_dataset = datasets(root = self.data_dir, train = True, transform=transform, target_transform=None, download=False)
        self.val_dataset = datasets(root = self.data_dir, train = False, transform=transform, target_transform=None, download=False)
        
        max_val_num = getattr(self,'max_val_num',False)
        if max_val_num:
            self.val_dataset.data    = self.val_dataset.data[:max_val_num]
            self.val_dataset.targets = self.val_dataset.targets[:max_val_num]     
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batchsize, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batchsize, num_workers=self.num_workers, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self.val_dataloader()

def get_datamodule(hparams):
    if isinstance(hparams,dict):
        hparams = Namespace(**hparams)

    seed = 0 #hparams.random_seed 
    # if seed > 0 and getattr(hparams,'trial_num',None) is not None:
    #     seed += hparams.trial_num

    dm_args = { "data_dir": hparams.data_path,  
                # "loss_type": hparams.loss_type,
                "batchsize": hparams.batch_size,
                "val_batchsize": hparams.val_batch_size,
                "seed": seed,
                "task": hparams.dataset,
                "num_workers": hparams.num_workers,
                # "prepare_data_per_node":  getattr(hparams, 'prepare_data_per_node', 'False'),
                }

    optional = [ ]
    for k in optional:
        dm_args[k] = getattr(hparams, k, None) 

    dm =  DataModule(**dm_args)
    return dm
               
def get_trainer(hparams, ckpt_path=None):
    if isinstance(hparams,dict):
        hparams = Namespace(**hparams)

    # utils.setup_experiment(hparams)
    log_name = hparams.experiment_dir # get_log_name(hparams)

    version = getattr(hparams, 'logger_version', None)
    # print(hparams)
    logger = TensorBoardLogger(hparams.logdir, name = log_name, default_hp_metric=False, version=version) 

    trainer_args = {
        # "max_steps": hparams.max_steps,
        "max_epochs": hparams.max_epochs, # int(1e8),
        "val_check_interval": hparams.val_check_interval,  #10,
        "check_val_every_n_epoch": hparams.check_val_every_n_epoch,
        "profiler": False,
        "logger": logger,
        "log_every_n_steps": 5, #10,
        # "flush_logs_every_n_steps": 1000,
        "callbacks": [CustomProgressBar(refresh_rate=1, enable_val=hparams.enable_val_progress), 
                        # WeightDecayMonitor(logging_interval='step'),  
                        LearningRateMonitor(logging_interval='step'),
                    #   earlystop_callback,
                        ],
        "gradient_clip_val": hparams.grad_clip, #0.1/5, 
        "gradient_clip_algorithm": "norm", #"value"
        "profiler": "pytorch" if getattr(hparams,'profiler',False) else None,
        }

    if ckpt_path is not None:
        trainer_args.update({'resume_from_checkpoint': ckpt_path})
        # trainer_args.update({'ckpt_path': ckpt_path})  # For Next version of Trainer.

    if torch.cuda.is_available() and len(hparams.gpus) > 0:
        trainer_args.update({ "accelerator": "gpu", "devices": hparams.gpus,        # "strategy": "ddp", 
                                })
    trainer = Trainer(**trainer_args)      # trainer = Trainer(**trainer_args)
    return trainer 


##################################

import sys
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

class CustomProgressBar(TQDMProgressBar):
    def __init__(self, enable_val=False, *args, **kwargs):
        self.enable_validation_bar=enable_val
        super().__init__(*args, **kwargs)
        
    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=not self.enable_validation_bar, #self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

