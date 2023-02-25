
# from my_utils import get_exp_str #, CustomProgressBar
# from my_lightning import LITmodel, get_datamodule, get_trainer

# def get_hparams():

#     BIAS = False
#     # eps=1.0 #eps = 1e-5
#     # group_norm_args = dict(bias = False, weight=False, eps=eps)
#     group_norm_args = dict(identity=True)

#     ## learning rate
#     lr=1e-3 #@param {'type':'number'}
#     ## weight decay
#     wd = 0  #@param {'type':'number'} #1e-3

#     sigma =  25.0 #@param {'type':'number'}

#     hparams_model = dict(model_type = 'ScoreNet', sigma=sigma, bias=BIAS, group_norm_args=group_norm_args, lr = lr, weight_decay=wd,
#                         channels=[32, 64, 128, 256], embed_dim=2)                    # channels=[32, 64, 128, 256], embed_dim=256 )

#     exp_str = get_exp_str(hparams_model)


#     gpus = [1]  # device = 'cuda:0' #@param ['cuda:0', 'cuda:1', 'cpu'] {'type':'string'}
#     n_epochs =  50#@param {'type':'integer'}
#     grad_clip = 1e-3
#     hparams_trainer = dict(max_epochs = n_epochs, logdir = 'lightning_logs', experiment_dir = exp_str, gpus = gpus, 
#                         val_check_interval = None, 
#                         check_val_every_n_epoch = 1,
#                         enable_val_progress = True, grad_clip = grad_clip, max_val_num=32)


#     ## size of a mini-batch
#     batch_size =  64 #@param {'type':'integer'}

#     hparams_dm = dict(data_path = 'data', dataset = 'MNIST', num_workers = 16, batch_size = batch_size, val_batch_size = 5000)

#     return hparams_model, hparams_dm, hparams_trainer

# #######################

# def main():
#     hparams_model, hparams_dm, hparams_trainer = get_hparams()
#     model = LITmodel(hparams_model)
#     datamodule = get_datamodule(hparams_dm)
#     trainer = get_trainer(hparams_trainer)

#     trainer.fit(model=model, datamodule=datamodule) 
#     # this breaks somehow
    
# if __name__ == "__main__":
#     main()
