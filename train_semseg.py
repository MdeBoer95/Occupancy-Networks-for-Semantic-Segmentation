import torch
import torch.optim as optim
import torch.utils.data as torch_data
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib; matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
import im2mesh.data.ct_dataloading as ct
import json
import math
import csv
import warnings

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config('configs/semseg/onet.yaml', 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands hardcoded
root = "/visinf/projects_students/VCLabOccNet/preprocessedSamples"
out_dir = "out/semseg/onet"
batch_size = 64
backup_every = 500
exit_after = args.exit_after
epoch_limit = 1000000

model_selection_metric = "points_accuracy"
model_selection_sign = 1

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Split dataset
# 70% train, 10% val, 20% test
dataset = ct.CTImagesDataset(root)
dataset_length = len(dataset)
train_length = math.floor(0.7*dataset_length)
val_length = math.ceil(0.1*dataset_length)
test_length = math.floor(0.2*dataset_length)
print("Dataset lengths:", "Train:", train_length, "Val:", val_length, "Test:", test_length, "Total:", dataset_length)
train_dataset, val_dataset, test_dataset = torch_data.random_split(dataset, [train_length, val_length, test_length])

# Loader for train_dataset
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


# Loader for val_dataset
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=10, num_workers=4, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
'''
# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=12, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))
'''
# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# Hack because of previous bug in code
# TODO: remove, because shouldn't be necessary
if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

# TODO: remove this switch
# metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

# TODO: reintroduce or remove scheduler?
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
#                                       gamma=0.1, last_epoch=epoch_it)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
# print(model)
print('Total number of parameters: %d' % nparameters)

metric_log_path = "metric_log"
if os.path.exists(metric_log_path) and not os.stat(metric_log_path).st_size == 0:
    warnings.warn("Metric log file: \"" + metric_log_path + "\" already exists and is not empty.")

while True:
    epoch_it += 1
#     scheduler.step()
    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))
        '''
        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            trainer.visualize(data_vis)
        '''
        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                           loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)

            with open(metric_log_path, 'a') as metric_log_file:
                metric_logger = csv.writer(metric_log_file)
                if os.stat(metric_log_path).st_size == 0:
                    header = list(eval_dict.keys())
                    header.insert(0, 'iteration')
                    metric_logger.writerow(header)
                row = list(eval_dict.values())
                row.insert(0, it)
                metric_logger.writerow(row)

            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after or epoch_it >= epoch_limit:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            metric_log_file.close()
            exit(3)