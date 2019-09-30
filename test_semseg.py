import argparse
import csv
import os
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as torch_data
from tensorboardX import SummaryWriter

import im2mesh.data.ct_dataloading as ct
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO

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
root = "/visinf/projects_students/VCLabOccNet/test"
out_dir = "out/semseg/onet"
batch_size = 1
backup_every = 0
exit_after = args.exit_after
epoch_limit = 100000

model_selection_metric = 'iou_complete'
model_selection_sign = 1
# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Split dataset
# 70% train, 10% val, 20% test
dataset = ct.CTImagesDataset(root, sampled_points=cfg['data']['points_subsample'])
dataset_length = len(dataset)
test_dataset = torch_data.Subset(dataset, list(range(2034, dataset_length)))

# Loader for test_dataset
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=0, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


# Model
model = config.get_model(cfg, device=device, dataset=test_dataset)

# Initialize training
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

metric_log_path = "metric_log_testing"
if os.path.exists(metric_log_path) and not os.stat(metric_log_path).st_size == 0:
    warnings.warn("Metric log file: \"" + metric_log_path + "\" already exists and is not empty.")

for i, batch in enumerate(test_loader, 0):
    eval_dict = trainer.evaluate(batch, testing=True)

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
    print('Accuracy: ', eval_dict['accuracy'])
    print('Precision: ', eval_dict['precision'])
    print('Recall: ', eval_dict['recall'])
    print('IoU inside: ', eval_dict['iou_label'])

    for k, v in eval_dict.items():
        logger.add_scalar('val/%s' % k, v, it)

