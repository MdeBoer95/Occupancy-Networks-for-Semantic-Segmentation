import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from im2mesh.onet.generation import Generator3D
import numpy as np
import matplotlib.pyplot as plt


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    acc = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    return true_positives, false_positives, true_negatives, false_negatives, acc


class Trainer(BaseTrainer):
    """ Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    """

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        print("Threshold: ", threshold)
        self.eval_sample = eval_sample
        '''
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        '''

    def train_step(self, data):
        """ Performs a training step.

        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
        # Original Code
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict'''
        # Our Code

        device = self.device
        generator = Generator3D(self.model, device=device)
        self.model.eval()
        threshold = self.threshold
        eval_dict = {}

        # Sampled points evaluation:
        with torch.no_grad():
            smooth = 1e-6
            points = data.get('points').to(device)
            occ = data.get('points.occ').to(device)
            inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
            kwargs = {}
            p_out = self.model(points, inputs,
                               sample=self.eval_sample, **kwargs)
            probabilities = p_out.probs
            occ_pred = (probabilities >= threshold).float()
            acc = (occ_pred == occ).sum().float() / occ.numel()
            acc = acc.cpu().numpy()
            metrics = confusion(occ, occ_pred)
            eval_dict['points_accuracy'] = acc
            eval_dict['tp'] = np.array(metrics[0])
            eval_dict['fp'] = np.array(metrics[1])
            eval_dict['tn'] = np.array(metrics[2])
            eval_dict['fn'] = np.array(metrics[3])
            eval_dict['precision'] = (eval_dict['tp'] + smooth) / ((eval_dict['tp'] + eval_dict['fp']) + smooth)
            eval_dict['recall'] = (eval_dict['tp'] + smooth) / ((eval_dict['tp'] + eval_dict['fn']) + smooth)

        # Value grid evaluation:

        # generates mesh from 2nd sample in dataloader
        mesh, stats, occ_grid = generator.generate_mesh(data)

        # calculate simple IoU:
        def label_iou(guessed_label, ground_truth_label, s):
            """
            Function to calculate simple IoU between a label and a prediction, cut down to the label
            :param guessed_label: predicted value grid, cropped to label shape at label position
            :param ground_truth_label: Bit volume
            :param s: smoothing factor to prevent division by 0
            :return: simple IoU between guessed_grid and ground_truth_label, cropped to label
            """
            intersection = (ground_truth_label & guessed_label).sum() + s
            union = (ground_truth_label | guessed_label).sum() + s

            return intersection / union

        def overall_iou(guessed_grid, guessed_label, ground_truth_label, s):
            """
            Function to calculate simple IoU between label and prediction
            :param guessed_grid: predicted value grid
            :param ground_truth_label: Bit volume
            :param guessed_label: predicted value grid, cropped to label shape at label position
            :param s: smoothing factor to prevent division by 0
            :return: simple IoU between guessed grid and ground truth
            """
            intersection = (ground_truth_label & guessed_label).sum() + s
            union = (ground_truth_label | guessed_label).sum() + (guessed_grid.sum() - guessed_label.sum()) + s
            return intersection / union

        # remove padding from grid
        occ_pred = (occ_grid >= threshold).astype(int)[:640, :448, :512]
        # Get label data
        offset = data.get('label_offset')[1].numpy().astype(int)
        shape = data.get('label_shape')[1].numpy()
        # get the part from occ_grid where the label should be
        occ_pred_label = \
            occ_pred[offset[0]:offset[0] + shape[0], offset[1]:offset[1] + shape[1], offset[2]:offset[2] + shape[2]]
        # remove padding from label
        label = data.get('padded_label')[1].numpy()[:shape[0], :shape[1], :shape[2]]

        eval_dict['iou_label'] = label_iou(occ_pred_label, label, smooth)
        eval_dict['iou_complete'] = overall_iou(occ_pred, occ_pred_label, label, smooth)

        # Visualization, set debug point at return

        print('Started label')
        label_truth = np.zeros((640, 448, 512))
        label_truth[offset[0]:offset[0] + shape[0], offset[1]:offset[1] + shape[1],
        offset[2]:offset[2] + shape[2]] = label
        X_pred = []
        Y_pred = []
        Z_pred = []
        X_lab = []
        Y_lab = []
        Z_lab = []
        print('Started for loops')
        for x in range(occ_pred_label.shape[0]):
            print('X: ', x)
            for y in range(occ_pred_label.shape[1]):
                print('Y: ', y)
                for z in range(occ_pred_label.shape[2]):
                    if occ_pred_label[x, y, z] == 1:
                        X_pred.append(int(x))
                        Y_pred.append(int(y))
                        Z_pred.append(int(z))
                    if label[x, y, z] == 1:
                        X_lab.append(int(x))
                        Y_lab.append(int(y))
                        Z_lab.append(int(z))
        # plt.interactive(False)

        print('For loop done. Started scattering')

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(np.array(X_pred), np.array(Y_pred), np.array(Z_pred), marker=',', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, occ_pred_label.shape[0])
        ax.set_ylim(0, occ_pred_label.shape[1])
        ax.set_zlim(0, occ_pred_label.shape[2])

        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.scatter(np.array(X_lab), np.array(Y_lab), np.array(Z_lab), marker=',', alpha=0.5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(0, occ_pred_label.shape[0])
        ax1.set_ylim(0, occ_pred_label.shape[1])
        ax1.set_zlim(0, occ_pred_label.shape[2])
        print('Done scattering')
        # rotate the axes and update
        for angle in range(0, 360, 60):
            ax.view_init(30, angle)
            ax1.view_init(30, angle)
            plt.draw()
            plt.savefig('img_without_mise' + str(angle))

        exit()

        '''
        # Recalculate threshold
        threshold_grid = np.log(threshold) - np.log(1. - threshold) # Always 0?
        occ_grid_pred = np.zeros(occ_grid.shape)
        occ_grid_idx = occ_grid >= threshold
        occ_grid[occ_grid_idx] = 1
        print(np.count_nonzero(occ_grid_pred))
        '''

        return eval_dict

    def visualize(self, data):
        """
         Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        """
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid((-0.5,) * 3, (0.5,) * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        """ Computes the loss.

        Args:
            data (dict): data dictionary
            :param data:
            :param self:
        """

        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.model.decode(p, z, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss
