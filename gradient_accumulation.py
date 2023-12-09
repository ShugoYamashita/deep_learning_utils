import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# --- Parse hyper-parameters  --- #
def get_args_parser():
    parser = argparse.ArgumentParser(description='Hyper-parameters for network')
    parser.add_argument('-train_batch_size', default=32, type=int, help='Set the training batch size')
    parser.add_argument('-update_freq', default=1, type=int, help='gradient accumulation steps')
    return parser

def train_one_epoch_accum(net, data_loader, optimizer, epoch, device, update_freq=2, args=None):

    net.train()
    optimizer.zero_grad()

    for data_iter_step, train_data in enumerate(data_loader):
        input_image, gt = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        pred_image = net(input_image, return_features=False)
        loss = F.smooth_l1_loss(pred_image, gt)

        # gradient accumulation
        loss /= update_freq
        loss.backward()
        if (data_iter_step + 1) % update_freq == 0 or (data_iter_step + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()

def main(args):
    update_freq = args.update_freq
    for epoch in range(epoch_start, num_epochs):
        train_one_epoch_accum(
            net=net, data_loader=lbl_train_data_loader, optimizer=optimizer,
            epoch=epoch, device=device, update_freq=update_freq, args=None
        )

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
