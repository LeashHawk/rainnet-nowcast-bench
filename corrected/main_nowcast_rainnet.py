"""[NEW-2026-10-18] RainNet hourly nowcasting entry for the properly wired
baselines in model_nowcast.py (ConvLSTM encoder-forecaster, U-Net), with plain
MSE or the Shi et al. (2017) balanced loss. Checkpoint selection and the final
TEST line use plain MSE for every model (train_function), so all rows of the
benchmark table are selected under the same criterion.
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader.dataloader_rainnet_fixed import load_data as load_rainnet
from model_nowcast import BalancedLoss, TAULoss, build
from tools.check import check_device
from tools.train_function import train_model, test_model


def main(args):
    train_loader, eval_loader, test_loader, _, _ = load_rainnet(
        root=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build(args.arch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'MODEL:::arch={args.arch} loss={args.loss} params={n_params}', flush=True)
    if args.loss == 'balanced':
        criterion = BalancedLoss()
    elif args.loss == 'tau_ddr':
        criterion = TAULoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = check_device()
    model.to(device)

    train_model(model=model, train_loader=train_loader, eval_loader=eval_loader,
                criterion=criterion, optimizer=optimizer, device=device, args=args)

    model.load_state_dict(torch.load(args.checkpoints, map_location=device))
    test_mse_loss, test_mae_loss = test_model(model, test_loader, device)
    print(f'TEST:::MSE Loss: {test_mse_loss:.7f}, MAE Loss: {test_mae_loss:.7f}')


if __name__ == "__main__":
    p = argparse.ArgumentParser('RainNet nowcast baselines', add_help=False)
    p.add_argument('--arch', required=True, choices=['convlstm_ed', 'unet', 'simvp', 'tau'])
    p.add_argument('--loss', default='mse', choices=['mse', 'balanced', 'tau_ddr'])
    p.add_argument('--if_pretrained', type=bool, default=False)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_schedule', type=str, default='cosine', choices=['const', 'cosine'])
    p.add_argument('--num_epochs', type=int, default=30)
    p.add_argument('--early_stop_patience', type=int, default=8)
    p.add_argument('--resume', type=str, default='no', choices=['no', 'auto'])
    p.add_argument('--log_per_samples', type=int, default=100)
    p.add_argument('--log_path', type=str, required=True)
    p.add_argument('--checkpoints', type=str, required=True)
    main(p.parse_args())
