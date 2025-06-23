import sys

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from exp.exp_main import Exp_Main
from utils.tools import StandardScaler
from data_provider.data_factory import data_provider
import os

def load_model(args):
    exp = Exp_Main(args)
    exp.model.eval()

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = f"./checkpoints/{args.model}_{args.data}_{args.pred_len}_0.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    print(f"[✔] Loaded checkpoint from: {checkpoint_path}")
    return exp


def forecast(exp, args):
    data_set, data_loader = data_provider(args, flag='pred')

    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        device = args.device
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)  # thêm dòng này nếu bạn dùng batch_y tạo dec_inp
        batch_x_mark = batch_x_mark.to(device=device, dtype=torch.float32)
        batch_y_mark = batch_y_mark.to(device=device, dtype=torch.float32)

        break

    with torch.no_grad():
        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]],
                              dtype=torch.float32, device=device)

        outputs = exp.model(batch_x, batch_x_mark, batch_y_mark, dec_inp)

    pred = outputs.detach().cpu().numpy().squeeze()
    input_seq = batch_x.detach().cpu().numpy().squeeze()
    return input_seq, pred


def save_csv(input_seq, pred, args):
    forecast_start = len(input_seq)
    full = np.concatenate([input_seq, pred])
    timestamps = list(range(len(full)))

    df = pd.DataFrame({
        'step': timestamps,
        'value': full,
        'type': ['input'] * len(input_seq) + ['forecast'] * len(pred)
    })

    out_path = os.path.join('./outputs', 'forecast_output.csv')
    os.makedirs('./outputs', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[✔] Forecast saved to: {out_path}")

def plot_forecast(input_seq, pred, args):
    full = np.concatenate([input_seq, pred])
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(input_seq)), input_seq, label='Input Sequence')
    plt.plot(range(len(input_seq), len(full)), pred, label='Forecast', linestyle='--')
    plt.axvline(x=len(input_seq) - 1, color='r', linestyle=':', label='Forecast Start')
    plt.legend()
    plt.title(f"Forecasting {args.pred_len} steps using FEDformer")
    plt.grid(True)

    os.makedirs('./outputs', exist_ok=True)
    plot_path = os.path.join('./outputs', 'forecast_plot.png')
    plt.savefig(plot_path)
    print(f"[✔] Forecast plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--is_training', type=int, default=0)
    parser.add_argument('--model', type=str, default='FEDformer')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='traffic_time_only.csv')
    parser.add_argument('--target', type=str, default='Vehicles')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--freq', type=str, default='t')
    parser.add_argument('--detail_freq', type=str, default='t')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--use_multi_gpu', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False ,help='whether to output attention in ecoder')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')


    args = parser.parse_args()
    print("[DEBUG] args type:", type(args))
    if isinstance(args, argparse.Namespace):
        print("[DEBUG] args.data =", args.data)
    else:
        print("[ERROR] args đã bị gán nhầm, hiện là:", args)
        sys.exit(1)

    exp = load_model(args)

    input_seq, pred = forecast(exp, args)
    save_csv(input_seq, pred, args)
    plot_forecast(input_seq, pred, args)
