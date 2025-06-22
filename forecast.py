import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import argparse
from models.FEDformer import Model
from data_provider.data_loader import Dataset_Pred
from torch.utils.data import DataLoader

# Hàm tải và xử lý dữ liệu
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    values = df['Vehicles'].values.astype(float)

    # Chuẩn hóa dữ liệu
    mean = values.mean()
    std = values.std() + 1e-8  # Tránh chia cho 0
    values_normalized = (values - mean) / std

    return df, values_normalized, mean, std

# Hàm chuẩn bị dữ liệu cho FEDformer
def prepare_data(values_normalized, seq_len=96, label_len=48, pred_len=96):
    # Tạo DataFrame tạm để lưu dữ liệu
    df_temp = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01 00:00:00', periods=len(values_normalized), freq='T'),
        'Vehicles': values_normalized
    })
    df_temp.to_csv('temp_data.csv', index=False)

    # Khởi tạo Dataset_Pred với các tham số trực tiếp
    dataset = Dataset_Pred(
        root_path='.',  # Thư mục hiện tại
        data_path='temp_data.csv',
        target='Vehicles',  # Cột mục tiêu
        freq='t',  # Tần suất phút
        size=[seq_len, label_len, pred_len],
        scale=True,  # Chuẩn hóa dữ liệu
        timeenc=0,  # Đặc trưng thời gian cơ bản
        features='S',  # Chuỗi đơn biến
        inverse=False  # Không cần inverse trong dataset
    )

    # Tạo DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader

# Hàm dự báo
def forecast(model, data_loader, mean, std, pred_len):
    model.eval()
    with torch.no_grad():
        # Lấy batch cuối cùng từ data_loader
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            # Dự báo
            outputs = model(batch_x, batch_x_mark, None, None)
            pred = outputs[:, -pred_len:, :].squeeze().cpu().numpy()  # Lấy pred_len cuối
            pred = pred * std + mean  # Khôi phục giá trị gốc
            pred = np.clip(pred, 0, 4)  # Giới hạn trong [0, 4]
            return pred
    return None

# Hàm chính
def main():
    # Tham số
    file_path = "/home/tupham/Documents/Development/FEDformer/FEDformer/dataset/traffic_time_only.csv"
    seq_len = 96
    label_len = 48
    pred_len = 96
    weight_path = "/home/tupham/Documents/Development/FEDformer/FEDformer/checkpoints/21jun/checkpoint.pth"
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tải dữ liệu
    df, values_normalized, mean, std = load_data(file_path)

    # Kiểm tra số lượng mẫu
    if len(df) < seq_len:
        raise ValueError(f"Dữ liệu chỉ có {len(df)} mẫu, cần ít nhất {seq_len} mẫu.")

    # Chuẩn bị dataset
    data_loader = prepare_data(values_normalized, seq_len, label_len, pred_len)

    # Tạo đối tượng args cho mô hình
    args = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=1,
        dec_in=1,
        c_out=1,  # Số kênh đầu ra
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        embed='timeF',
        freq='t',
        activation='gelu',
        output_attention=False,
        moving_avg=24,
        mode_select = 'random',
        version='Fourier',
        modes=64,
        L=3,
        base='legendre',
        cross_activation='tanh'
    )

    model = Model(args).to(device)
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Lỗi khi tải trọng số: {e}")
        return

    # Dự báo
    predictions = forecast(model, data_loader, mean, std, pred_len)
    if predictions is None:
        print("Lỗi khi dự báo: Không lấy được batch dữ liệu.")
        return

    # Chuẩn bị dữ liệu cho biểu đồ
    historical_dates = df['date'].values
    historical_values = df['Vehicles'].values
    last_date = historical_dates[-1]
    forecast_dates = [last_date + timedelta(minutes=i) for i in range(1, pred_len + 1)]

    plt.figure(figsize=(12, 6))
    plt.plot(historical_dates, historical_values, label='Dữ liệu lịch sử', color='#1e90ff', linewidth=2)
    plt.plot(forecast_dates, predictions, label='Dữ liệu dự báo', color='#ff4500', linewidth=2, linestyle='--')
    plt.xlabel('Thời gian')
    plt.ylabel('Số lượng xe')
    plt.title('Dự báo lưu lượng giao thông (FEDformer)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('traffic_forecast.png')
    plt.show()

    print(f"Số điểm dữ liệu lịch sử: {len(historical_values)}")
    print(f"Số điểm dữ liệu dự báo: {len(predictions)}")
    print(f"Trung bình số xe lịch sử: {historical_values.mean():.2f}")
    print(f"Trung bình số xe dự báo: {predictions.mean():.2f}")

if __name__ == "__main__":
    main()