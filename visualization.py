import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch.nn as nn
from scipy.spatial import Delaunay
import json
import cmocean
import processing
import nodetransformer
import torch
from matplotlib.colors import TwoSlopeNorm
import time
import os

def img_comparison(path, title, fvcomtitle, aititle, unit, output, target, min_value=99, max_value=-99, diff_min=-50, diff_max=50):
    with open('img_config.json', 'r') as f:
        params = json.load(f)
        lon = np.array(params['lon'])
        lat = np.array(params['lat'])
        lon_min = params['lon_min']
        lon_max = params['lon_max']
        lat_min = params['lat_min']
        lat_max = params['lat_max']
        delta_lon = params['delta_lon']
        delta_lat = params['delta_lat']
        triangles = np.array(params['triangles'])

    
    if not os.path.exists(path):
        os.makedirs(path)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # norm_main = PowerNorm(gamma=0.5, vmin=min_value, vmax=max_value)

    sc1 = ax1.tripcolor(lon, lat, triangles, target, vmin=min_value, vmax=max_value,
                        shading='flat', cmap=cmocean.cm.thermal)
    margin_lon = delta_lon * 0.02
    margin_lat = delta_lat * 0.02
    ax1.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax1.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax1.set_xlabel('Longitude (°E)', fontsize=12)
    ax1.set_ylabel('Latitude (°N)', fontsize=12)
    ax1.set_title(fvcomtitle, fontsize=14)
    fig.colorbar(sc1, ax=ax1, shrink=0.8, label=unit)

    sc2 = ax2.tripcolor(lon, lat, triangles, output, vmin=min_value, vmax=max_value,
                        shading='flat', cmap=cmocean.cm.thermal)
    ax2.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax2.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax2.set_xlabel('Longitude (°E)', fontsize=12)
    ax2.set_ylabel('Latitude (°N)', fontsize=12)
    ax2.set_title(aititle, fontsize=14)
    fig.colorbar(sc2, ax=ax2, shrink=0.8, label=unit)

    diff = output - target
    # diff_abs_max = np.abs(diff)
    
    norm_diff = TwoSlopeNorm(vmin=diff_min, vcenter=0, vmax=diff_max)
    sc3 = ax3.tripcolor(lon, lat, triangles, diff,
                        shading='flat', cmap=cmocean.cm.balance, norm=norm_diff)
    ax3.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax3.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax3.set_xlabel('Longitude (°E)', fontsize=12)
    ax3.set_ylabel('Latitude (°N)', fontsize=12)
    ax3.set_title('Difference (AI - FVCOM)', fontsize=14)
    fig.colorbar(sc3, ax=ax3, shrink=0.8, label=unit)

    plt.tight_layout()
    plt.savefig(path + '/' + title + '.jpg', dpi=300, bbox_inches='tight')
    print(path + '/' + title + '.jpg saved successfully')
    plt.close()

def int_to_time_str(i: int) -> str:
    if not (0 <= i <= 143):
        raise ValueError("Input i must be between 0 and 142")
    total_minutes = i * 10 + 10
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"

def predict_one_day(model, checkpoint_name, node_data, triangle_data,
                    node_json, triangle_json, output_node_npy, output_triangle_npy, device):
    checkpoint_name=checkpoint_name
    criterion = nodetransformer.WeightedMAEMSELoss().cuda()
    # criterion = nn.HuberLoss(delta=1.0).cuda()
    output_node_data = []
    output_triangle_data = []
    daily_length = 144
    predict_length = 1
    for i in range(0, daily_length - predict_length):
        step = i
        input_node_data = torch.tensor(node_data[step,:,:]).to(device)
        target_node_data = torch.from_numpy(node_data[step+predict_length,:,:]).unsqueeze(0).unsqueeze(0).to(device)
        input_triangle_data = torch.tensor(triangle_data[step,:,:]).to(device)
        target_triangle_data = torch.from_numpy(triangle_data[step+predict_length,:,:]).unsqueeze(0).unsqueeze(0).to(device)
        output = model.predict(input_node_data, input_triangle_data, checkpoint_name=checkpoint_name)
        print('step: ' + str(step) + ' loss: ', criterion(output[0], target_node_data)+ criterion(output[1], target_triangle_data))
        
        output_node_data.append(output[0].cpu().numpy())
        output_triangle_data.append(output[1].cpu().numpy())
    
    output_node_data = processing.denormalization(np.array(output_node_data), 1, 13, node_json)
    output_triangle_data = processing.denormalization(np.array(output_triangle_data), 1, 18, triangle_json)
    # print("output shape:", output.shape)
    np.save(output_node_npy, output_node_data)
    np.save(output_triangle_npy, output_triangle_data)
    return output_node_data, output_triangle_data

def predict_mul_day(n, model, checkpoint_name, node_data, triangle_data,
                    node_json, triangle_json, output_node_npy, output_triangle_npy, device):
    checkpoint_name=checkpoint_name
    criterion = nodetransformer.WeightedMAEMSELoss().cuda()
    # criterion = nn.HuberLoss(delta=1.0).cuda()
    output_node_data = []
    output_triangle_data = []
    daily_length = n
    predict_length = 1

    input_node_data = torch.tensor(node_data[0,:,:]).to(device)
    target_node_data = torch.from_numpy(node_data[0+predict_length,:,:]).unsqueeze(0).unsqueeze(0).to(device)
    input_triangle_data = torch.tensor(triangle_data[0,:,:]).to(device)
    target_triangle_data = torch.from_numpy(triangle_data[0+predict_length,:,:]).unsqueeze(0).unsqueeze(0).to(device)
    output = model.predict(input_node_data, input_triangle_data, checkpoint_name=checkpoint_name)
    output_node_data.append(output[0].cpu().numpy())
    output_triangle_data.append(output[1].cpu().numpy())
    print('step: 0 loss: ', criterion(output[0], target_node_data)+ criterion(output[1], target_triangle_data))

    for i in range(1, daily_length - predict_length):
        step = i
        input_node_data = output[0].to(device)
        target_node_data = torch.from_numpy(node_data[step+predict_length,:,:]).unsqueeze(0).unsqueeze(0).to(device)
        input_triangle_data = output[1].to(device)
        target_triangle_data = torch.from_numpy(triangle_data[step+predict_length,:,:]).unsqueeze(0).unsqueeze(0).to(device)
        output = model.predict(input_node_data, input_triangle_data, checkpoint_name=checkpoint_name)
        print('step: ' + str(step) + ' loss: ', criterion(output[0], target_node_data)+ criterion(output[1], target_triangle_data))
        output_node_data.append(output[0].cpu().numpy())
        output_triangle_data.append(output[1].cpu().numpy())
    
    output_node_data = processing.denormalization(np.array(output_node_data), 1, 13, node_json)
    output_triangle_data = processing.denormalization(np.array(output_triangle_data), 1, 18, triangle_json)
    # print("output shape:", output.shape)
    np.save(output_node_npy, output_node_data)
    np.save(output_triangle_npy, output_triangle_data)
    return output_node_data, output_triangle_data

if __name__ == "__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_name='checkpoints/2025_12_05_11_32_Inha_GPU_1xA100.pth'
    node_json_file='dataset/node/json/GGB_240606T00.json'
    triangle_json_file='dataset/triangle/json/GGB_240606T00.json'
    output_node_npy= 'crossattentionnode240606.npy'
    output_triangle_npy= 'crossattentiontriangle240606.npy'
    batch_size=1
    node_data_dir="dataset/node/data/",
    tri_data_dir="dataset/triangle/data/",
    num_epochs=100,
    total_timesteps=144,
    pred_step=1,
    batch_size=1
    steps_per_file=144,
    early_stop_patience=25
    start_time = time.time()
    best_loss = float('inf')
    early_stop_cnt = 0
    best_epoch = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nodetransformer.FVCOMModel(
        node=60882, triangle=115443, node_var=13,
        triangle_var=18, embed_dim=256,
        mlp_ratio=4., nhead=2, num_layers=2,
        neighbor_table=None, dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nodetransformer.WeightedMAEMSELoss().to(device)
    node_input = np.load('dataset/node/data/GGB_240606T00.npy')
    triangle_input = np.load('dataset/triangle/data/GGB_240606T00.npy')

    # node_pred, triangle_pred = predict_mul_day(144, model, checkpoint_name, node_input, triangle_input,
    #                                           node_json_file, triangle_json_file,
    #                                           output_node_npy, output_triangle_npy, device)

    node_pred = np.load(output_node_npy)
    triangle_pred = np.load(output_triangle_npy)
    node_name = ['Temperature', 'Salinity', 'Zeta', 'Short_Wave', 'Net_Heat_Flux']
    triangle_name = ['U', 'V', 'WW', 'Tauc', 'UWind_Speed', 'VWind_Speed']

    # ['degrees_C', '1e-3', 'meters', 'W m-2', 'W m-2']
    node_unit = ['°', '1e-3', 'm', 'W m-2', 'W m-2']
    triangle_unit = ['meters s-1', 'meters s-1', 'm^2 s^-2', 'meters s-1', 'meters s-1', 'meters s-1']
    node_target_data = processing.denormalization(node_input[1:,:,:], 0, 13, node_json_file)
    triangle_target_data = processing.denormalization(triangle_input[1:,:,:], 1, 18, triangle_json_file)
    print(triangle_target_data.shape, node_pred.shape, node_pred.shape, triangle_target_data.shape)
    level_index = 10
    min_value = min(np.percentile(node_target_data[:,:,level_index], 2.5), np.percentile(node_pred[:,:,level_index], 2.5))
    max_value = max(np.percentile(node_target_data[:,:,level_index], 97.5), np.percentile(node_pred[:,:,level_index], 97.5))
    # diff_min = (np.percentile(triangle_pred[:,:,level_index], 97.5) - np.percentile(triangle_target_data[:,:,level_index], 2.5)).min()
    # diff_max = (np.percentile(triangle_pred[:,:,level_index], 97.5) - np.percentile(triangle_target_data[:,:,level_index], 2.5)).max()
    fvcomtitle = node_name[level_index//5]
    aititle = node_name[level_index//5]
    unit = node_unit[level_index//5]
    print(min_value, max_value)
    date = ' 2024.06.06 '
    for i in range(0, 143):
        time_index = i
        img_comparison('result/' + fvcomtitle, str(i).zfill(3), 'FVCOM ' + fvcomtitle + date + int_to_time_str(i),
                       'AI ' + aititle + date + int_to_time_str(i), unit,
                       node_pred[time_index, :, level_index],
                       node_target_data[time_index, :, level_index],
                       min_value=min_value, max_value=max_value, diff_min=min_value, diff_max=max_value)
