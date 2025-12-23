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
import elementtransformer
import torch
from matplotlib.colors import TwoSlopeNorm
import time
import os
import pandas as pd

def img_visualization(path, title, unit, data, min_value=99, max_value=-99):
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sc = ax.tripcolor(lon, lat, triangles, data, vmin=np.percentile(data, 2.5), vmax=np.percentile(data, 97.5),
                        shading='flat', cmap=cmocean.cm.thermal)
    margin_lon = delta_lon * 0.02
    margin_lat = delta_lat * 0.02
    ax.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=unit)

    plt.tight_layout()
    plt.savefig(path + '/' + title + '.jpg', dpi=300, bbox_inches='tight')
    print(path + '/' + title + '.jpg saved successfully')
    plt.close()

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

    sc1 = ax1.tripcolor(lon, lat, triangles, target, vmin=min_value, vmax=max_value,
                        shading='flat', cmap=cmocean.cm.speed)
    margin_lon = delta_lon * 0.02
    margin_lat = delta_lat * 0.02
    ax1.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax1.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax1.set_xlabel('Longitude (°E)', fontsize=12)
    ax1.set_ylabel('Latitude (°N)', fontsize=12)
    ax1.set_title(fvcomtitle, fontsize=14)
    fig.colorbar(sc1, ax=ax1, shrink=0.8, label=unit)

    sc2 = ax2.tripcolor(lon, lat, triangles, output, vmin=min_value, vmax=max_value,
                        shading='flat', cmap=cmocean.cm.speed)
    ax2.set_xlim(lon_min - margin_lon, lon_max + margin_lon)
    ax2.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax2.set_xlabel('Longitude (°E)', fontsize=12)
    ax2.set_ylabel('Latitude (°N)', fontsize=12)
    ax2.set_title(aititle, fontsize=14)
    fig.colorbar(sc2, ax=ax2, shrink=0.8, label=unit)

    norm_diff = TwoSlopeNorm(vmin=diff_min, vcenter=(diff_max+diff_min)/2, vmax=diff_max)

    sc3 = ax3.tripcolor(lon, lat, triangles, output-target,
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

def int_to_time_str_144(i: int) -> str:
    if not (0 <= i <= 143):
        raise ValueError("Input i must be between 0 and 142")
    total_minutes = i * 10 + 10
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"

def int_to_time_str_24(i: int) -> str:
    if not (0 <= i <= 23):
        raise ValueError("Input i must be between 0 and 23")
    hours = i + 1
    minutes = 0
    return f"{hours:02d}:{minutes:02d}"

from datetime import datetime, timedelta

def int_to_time_str_167(i: int) -> str:
    if not (0 <= i <= 166):
        raise ValueError("Input i must be between 0 and 166 (inclusive)")
    
    start = datetime(2025, 1, 1, 1, 0)
    
    target = start + timedelta(hours=i)
    
    return target.strftime("%Y.%m.%d %H:%M")

def predict_n_step(n, model, checkpoint_name, node_data, triangle_data,
                    node_json, triangle_json, output_node_npy, output_triangle_npy, device):
    checkpoint_name=checkpoint_name
    criterion = elementtransformer.WeightedMAEMSELoss().cuda()
    output_node_data = []
    output_triangle_data = []
    daily_length = n
    predict_length = 1
    for i in range(0, daily_length - predict_length):
        step = i
        input_node_data = torch.tensor(node_data[step,:,:]).to(device)
        target_node_data = torch.from_numpy(node_data[step+predict_length,:,:]).to(device)
        input_triangle_data = torch.tensor(triangle_data[step,:,:]).to(device)
        target_triangle_data = torch.from_numpy(triangle_data[step+predict_length,:,:]).to(device)
        print(input_node_data.shape)
        print(target_node_data.shape)
        print(input_triangle_data.shape)
        print(target_triangle_data.shape)
        output = model.predict(input_node_data, input_triangle_data, checkpoint_name=checkpoint_name)
        loss = criterion(output[0], target_node_data)+ criterion(output[1], target_triangle_data)
        print('step: ' + str(step) + ' loss: ', loss)
        
        output_node_data.append(output[0].squeeze(0).cpu().numpy())
        output_triangle_data.append(output[1].squeeze(0).cpu().numpy())
    
    output_node_data = processing.denormalization(np.array(output_node_data), 0, node_json)
    output_triangle_data = processing.denormalization(np.array(output_triangle_data), 1, triangle_json)
    np.save(output_node_npy, output_node_data)
    np.save(output_triangle_npy, output_triangle_data)
    return output_node_data, output_triangle_data

def img_output(node_target, triangle_target, node_pred, triangle_pred):

    node_name = ['Temperature0', 'Temperature1', 'Temperature2', 'Temperature3', 'Temperature4',
                 'Salinity0', 'Salinity1', 'Salinity2', 'Salinity3', 'Salinity4',
                 'Zeta', 'Short_Wave', 'Net_Heat_Flux']
    triangle_name = ['U0', 'U1', 'U2', 'U3', 'U4',
                     'V0', 'V1', 'V2', 'V3', 'V4',
                     'WW0', 'WW1', 'WW2', 'WW3', 'WW4',
                     'Tauc', 'UWind_Speed', 'VWind_Speed']
    node_unit = ['°', '°', '°', '°', '°',
                 '1e-3', '1e-3', '1e-3', '1e-3', '1e-3',
                 'm', 'W m-2', 'W m-2']
    triangle_unit = ['meters s-1', 'meters s-1', 'meters s-1', 'meters s-1', 'meters s-1',
                     'meters s-1', 'meters s-1', 'meters s-1', 'meters s-1', 'meters s-1',
                     'm^2 s^-2', 'm^2 s^-2', 'm^2 s^-2', 'm^2 s^-2', 'm^2 s^-2',
                     'meters s-1', 'meters s-1', 'meters s-1']

    # node_pred = processing.denormalization(node_pred, 0, node_json_file)
    # node_target_data = processing.denormalization(node_target, 0, node_json_file)
    # triangle_target_data = processing.denormalization(triangle_target, 1, triangle_json_file)
    # triangle_pred = processing.denormalization(triangle_pred, 1, triangle_json_file)
    print(triangle_pred.shape, node_pred.shape, triangle_target.shape, node_target.shape)
    for n in range(0, len(triangle_name)):
        output = triangle_pred[:, :, n],
        target = triangle_target[:, :, n]

        min_value = min(np.percentile(target, 2.5), np.percentile(output, 2.5))
        max_value = max(np.percentile(target, 97.5), np.percentile(output, 97.5))
        diff = output - target
        diff_min = np.percentile(diff, 2.5)
        diff_max = np.percentile(diff, 97.5)
        print(min_value, max_value, diff_min, (diff_max+diff_min)/2, diff_max)
        fvcomtitle = triangle_name[n]
        aititle = triangle_name[n]
        unit = triangle_unit[n]
        for i in range(0, 167):
            img_comparison('result/2025-7days/' + fvcomtitle, str(i).zfill(3),
                           'FVCOM ' + fvcomtitle + ' ' + int_to_time_str_167(i),
                           'AI ' + aititle + ' ' + int_to_time_str_167(i), unit,
                           triangle_pred[i, :, n],
                           triangle_target[i, :, n],
                           min_value=min_value, max_value=max_value, diff_min=diff_min, diff_max=diff_max)

def seven_days_visualization():
    node_file_list = sorted(os.listdir('dataset/'))
    triangle_file_list = sorted(os.listdir('dataset/'))
    node_target = []
    triangle_target = []
    for i in node_file_list:
        if i.endswith('node.npy'):
            print(i)
            data = np.load('dataset/' + i)
            print(data.shape)
            node_target.append(data)
        elif i.endswith('triangle.npy'):
            print(i)
            data = np.load('dataset/' + i)
            print(data.shape)
            triangle_target.append(data)
        else:
            print(i, 'skip')
    node_target = np.array(node_target)
    node_target = node_target.reshape(7 * 24, 60882, 13)

    triangle_target = np.array(triangle_target)
    triangle_target = triangle_target.reshape(7 * 24, 115443, 18)
    node_pred = np.load('7-node_pred_list.npy')
    triangle_pred = np.load('7-tri_pred_list.npy')

    for i in range(1, 7):
        print(24*(i-1), 24*i)
        triangle_pred[24*(i-1):24*i] = processing.denormalization(triangle_pred[24*(i-1):24*i], 1, 'dataset/triangle/json/GGB_25010'+str(i)+'T00.json')
    triangle_pred[167-23:167] = processing.denormalization(triangle_pred[167-23:167], 1, 'dataset/triangle/json/GGB_250107T00.json')

    print(node_pred.shape)
    print(node_target[1:,:,:].shape)
    print(triangle_pred.shape)
    print(triangle_target[1:,:,:].shape)
    img_output(node_target[1:,:,:], triangle_target[1:,:,:], node_pred, triangle_pred)

def point_difference(type, variable, position, ai, fvcom):
    date_range = pd.date_range(start='2025-01-01 01:00', end='2025-01-07 23:00', freq='h')
    df = pd.DataFrame({'Time': date_range, 'Y1': ai, 'Y2': fvcom})

    plt.figure(figsize=(14, 7))
    plt.plot(df['Time'], df['Y1'], label='AI', marker='.')
    plt.plot(df['Time'], df['Y2'], label='FVCOM', marker='x')

    plt.gcf().autofmt_xdate()

    plt.title('Comparison of Two Data Sets Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # device='cuda' if torch.cuda.is_available() else 'cpu'
    # model = elementtransformer.FVCOMModel(
    #     node=60882, triangle=115443, node_var=13,
    #     triangle_var=18, embed_dim=256,
    #     mlp_ratio=4., nhead=2, num_layers=2,
    #     neighbor_table=None, dropout=0.1
    # ).to(device)
    # criterion = elementtransformer.WeightedMAEMSELoss().to(device)
    # node_data_dir="dataset/node/data/"
    # tri_data_dir="dataset/triangle/data/"
    # checkpoint_name='checkpoints/element-transformer-v1.0.pth'

    # node_input = np.load('dataset/node/data/GGB_250101T00.npy')
    # triangle_input = np.load('dataset/triangle/data/GGB_250101T00.npy')

    # print(node_input.shape)
    # print(triangle_input.shape)
    # node_json_file='dataset/node/json/GGB_250101T00.json'
    # triangle_json_file='dataset/triangle/json/GGB_250101T00.json'

    # output_node_npy= 'output/250101node.npy'
    # output_triangle_npy= 'output/250101triangle.npy'
    # node_pred, triangle_pred = predict_n_step(24, model, checkpoint_name, node_input, triangle_input,
    #                                         node_json_file, triangle_json_file,
    #                                         output_node_npy, output_triangle_npy, device)
    node_file_list = sorted(os.listdir('dataset/'))
    triangle_file_list = sorted(os.listdir('dataset/'))
    node_target = []
    triangle_target = []
    for i in node_file_list:
        if i.endswith('node.npy'):
            print(i)
            data = np.load('dataset/' + i)
            print(data.shape)
            node_target.append(data)
        elif i.endswith('triangle.npy'):
            print(i)
            data = np.load('dataset/' + i)
            print(data.shape)
            triangle_target.append(data)
        else:
            print(i, 'skip')
    node_target = np.array(node_target)
    node_target = node_target.reshape(7 * 24, 60882, 13)

    triangle_target = np.array(triangle_target)
    triangle_target = triangle_target.reshape(7 * 24, 115443, 18)
    node_pred = np.load('7-node_pred_list.npy')
    triangle_pred = np.load('7-tri_pred_list.npy')

    for i in range(1, 7):
        print(24*(i-1), 24*i)
        triangle_pred[24*(i-1):24*i] = processing.denormalization(triangle_pred[24*(i-1):24*i], 1, 'dataset/triangle/json/GGB_25010'+str(i)+'T00.json')
        node_pred[24*(i-1):24*i] = processing.denormalization(node_pred[24*(i-1):24*i], 0, 'dataset/node/json/GGB_25010'+str(i)+'T00.json')
    triangle_pred[167-23:167] = processing.denormalization(triangle_pred[167-23:167], 1, 'dataset/triangle/json/GGB_250107T00.json')
    node_pred[167-23:167] = processing.denormalization(node_pred[167-23:167], 0, 'dataset/node/json/GGB_250107T00.json')
    np.save('dash-triangle-pred-visualization-7days.npy', triangle_pred)
    np.save('dash-node-pred-visualization-7days.npy', node_pred)
    np.save('dash-triangle-target-visualization-7days.npy', triangle_target[1:,:,:])
    np.save('dash-node-target-visualization-7days.npy', node_target[1:,:,:])
    print(node_pred.shape)
    print(node_target[1:,:,:].shape)
    print(triangle_pred.shape)
    print(triangle_target[1:,:,:].shape)
    # img_output(node_target[1:,:,:], triangle_target[1:,:,:], node_pred, triangle_pred)
    # point_difference(node_pred[:,20000,0], node_target[1:,20000,0])



    
