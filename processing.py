import xarray as xr
import numpy as np
import os
import json
import torch

def normalization(data, output_name, json_path):
    # data (144, 115443, 18)
    print("Original data shape:", data.shape)
    mins = data.min(axis=(0, 1))
    maxs = data.max(axis=(0, 1))
    eps = 1e-8
    ranges = maxs - mins + eps
    normalized_data = 2 * (data - mins) / ranges - 1
    print("Normalized min per channel:", normalized_data.min(axis=(0, 1)))
    print("Normalized max per channel:", normalized_data.max(axis=(0, 1)))
    np.save(output_name, normalized_data)
    print(f"Normalized data saved to {output_name}")
    norm_params = {
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "ranges": ranges.tolist(),
        "shape_before_normalization": list(data.shape),
        "normalized_range": [-1.0, 1.0]
    }
    with open(json_path, 'w') as f:
        json.dump(norm_params, f, indent=4)
    print(f"Normalization parameters saved to {json_path}")

def denormalization(normalized_data, pos, json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    mins = np.array(params['mins'])
    maxs = np.array(params['maxs'])
    ranges = maxs - mins
    var = 0
    if pos == 0:
        var = 13
        if normalized_data.shape[-1] != 13:
            raise ValueError(f"Expected last dimension=13, got {normalized_data.shape[-1]}")
    elif pos == 1:
        var = 18
        if normalized_data.shape[-1] != 18:
            raise ValueError(f"Expected last dimension=18, got {normalized_data.shape[-1]}")
    else:
        raise ValueError(f"pos expected only 0 or 1")
    reshape_shape = [1] * (normalized_data.ndim - 1) + [var]
    mins = mins.reshape(reshape_shape)
    ranges = ranges.reshape(reshape_shape)
    original_data = (normalized_data + 1.0) * ranges / 2.0 + mins
    return original_data

def nc2npy(nc_path_in):
    file_list = sorted(os.listdir(nc_path_in))
    for i in file_list:
        if not i.endswith('.nc'):
            continue
        ds = xr.open_dataset(nc_path_in + '/' + i, decode_times=False)
        u = ds['u'].values[:, 0:40:8, :]
        v = ds['v'].values[:, 0:40:8, :]
        ww = ds['ww'].values[:, 0:40:8, :]
        tauc = ds['tauc'].values
        tauc = np.expand_dims(tauc, 1)
        uwind_speed = ds['uwind_speed'].values
        uwind_speed = np.expand_dims(uwind_speed, 1)
        vwind_speed = ds['vwind_speed'].values
        vwind_speed = np.expand_dims(vwind_speed, 1)

        temperature = ds['temp'].values[:, 0:40:8, :]
        salinity = ds['salinity'].values[:, 0:40:8, :]

        zeta = ds['zeta'].values
        zeta = np.expand_dims(zeta, 1)

        short_wave = ds['short_wave'].values
        short_wave = np.expand_dims(short_wave, 1)

        net_heat_flux = ds['net_heat_flux'].values
        net_heat_flux = np.expand_dims(net_heat_flux, 1)

        print(temperature.shape, salinity.shape, zeta.shape, short_wave.shape, net_heat_flux.shape)
        print(u.shape, v.shape, ww.shape, tauc.shape, uwind_speed.shape, vwind_speed.shape)

        stacked_node = np.concatenate([temperature, salinity, zeta, short_wave, net_heat_flux], axis=1)
        stacked_triangle = np.concatenate([u, v, ww, tauc, uwind_speed, vwind_speed], axis=1)
        print(stacked_node.shape)
        print(stacked_triangle.shape)
        transposed_node = np.transpose(stacked_node, (0, 2, 1))
        transposed_triangle = np.transpose(stacked_triangle, (0, 2, 1))
        print(transposed_node.shape)
        print(transposed_triangle.shape)
        # output_name = i.split('.')[0]
        # normalization(transposed_node, 'dataset/node/data/' + output_name + '.npy',
        #             'dataset/node/json/' + output_name + '.json')
        # normalization(transposed_triangle, 'dataset/triangle/data/' + output_name + '.npy',
        #             'dataset/triangle/json/' + output_name + '.json')
        # np.save('240506node.npy', transposed_node)
        # np.save('240506triangle.npy', transposed_triangle)
        ds.close()

def npy_normalization(data_in, data_out, json_out):
    npy_file = os.listdir(data_in)
    for i in npy_file:
        name = i.split('.')[0]
        dataset = np.load(data_in+'/'+i)
        normalization(dataset, data_out + '/' + i, 
                      json_out + '/' + name + '.json')

def adj_list_to_edge_index(adj_list):
    rows, cols = [], []
    for i, nbrs in enumerate(adj_list):
        rows.extend([i] * len(nbrs))
        cols.extend(nbrs)
    return torch.tensor([rows, cols], dtype=torch.long)

def generate_sparse_graph():
    with open("fvcom_topology_clean.json", "r") as f:
        data = json.load(f)
    # nn_neighbor = data["nn_neighbor"]
    # tt_neighbor = data["tt_neighbor"]
    # tn_neighbor = data["tn_neighbor"]
    # nt_neighbor = data["nt_neighbor"]
    
    node_edge_index = adj_list_to_edge_index(data["nn_neighbor"])
    tri_edge_index  = adj_list_to_edge_index(data["tt_neighbor"])
    tn_edge_index   = adj_list_to_edge_index(data["tn_neighbor"])
    nt_edge_index   = adj_list_to_edge_index(data["nt_neighbor"])

    return node_edge_index, tri_edge_index, tn_edge_index, nt_edge_index

import json
import numpy as np
from pathlib import Path

def merge_min_max_from_jsons(file_paths):
    """
    从多个 JSON 文件中合并 mins/maxs：
      - mins: 逐元素取所有文件中的最小值
      - maxs: 逐元素取所有文件中的最大值
      - ranges: 重新计算为 maxs - mins
    
    参数:
        file_paths (list): 7 个 JSON 文件路径列表
    
    返回:
        dict: 合并后的统计字典
    """
    if not file_paths:
        raise ValueError("文件路径列表不能为空")
    
    data_list = []
    for fp in file_paths:
        with open('dataset/7-json/triangle'+'/'+fp, 'r') as f:
            data = json.load(f)
            data_list.append(data)
    
    # 检查所有文件的 mins/maxs 长度是否一致
    first_len = len(data_list[0]['mins'])
    for i, d in enumerate(data_list):
        if len(d['mins']) != first_len or len(d['maxs']) != first_len:
            raise ValueError(f"文件 {file_paths[i]} 的 mins 或 maxs 长度不匹配")
    
    # 转为 numpy 数组
    all_mins = np.array([d['mins'] for d in data_list])   # shape: (7, D)
    all_maxs = np.array([d['maxs'] for d in data_list])   # shape: (7, D)
    
    # 逐元素取 min 和 max
    merged_mins = np.min(all_mins, axis=0)   # shape: (D,)
    merged_maxs = np.max(all_maxs, axis=0)   # shape: (D,)
    merged_ranges = merged_maxs - merged_mins
    
    # 元数据一致性检查
    meta_keys = ['shape_before_normalization', 'normalized_range']
    merged = {
        'mins': merged_mins.tolist(),
        'maxs': merged_maxs.tolist(),
        'ranges': merged_ranges.tolist(),
    }
    
    for key in meta_keys:
        first_val = data_list[0][key]
        for d in data_list[1:]:
            if d[key] != first_val:
                raise ValueError(f"元数据字段 '{key}' 在不同文件中不一致！")
        merged[key] = first_val
    
    return merged


if __name__ == "__main__":
    # nc2npy('dataset/nc')
    files = os.listdir('dataset/7-json/triangle')
    print(files)
    try:
        result = merge_min_max_from_jsons(files)
        # 可选：保存结果到新文件
        with open("dataset/7-json/triangle/averaged_stats_triangle.json", "w") as f:
            json.dump(result, f, indent=4)
        print("saved averaged_stats.json")
    except Exception as e:
        print("error: ", e)
    # npy_normalization('D:/FVCOM/1hour/data', 'D:/FVCOM/1hour/normalized_data', 'D:/FVCOM/1hour/json')