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

def denormalization(normalized_data, pos, var, json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    mins = np.array(params['mins'])
    maxs = np.array(params['maxs'])
    ranges = maxs - mins
    if pos == 0:
        if normalized_data.shape[-1] != 13:
            raise ValueError(f"Expected last dimension=13, got {normalized_data.shape[-1]}")
    elif pos == 1:
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
        output_name = i.split('.')[0]
        normalization(transposed_node, 'dataset/node/data/' + output_name + '.npy',
                    'dataset/node/json/' + output_name + '.json')
        normalization(transposed_triangle, 'dataset/triangle/data/' + output_name + '.npy',
                    'dataset/triangle/json/' + output_name + '.json')
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

if __name__ == "__main__":
    nc2npy('dataset')


    # npy_normalization('D:/FVCOM/1hour/data', 'D:/FVCOM/1hour/normalized_data', 'D:/FVCOM/1hour/json')