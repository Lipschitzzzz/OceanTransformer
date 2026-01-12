import os
import torch
import elementtransformer
import processing
import visualization
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict_n_steps(
    checkpoint_name,
    node_data_dir,
    triangle_data_dir,
    total_timesteps=144,
    steps_per_file=144,
    pred_step=1
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = elementtransformer.FVCOMDataset(
        node_data_dir=node_data_dir,
        triangle_data_dir=triangle_data_dir,
        total_timesteps=total_timesteps,
        steps_per_file=steps_per_file,
        pred_step=pred_step
    )

    total_samples = len(full_dataset)

    test_indices = list(range(total_samples))

    test_dataset = Subset(full_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = elementtransformer.FVCOMModel(
        node=60882, triangle=115443, node_var=13,
        triangle_var=18, embed_dim=256,
        mlp_ratio=4., nhead=2, num_layers=2,
        neighbor_table=None, dropout=0.1
    ).to(device)
    # criterion = elementtransformer.WeightedMAEMSELoss().to(device)
    huber = torch.nn.HuberLoss(delta=0.1, reduction='mean').to(device)
    mae = torch.nn.L1Loss(reduction='mean').to(device)

    W_triangle = 1.0
    W_node = 0.3

    model.eval()
    test_loss_sum = 0
    test_count = 0
    loss_list = []
    node_pred_list =[]
    triangle_pred_list = []
    iter = 0
    with torch.no_grad():
        for (node_x, triangle_x), (node_y, triangle_y) in test_loader:
            node_x, triangle_x = node_x.to(device), triangle_x.to(device)
            node_y, triangle_y = node_y.to(device), triangle_y.to(device)
            node_pred, triangle_pred = model.predict(node_x, triangle_x, checkpoint_name)
            loss_node = mae(node_pred, node_y)
            loss_triangle = huber(triangle_pred, triangle_y)
            loss = W_node * loss_node + W_triangle * loss_triangle

            node_pred = node_pred.cpu().squeeze(0).numpy()
            triangle_pred = triangle_pred.cpu().squeeze(0).numpy()
            # node_pred_list.append(processing.denormalization(node_pred, 0, 'dataset/node/json/GGB_240816T00.json'))
            node_pred_list.append(node_pred)
            # triangle_pred_list.append(processing.denormalization(triangle_pred, 1, 'dataset/triangle_pred/json/GGB_240816T00.json'))
            triangle_pred_list.append(triangle_pred)
            test_loss_sum += loss.detach()
            loss_list.append(loss.cpu())
            test_count += 1
            iter += 1
            print('iter:', iter, 'loss', loss, 'input', node_x.shape, triangle_x.shape)
            print('iter:', iter, 'loss', loss, 'output', node_pred.shape, triangle_pred.shape)
        val_loss = (test_loss_sum / test_count).item()
    np.save('node_pred.npy', node_pred_list)
    np.save('triangle_pred.npy', triangle_pred_list)
    np.save('loss_pred.npy', loss_list)
    print('average loss', val_loss)
    return val_loss

def evaluation(
    checkpoint_name,
    node_data_dir,
    tri_data_dir,
    total_timesteps=144,
    steps_per_file=144,
    pred_step=1
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = elementtransformer.FVCOMDataset(
        node_data_dir=node_data_dir,
        tri_data_dir=tri_data_dir,
        total_timesteps=total_timesteps,
        steps_per_file=steps_per_file,
        pred_step=pred_step
    )

    total_samples = len(full_dataset)

    test_indices = list(range(total_samples))

    test_dataset = Subset(full_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = elementtransformer.FVCOMModel(
        node=60882, triangle=115443, node_var=13,
        triangle_var=18, embed_dim=256,
        mlp_ratio=4., nhead=2, num_layers=2,
        neighbor_table=None, dropout=0.1
    ).to(device)

    criterion = elementtransformer.WeightedMAEMSELoss().to(device)

    model.eval()
    # test_loss = 0.0
    loss_list = []
    iter = 1
    node_pred_list = []
    tri_pred_list = []
    with torch.no_grad():
        for (node_x, tri_x), (node_y, tri_y) in test_loader:
            node_x, tri_x = node_x.to(device), tri_x.to(device)
            node_y, tri_y = node_y.to(device), tri_y.to(device)

            node_pred, tri_pred = model.predict(node_x, tri_x, checkpoint_name)

            loss_node = criterion(node_pred, node_y)
            loss_tri = criterion(tri_pred, tri_y)
            loss = loss_node + loss_tri

            # test_loss += loss.item()
            node_pred_list.append(node_pred.squeeze(0).cpu())
            tri_pred_list.append(tri_pred.squeeze(0).cpu())
            loss_list.append(loss.item())
            print(iter)
            iter += 1
    # visualization.img_output(np.array(node_target_list), np.array(tri_target_list),
    #                          np.array(node_pred_list),np.array(tri_pred_list))

    # test_loss /= len(test_loader)
    np.save('144-mul-loss_list', np.array(loss_list))
    np.save('144-mul-node_pred_list', np.array(node_pred_list))
    np.save('144-mul-tri_pred_list', np.array(tri_pred_list))
    print(np.array(node_pred_list).shape)
    print(np.array(tri_pred_list).shape)
    return loss_list

def test_figure_export(title, loss2, center_text="Your Text Here"):
    x = np.arange(0, len(loss2))

    plt.figure(figsize=(12, 6))
    plt.plot(x, loss2, label='single step', color='b')
    # plt.plot(x, loss2, label='auto-regression', color='r')
    plt.title(title, fontsize=16)
    plt.xlabel('Step', fontsize=16)
    plt.ylabel('test loss', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.gcf().autofmt_xdate()

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # plt.text(
    #     0.5, 0.5, 
    #     center_text,
    #     fontsize=14,
    #     ha='center',
    #     va='bottom',
    #     transform=plt.gca().transAxes,
    #     color='red',
    #     alpha=0.8
    # )

    plt.savefig('1.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # checkpoint_name='checkpoints/2025_12_31_17_40_4A40.pth'

    # loss_list = predict_n_steps(
    #     checkpoint_name=checkpoint_name,
    #     node_data_dir="dataset/node/data/",
    #     triangle_data_dir="dataset/triangle/data/",
    #     total_timesteps=144,
    #     steps_per_file=144,
    #     pred_step=1)
    
    # print(test_loss)
    loss_list = np.load('loss_pred.npy')
    print(loss_list.shape)
    test_figure_export('Prediction Loss', loss_list)

    # test_loss = np.load('test_loss.npy')
    # loss_list = np.load('loss_list.npy')
    # print(len(loss_list))
    # print(test_loss)
    # month_loss_list = []
    # for i in range(0, 31):
    #     month_loss_list.append(sum(loss_list[i:i+23])/23)
    # print(len(month_loss_list))