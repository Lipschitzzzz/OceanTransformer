import os
import torch
import elementtransformer
import processing
import visualization
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluation(
    checkpoint_name,
    node_data_dir,
    tri_data_dir,
    total_timesteps=144 * 7,
    steps_per_file=144,
    pred_step=1,
    batch_size=1
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

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = elementtransformer.FVCOMModel(
        node=60882, triangle=115443, node_var=13,
        triangle_var=18, embed_dim=256,
        mlp_ratio=4., nhead=2, num_layers=2,
        neighbor_table=None, dropout=0.1
    ).to(device)

    criterion = elementtransformer.WeightedMAEMSELoss().to(device)

    model.eval()
    test_loss = 0.0
    loss_list = []
    iter = 1
    node_pred_list = []
    tri_pred_list = []
    with torch.no_grad():
        for (node_x, tri_x), (node_y, tri_y) in test_loader:
            node_x, tri_x = node_x.to(device), tri_x.to(device)
            node_y, tri_y = node_y.to(device), tri_y.to(device)

            node_pred, tri_pred = model.predict(node_x, tri_x, checkpoint_name)

            # loss_node = criterion(node_pred, node_y)
            # loss_tri = criterion(tri_pred, tri_y)
            # loss = loss_node + loss_tri

            # test_loss += loss.item()
            node_pred_list.append(node_pred.squeeze(0).cpu())
            tri_pred_list.append(tri_pred.squeeze(0).cpu())
            print(iter)
            iter += 1
    # visualization.img_output(np.array(node_target_list), np.array(tri_target_list),
    #                          np.array(node_pred_list),np.array(tri_pred_list))

    # test_loss /= len(test_loader)
    # np.save('7-loss_list', np.array(loss_list))
    # np.save('7-test_loss', np.array(test_loss))
    np.save('7-node_pred_list', np.array(node_pred_list))
    np.save('7-tri_pred_list', np.array(tri_pred_list))
    # print(np.array(node_pred_list).shape)
    # print(np.array(tri_pred_list).shape)
    return loss_list, test_loss

def train_figure_export(title, loss):
    epochs = np.arange(1, 201)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label=title, color='b')
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(title, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def test_figure_export(title, loss, center_text="Your Text Here"):
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
    plt.figure(figsize=(12, 6))
    plt.plot(dates, loss, label=title, color='b')
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('test loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.gcf().autofmt_xdate()

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 在图像正中央添加文字（使用 axes 坐标：x=0.5, y=0.5）
    plt.text(
        0.5, 0.5, 
        center_text,
        fontsize=14,
        ha='center',      # 水平居中
        va='bottom',      # 垂直居中
        transform=plt.gca().transAxes,  # 使用 axes 坐标系（不是数据坐标）
        color='red',      # 可选：文字颜色
        alpha=0.8         # 可选：透明度
    )

    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    checkpoint_name='checkpoints/element-transformer-v1.0.pth'
    loss_list, test_loss = evaluation(
        checkpoint_name=checkpoint_name,
        node_data_dir="dataset/node/data/",
        tri_data_dir="dataset/triangle/data/",
        total_timesteps=24*7,
        steps_per_file=24,
        pred_step=1,
        batch_size=1)
    # print(test_loss)
    # print(loss_list)
    # training_loss = np.load('train_loss.npy')
    # val_loss = np.load('val_loss.npy')
    # lr = np.load('lr.npy')
    # figure_export('training_loss', training_loss)
    # figure_export('validation_loss', val_loss)
    # figure_export('learning_rate', lr)

    # test_loss = np.load('test_loss.npy')
    # loss_list = np.load('loss_list.npy')
    # print(len(loss_list))
    # print(test_loss)
    # month_loss_list = []
    # for i in range(0, 31):
    #     month_loss_list.append(sum(loss_list[i:i+23])/23)
    # print(len(month_loss_list))
    # test_figure_export('2025.01 test loss', month_loss_list, 'average = 0.084980')