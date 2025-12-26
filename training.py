import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import elementtransformer
import time
import processing

def training_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 6
    N_node = 60882
    N_tri = 115443
    node_in_dim = 13
    triangle_in_dim = 18
    embed_dim=256

    node_feat = torch.randn(T, N_node, node_in_dim).to(device)
    triangle_feat = torch.randn(T, N_tri, triangle_in_dim).to(device)

    print(f"Input shapes:")
    print(f"  node_feat: {node_feat.shape}")
    print(f"  triangle_feat: {triangle_feat.shape}\n")

    model = elementtransformer.FVCOMModel(
        node=N_node, triangle=N_tri, node_var=node_in_dim,
        triangle_var=triangle_in_dim, embed_dim=embed_dim,
        mlp_ratio=4., nhead=2, num_layers=2,
        neighbor_table=None, dropout=0.1
    ).to(device)

    with torch.no_grad():
        node_pred, triangle_pred = model(node_feat, triangle_feat)
    print(node_pred.shape)
    print(triangle_pred.shape)

    print("\n All shapes are correct!")

def training(
    node_data_dir,
    tri_data_dir,
    num_epochs,
    checkpoint_name_out,
    total_timesteps=24,
    steps_per_file=24,
    batch_size=1,
    t_in=1,
    t_out=1,
    early_stop_patience=25
):
    start_time = time.time()
    best_loss = float('inf')
    early_stop_cnt = 0
    best_epoch = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = elementtransformer.FVCOMDataset(
        node_data_dir=node_data_dir,
        tri_data_dir=tri_data_dir,
        total_timesteps=total_timesteps,
        steps_per_file=steps_per_file,
        t_in=t_in, t_out=t_out
    )

    total_samples = len(full_dataset)
    print(total_samples)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = elementtransformer.FVCOMModel(
        node=60882, triangle=115443, node_var=5,
        triangle_var=6, embed_dim=256,
        mlp_ratio=4., nhead=2, num_layers=2,
        t_in=t_in, t_out=t_out,
        neighbor_table=None, dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = elementtransformer.WeightedMAEMSELoss().to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        iter = 0
        for (node_x, tri_x), (node_y, tri_y) in train_loader:
            node_x, tri_x = node_x.to(device), tri_x.to(device)
            node_y, tri_y = node_y.squeeze(0).to(device), tri_y.squeeze(0).to(device)

            node_pred, tri_pred = model(node_x, tri_x)
            print("Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "input node:      ", node_x.shape)
            print("Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "input triangle:  ", tri_x.shape)
            print("Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "target node:     ", node_y.shape)
            print("Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "target triangle: ", tri_y.shape)
            print("Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "pred node:       ", node_pred.shape)
            print("Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "pred triangle:   ", tri_pred.shape)

            loss_node = criterion(node_pred, node_y)
            loss_tri = criterion(tri_pred, tri_y)
            loss = loss_node + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            iter += 1


        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        iter = 0
        with torch.no_grad():
            for (node_x, tri_x), (node_y, tri_y) in val_loader:
                node_x, tri_x = node_x.to(device), tri_x.to(device)
                node_y, tri_y = node_y.squeeze(0).to(device), tri_y.squeeze(0).to(device)

                node_pred, tri_pred = model(node_x, tri_x)
                print("Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "input node:      ", node_x.shape)
                print("Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "input triangle:  ", tri_x.shape)
                print("Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "target node:     ", node_y.shape)
                print("Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "target triangle: ", tri_y.shape)
                print("Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "pred node:       ", node_pred.shape)
                print("Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "pred triangle:   ", tri_pred.shape)

                loss_node = criterion(node_pred, node_y)
                loss_tri = criterion(tri_pred, tri_y)
                loss = loss_node + loss_tri

                val_loss += loss.item()
                iter += 1

        val_loss /= len(val_loader)
        model.train()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best Val: {best_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            early_stop_cnt = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_name_out)

            print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
        else:
            early_stop_cnt += 1
            print(f"No improvement. Patience: {early_stop_cnt}/{early_stop_patience}")

        if early_stop_cnt >= early_stop_patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training finished. Total time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print(f"Best model at epoch {best_epoch} with val_loss: {best_loss:.6f}")

if __name__ == "__main__":
    start_time = time.time()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime(start_time))
    # training_test()

    training(
    node_data_dir="dataset/node/data/",
    tri_data_dir="dataset/triangle/data/",
    num_epochs=100,
    checkpoint_name_out="checkpoints/" + timestamp_str+ "_best_model.pth",
    total_timesteps=6,
    batch_size=1)
    