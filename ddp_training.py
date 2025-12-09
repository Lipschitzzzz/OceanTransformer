import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import time
import os
import numpy as np
import nodetransformer

def train_zero_epoch_ddp(node_data_dir,
    tri_data_dir,
    num_epochs,
    checkpoint_name_out,
    total_timesteps=144 * 7,
    steps_per_file=144,
    pred_step=1,
    batch_size=1,
    early_stop_patience=25
):
    start_time = time.time()
    train_loss_dataset = []
    val_loss_dataset = []
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    model = nodetransformer.FVCOMModel(
        node=60882, triangle=115443, node_var=13,
        triangle_var=18, embed_dim=256,
        mlp_ratio=4., nhead=2, num_layers=2,
        neighbor_table=None, dropout=0.1
    ).to(device)

    full_dataset = nodetransformer.FVCOMDataset(
        node_data_dir=node_data_dir,
        tri_data_dir=tri_data_dir,
        total_timesteps=total_timesteps,
        steps_per_file=steps_per_file,
        pred_step=pred_step
    )

    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nodetransformer.WeightedMAEMSELoss().to(device)

    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=2, pin_memory=True
    )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        iter = 0
        for (node_x, tri_x), (node_y, tri_y) in train_loader:
            node_x, tri_x = node_x.to(device), tri_x.to(device)
            node_y, tri_y = node_y.to(device), tri_y.to(device)

            node_pred, tri_pred = model(node_x, tri_x)
            print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "input node:      ", node_x.shape)
            print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "input triangle:  ", tri_x.shape)
            print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "target node:     ", node_y.shape)
            print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "target triangle: ", tri_y.shape)
            print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "pred node:       ", node_pred.shape)
            print("GPU:", str(local_rank), "Training:", train_size, iter, "epoch:", epoch+1, '-', iter, "pred triangle:   ", tri_pred.shape)

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
                node_y, tri_y = node_y.to(device), tri_y.to(device)

                node_pred, tri_pred = model(node_x, tri_x)
                print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "input node:      ", node_x.shape)
                print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "input triangle:  ", tri_x.shape)
                print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "target node:     ", node_y.shape)
                print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "target triangle: ", tri_y.shape)
                print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "pred node:       ", node_pred.shape)
                print("GPU:", str(local_rank), "Validation:", val_size, iter, "epoch:", epoch+1, '-', iter, "pred triangle:   ", tri_pred.shape)

                loss_node = criterion(node_pred, node_y)
                loss_tri = criterion(tri_pred, tri_y)
                loss = loss_node + loss_tri

                val_loss += loss.item()
                iter += 1

        val_loss /= len(val_loader)
        model.train()

        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size

        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}")
            train_loss_dataset.append(train_loss)
            val_loss_dataset.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                early_stop_cnt = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, checkpoint_name_out)

                print(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}")
            else:
                early_stop_cnt += 1
                print(f"No improvement. Current val_loss: {val_loss:.6f}, Best so far: {best_loss:.6f}, Best epoch {best_epoch}")

            if early_stop_cnt > early_stop_patience:
                print("Early stopped.")
                break

    total_time = time.time() - start_time
    if local_rank == 0:
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    print(world_size, "GPU found")
    assert world_size > 0, "No GPUs available"
    start_time = time.time()
    timestamp_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime(start_time))
    train_zero_epoch_ddp(node_data_dir="dataset/node/data/",
    tri_data_dir="dataset/triangle/data/",
    num_epochs=100,
    checkpoint_name_out="checkpoints/" + timestamp_str+ "_2A100.pth",
    total_timesteps=144,
    pred_step=1,
    batch_size=1)
if __name__ == "__main__":
    main()