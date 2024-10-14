import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils import LossRecorder


def train(args, model, train_loader, val_loader,  device):
    if args.model_name == 'baseline' and args.add_noise:
        name = "baseline_add_noise"
    else:
        name = f"{args.model_name}"
    optimizer = Adam(model.parameters(), args.lr)

    os.makedirs('paras', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    records = []
    best_ce_loss = 1e6
    for epoch in range(args.n_epochs):
        model.train()
        t = tqdm(train_loader)
        train_recorder = LossRecorder(['ce'])
        for data in t:
            optimizer.zero_grad()
            
            data = data.to(device)
            ce_loss = model(data)
            
            ce_loss.backward()
            optimizer.step()

            train_recorder.update('ce', ce_loss.item(), data.mask.sum().item())

            t.set_description(
                "epoch: %d ce: %.3f" % (epoch, train_recorder.avg_loss['ce'])
            )
        
        model.eval()
        val_recorder = LossRecorder(['ce'])
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                ce_loss = model(batch)
                val_recorder.update('ce', ce_loss.item(), batch.mask.sum().item())

            val_ce_loss = val_recorder.avg_loss['ce']
        
        records.append(
            np.array([
                train_recorder.avg_loss['ce'], 
                val_ce_loss, 
            ])[None]
        )
        
        if val_ce_loss < best_ce_loss:
            best_ce_loss = val_ce_loss
            torch.save(model.state_dict(), f"paras/{name}_best.h5")
    
    records = np.concatenate(records, 0)
    torch.save(model.state_dict(), f"paras/{name}.h5")

    plt.plot(records[:, 0], label='train')
    plt.plot(records[:, 1], label='val')
    plt.legend()
    plt.savefig(f"outputs/{name}_loss.png")


def alignif_train(args, model, train_loader, val_loader,  device):
    if args.relax_label:
        name = f"{args.model_name}_relax_label"
    else:
        name = f"{args.model_name}"
    optimizer = Adam(model.parameters(), args.lr)

    os.makedirs('paras', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    records = []
    best_ce_loss = 1e6
    for epoch in range(args.n_epochs):
        model.train()
        t = tqdm(train_loader)
        train_recorder = LossRecorder(['ce', 'node_align', 'edge_align'])
        for data in t:
            optimizer.zero_grad()
            
            data = [datum.to(device) for datum in data]
            ce_loss, node_align_loss, node_align_counts, edge_align_loss, edge_align_counts = model(data)
            
            (ce_loss + node_align_loss + edge_align_loss).backward()
            optimizer.step()

            train_recorder.update('ce', ce_loss.item(), data[0].mask.sum().item())
            train_recorder.update('node_align', node_align_loss.item(), node_align_counts.item())
            train_recorder.update('edge_align', edge_align_loss.item(), edge_align_counts.item())

            t.set_description(
                "epoch: %d ce: %.3f node_align: %.3f edge_align: %.3f" % (
                    epoch, train_recorder.avg_loss['ce'], train_recorder.avg_loss['node_align'], train_recorder.avg_loss['edge_align']
                )
            )
        
        model.eval()
        val_recorder = LossRecorder(['ce', 'node_align', 'edge_align'])
        with torch.no_grad():
            for data in val_loader:
                data = [datum.to(device) for datum in data]
                ce_loss, node_align_loss, node_align_counts, edge_align_loss, edge_align_counts = model(data)
                val_recorder.update('ce', ce_loss.item(), data[0].mask.sum().item())
                val_recorder.update('node_align', node_align_loss.item(), node_align_counts.item())
                val_recorder.update('edge_align', edge_align_loss.item(), edge_align_counts.item())

            val_ce_loss = val_recorder.avg_loss['ce']
        
        records.append(
            np.array([
                train_recorder.avg_loss['ce'], 
                val_ce_loss, 
                train_recorder.avg_loss['node_align'], 
                val_recorder.avg_loss['node_align'], 
                train_recorder.avg_loss['edge_align'], 
                val_recorder.avg_loss['edge_align']
            ])[None]
        )
        
        if val_ce_loss < best_ce_loss:
            best_ce_loss = val_ce_loss
            torch.save(model.state_dict(), f"paras/{name}_best.h5")
    
    records = np.concatenate(records, 0)
    torch.save(model.state_dict(), f"paras/{name}.h5")

    plt.plot(records[:, 0], label='train')
    plt.plot(records[:, 1], label='val')
    plt.legend()
    plt.savefig(f"outputs/{name}_ce_loss.png")

    plt.clf()
    plt.plot(records[:, 2], label='train')
    plt.plot(records[:, 3], label='val')
    plt.legend()
    plt.savefig(f"outputs/{name}_node_align_loss.png")

    plt.clf()
    plt.plot(records[:, 4], label='train')
    plt.plot(records[:, 5], label='val')
    plt.legend()
    plt.savefig(f"outputs/{name}_edge_align_loss.png")