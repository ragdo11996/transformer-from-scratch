import time
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from dataset_iwslt import IWSLTTranslationDataset
from seq2seq_model import TransformerSeq2Seq

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = torch.stack(srcs)
    tgts = torch.stack(tgts)
    return srcs, tgts

def train_one_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_inp)
        B, T, V = logits.size()
        loss = criterion(logits.view(B*T, V), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_inp)
        B, T, V = logits.size()
        loss = criterion(logits.view(B*T, V), tgt_out.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # ===== 数据加载 =====
    seq_len = cfg["dataset"]["seq_len"]
    train_ds = IWSLTTranslationDataset(split="train", seq_len=seq_len)
    val_ds = IWSLTTranslationDataset(split="validation",seq_len=seq_len,src_lang=train_ds.src_lang,tgt_lang=train_ds.tgt_lang,
                                    src_stoi=train_ds.src_stoi,tgt_stoi=train_ds.tgt_stoi)
    batch_size = cfg["training"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    src_vocab = len(train_ds.src_stoi)
    tgt_vocab = len(train_ds.tgt_stoi)
    print(f"Src vocab={src_vocab}, Tgt vocab={tgt_vocab}")

    d_model=cfg["model"]["d_model"]
    n_heads=cfg["model"]["n_heads"]
    d_ff=cfg["model"]["d_ff"]
    n_layers=cfg["model"]["n_layers"]
    model = TransformerSeq2Seq(src_vocab, tgt_vocab, d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(),  lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=train_ds.tgt_stoi["<pad>"])

    os.makedirs("results", exist_ok=True)
    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss = evaluate(model, val_dl, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch}/{cfg['training']['epochs']}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={time.time()-t0:.2f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/best_iwslt.pt")

    # ===== 可视化 =====
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/iwslt_training_curve.png")
    plt.close()

# ===== 推理测试 =====
    model.load_state_dict(torch.load("results/best_iwslt.pt", map_location=device))
    model.eval()
    src, tgt = val_ds[0]
    src = src.unsqueeze(0).to(device)
    tgt_vocab = val_ds.tgt_itos

    pred = torch.tensor([[val_ds.tgt_stoi["<bos>"]]], dtype=torch.long).to(device)
    temperature = 0.9

    for _ in range(64):
        logits = model(src, pred)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

    # 默认贪心采样
        next_token = torch.multinomial(probs, num_samples=1)

    # 防止连续重复
        if pred.size(1) > 1 and next_token.item() == pred[0, -1].item():
            top2 = torch.topk(probs, k=2, dim=-1).indices  # shape [1, 2]
            next_token = top2[:, 1].unsqueeze(1)  # 取第二高概率词

        pred = torch.cat([pred, next_token], dim=1)
        if next_token.item() == val_ds.tgt_stoi["<eos>"]:
            break

    translation = " ".join([tgt_vocab.get(i.item(), "<unk>") for i in pred[0]])
    print("SRC:", " ".join([train_ds.src_itos.get(i.item(), "") for i in src[0] if i.item() > 3]))
    print("PRED:", translation)


if __name__ == "__main__":
    main()
