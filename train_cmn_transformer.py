#transformer模型训练
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dezh import TranslationDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad, log_softmax
from tqdm import tqdm
from pathlib import Path
from model.transformer import TranslationModel

# 工作目录配置
base_dir = "./train_process/transformer-dezh"
work_dir = Path(base_dir)
model_dir = Path(base_dir + "/transformer_checkpoints")

# 创建目录
work_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

log_dir = base_dir + "/logs"

# 训练参数配置
model_checkpoint = None  # 'model_10000.pt'
batch_size = 128
epochs = 70
save_after_step = 400
max_seq_length = 40
data_dir = "data/de-zh/de-zh.txt"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 数据加载和批处理函数
def collate_fn(batch):
    bs_id = 0  # <bos> index
    eos_id = 1  # <eos> index
    pad_id = 2  # <pad> index

    src_list, tgt_list = [], []

    for _src, _tgt in batch:
        src_tensor = torch.tensor(_src[:max_seq_length - 2], dtype=torch.int64)
        tgt_tensor = torch.tensor(_tgt[:max_seq_length - 2], dtype=torch.int64)

        processed_src = torch.cat([
            torch.tensor([bs_id], dtype=torch.int64),
            src_tensor,
            torch.tensor([eos_id], dtype=torch.int64)
        ])

        processed_tgt = torch.cat([
            torch.tensor([bs_id], dtype=torch.int64),
            tgt_tensor,
            torch.tensor([eos_id], dtype=torch.int64)
        ])

        src_list.append(pad(
            processed_src,
            (0, max_seq_length - len(processed_src)),
            value=pad_id
        ))

        tgt_list.append(pad(
            processed_tgt,
            (0, max_seq_length - len(processed_tgt)),
            value=pad_id
        ))

    src = torch.stack(src_list).to(device)
    tgt = torch.stack(tgt_list).to(device)
    tgt_y = tgt[:, 1:]
    tgt = tgt[:, :-1]
    n_tokens = (tgt_y != pad_id).sum()

    return src, tgt, tgt_y, n_tokens


# 数据集加载
dataset = TranslationDataset(data_dir)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 模型加载
if model_checkpoint:
    model = torch.load(model_dir / model_checkpoint)
else:
    model = TranslationModel(256, dataset.de_vocab, dataset.zh_vocab, max_seq_length, device)
model = model.to(device)


# 损失函数
class TranslationLoss(nn.Module):
    def __init__(self):
        super(TranslationLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = 2

    def forward(self, x, target):
        x = log_softmax(x, dim=-1)
        true_dist = torch.zeros_like(x)
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0)
        mask = (target == self.padding_idx).nonzero(as_tuple=False)
        if mask.numel() > 0:
            true_dist[mask.squeeze(), :] = 0
        return self.criterion(x, true_dist.detach())


criterion = TranslationLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
writer = SummaryWriter(log_dir)


# 训练函数
def train():
    step = 0
    pre_loss = float('inf')

    for epoch in range(epochs):
        loop = tqdm(train_loader)
        for src, tgt, tgt_y, n_tokens in loop:
            optimizer.zero_grad()

            out = model(src, tgt)
            out = model.predictor(out)

            # 计算损失
            loss = criterion(
                out.contiguous().view(-1, out.size(-1)),
                tgt_y.contiguous().view(-1)
            ) / n_tokens

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 记录日志
            writer.add_scalar('loss', loss.item(), step)

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item())

            step += 1

            # 定期保存模型
            if step != 0 and step % save_after_step == 0:
                torch.save(model, model_dir / f"model_{step}.pt")
                if loss.item() < pre_loss:
                    torch.save(model, model_dir / 'best.pt')
                    pre_loss = loss.item()


if __name__ == "__main__":
    train()
