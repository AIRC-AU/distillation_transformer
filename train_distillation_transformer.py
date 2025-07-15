import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dezh import TranslationDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad, log_softmax
from tqdm import tqdm
from pathlib import Path
from model.transformer import DistillationTranslationModel

#蒸馏模型训练
# 工作目录配置
base_dir = "./train_process/distillation-dezh"
work_dir = Path(base_dir)
model_dir = Path(base_dir + "/distillation_checkpoints")

# 创建目录
work_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

log_dir = base_dir + "/logs"

# 训练参数配置
teacher_model_path = "./train_process/transformer-dezh/transformer_checkpoints/best.pt"
model_checkpoint = None  # 'model_10000.pt'
batch_size = 90
epochs = 100
save_after_step = 600
max_seq_length = 40
data_dir = "data/de-zh/de-zh.txt"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 学生模型参数（比教师模型更小）
d_model_student = 128  # 教师模型是256，学生模型减半


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

# 知识蒸馏模型加载
if model_checkpoint:
    distillation_model = torch.load(model_dir / model_checkpoint)
else:
    distillation_model = DistillationTranslationModel(
        teacher_model_path, d_model_student, dataset.de_vocab, dataset.zh_vocab, max_seq_length, device
    )
distillation_model = distillation_model.to(device)


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
# 只优化学生模型的参数
optimizer = torch.optim.Adam(distillation_model.student_model.parameters(), lr=0.0001)
writer = SummaryWriter(log_dir)


# 知识蒸馏训练函数
def train_distillation():
    step = 0
    pre_loss = float('inf')

    print(f"开始知识蒸馏训练...")
    print(f"教师模型路径: {teacher_model_path}")
    print(f"学生模型维度: {d_model_student}")
    print(f"蒸馏温度: {distillation_model.temperature}")
    print(f"蒸馏权重: {distillation_model.alpha}")

    for epoch in range(epochs):
        loop = tqdm(train_loader)
        epoch_total_loss = 0
        epoch_task_loss = 0
        epoch_distill_loss = 0
        
        for src, tgt, tgt_y, n_tokens in loop:
            optimizer.zero_grad()

            # 获取学生和教师模型的输出
            student_output, student_logits, teacher_logits = distillation_model(src, tgt, mode='both')

            # 计算知识蒸馏损失
            total_loss, task_loss, distillation_loss = distillation_model.compute_distillation_loss(
                student_logits.contiguous().view(-1, student_logits.size(-1)),
                teacher_logits.contiguous().view(-1, teacher_logits.size(-1)),
                tgt_y.contiguous().view(-1),
                criterion
            )
            
            # 标准化损失
            total_loss = total_loss / n_tokens
            task_loss = task_loss / n_tokens
            distillation_loss = distillation_loss / n_tokens

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 累计损失
            epoch_total_loss += total_loss.item()
            epoch_task_loss += task_loss.item()
            epoch_distill_loss += distillation_loss.item()

            # 记录日志
            writer.add_scalar('loss/total', total_loss.item(), step)
            writer.add_scalar('loss/task', task_loss.item(), step)
            writer.add_scalar('loss/distillation', distillation_loss.item(), step)

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(
                total_loss=total_loss.item(),
                task_loss=task_loss.item(),
                distill_loss=distillation_loss.item()
            )

            step += 1

            # 定期保存模型
            if step != 0 and step % save_after_step == 0:
                torch.save(distillation_model, model_dir / f"distillation_model_{step}.pt")
                # 单独保存学生模型
                torch.save(distillation_model.student_model, model_dir / f"student_model_{step}.pt")
                
                if total_loss.item() < pre_loss:
                    torch.save(distillation_model, model_dir / 'best_distillation.pt')
                    torch.save(distillation_model.student_model, model_dir / 'best_student.pt')
                    pre_loss = total_loss.item()

        # 每个epoch结束后打印平均损失
        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_task_loss = epoch_task_loss / len(train_loader)
        avg_distill_loss = epoch_distill_loss / len(train_loader)
        
        print(f"Epoch {epoch} - 平均总损失: {avg_total_loss:.4f}, "
              f"任务损失: {avg_task_loss:.4f}, 蒸馏损失: {avg_distill_loss:.4f}")


if __name__ == "__main__":
    train_distillation()
