import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#transform-蒸馏模型

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""

    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TranslationModel(nn.Module):
    """Transformer-based translation model."""

    def __init__(self, d_model, src_vocab, tgt_vocab, max_seq_length, device, dropout=0.1):
        super(TranslationModel, self).__init__()
        self.device = device

        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)

        self.positional_encoding = PositionalEncoding(d_model, dropout, device, max_len=max_seq_length)

        self.transformer = nn.Transformer(
            d_model=d_model,
            dropout=dropout,
            batch_first=True
        )

        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        """
        Forward pass of the translation model.
        Args:
            src: (batch_size, src_seq_len)
            tgt: (batch_size, tgt_seq_len)
        Returns:
            Transformer decoder output
        """
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
        src_key_padding_mask = self.get_key_padding_mask(src)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt)

        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        out = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return out

    @staticmethod
    def get_key_padding_mask(tokens, pad_idx=2):
        """Create key padding mask for transformer."""
        return tokens == pad_idx


class DistillationTranslationModel(nn.Module):
    """知识蒸馏翻译模型，包含教师模型和学生模型"""

    def __init__(self, teacher_model_path, d_model_student, src_vocab, tgt_vocab, max_seq_length, device, dropout=0.1):
        super(DistillationTranslationModel, self).__init__()
        self.device = device
        self.temperature = 4.0  # 蒸馏温度
        self.alpha = 0.7  # 蒸馏损失权重

        # 加载预训练的教师模型
        self.teacher_model = torch.load(teacher_model_path, map_location=device)
        self.teacher_model.eval()
        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 创建学生模型（更小的模型）
        self.student_model = TranslationModel(
            d_model_student, src_vocab, tgt_vocab, max_seq_length, device, dropout
        )

    def forward(self, src, tgt, mode='student'):
        """
        前向传播
        Args:
            src: 源语言序列
            tgt: 目标语言序列
            mode: 'student', 'teacher', 或 'both'
        """
        if mode == 'teacher':
            with torch.no_grad():
                return self.teacher_model(src, tgt)
        elif mode == 'student':
            return self.student_model(src, tgt)
        elif mode == 'both':
            # 同时获取教师和学生的输出
            with torch.no_grad():
                teacher_output = self.teacher_model(src, tgt)
                teacher_logits = self.teacher_model.predictor(teacher_output)
            student_output = self.student_model(src, tgt)
            student_logits = self.student_model.predictor(student_output)
            return student_output, student_logits, teacher_logits

    def get_student_output(self, src, tgt):
        """获取学生模型输出"""
        return self.student_model(src, tgt)

    def get_teacher_output(self, src, tgt):
        """获取教师模型输出"""
        with torch.no_grad():
            return self.teacher_model(src, tgt)

    def compute_distillation_loss(self, student_logits, teacher_logits, target, criterion):
        """
        计算知识蒸馏损失
        Args:
            student_logits: 学生模型的logits
            teacher_logits: 教师模型的logits
            target: 真实标签
            criterion: 原始损失函数
        """
        # 计算原始任务损失（学生模型与真实标签）
        task_loss = criterion(student_logits, target)

        # 计算蒸馏损失（学生模型与教师模型的软标签）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

        # 组合损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss

        return total_loss, task_loss, distillation_loss


# -------------------------------
# Example usage
# -------------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from dataset.cmneng import TranslationDataset

    dataset = TranslationDataset("../data/cmn-eng/cmn.txt")

    model = TranslationModel(512, dataset.en_vocab, dataset.zh_vocab, 50, device)
    model = model.to(device)

    # Prepare input sentence
    en = "hello world"
    en_ids = [0] + dataset.en_vocab(dataset.en_tokenizer(en)) + [1]  # BOS + tokens + EOS
    input = torch.tensor([en_ids]).to(device)

    # Prepare partial target sentence (decoder input)
    zh = "你"
    zh_ids = [0] + dataset.zh_vocab(dataset.zh_tokenizer(zh))       # BOS + tokens
    output = torch.tensor([zh_ids]).to(device)

    # Model forward
    result = model(input, output)  # (batch_size, tgt_seq_len, d_model)

    # Predict next token
    predict = model.predictor(result[:, -1])  # Use last decoder output
    print("Logits:", predict)

    # Get the most probable token ID
    y = torch.argmax(predict, dim=1).cpu().item()
    print("Predicted token:", dataset.zh_vocab.lookup_tokens([y]))
