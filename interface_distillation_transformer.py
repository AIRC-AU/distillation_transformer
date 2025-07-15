#蒸馏模型预测

import torch
import time
from pathlib import Path
from dataset.dezh import TranslationDataset

# 配置工作目录和设备
base_dir = "./train_process/distillation-dezh"
work_dir = Path(base_dir)
model_dir = Path(base_dir + "/distillation_checkpoints")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = "data/de-zh/de-zh.txt"

# 初始化数据集
dataset = TranslationDataset(data_dir)
max_seq_length = 42  # 句子最大长度


def translate_with_student_model(src: str) -> str:
    """
    使用知识蒸馏训练的学生模型进行德中翻译
    :param src: 德语句子，如"Ich liebe maschinelles Lernen."
    :return: 翻译后的中文句子，如"我喜欢机器学习"
    """
    # 加载学生模型
    student_model = torch.load(model_dir / 'best_student.pt', map_location=device)
    student_model.to(device)
    student_model.eval()

    # 预处理输入句子
    src_tokens = [0] + dataset.de_vocab(dataset.de_tokenizer(src)) + [1]  # 使用德语分词器和词汇表
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)  # shape: [1, seq_len]

    # 初始化目标序列（以<bos>开头）
    tgt_tensor = torch.tensor([[0]]).to(device)  # shape: [1, 1]

    # 自回归生成翻译结果
    with torch.no_grad():
        for _ in range(max_seq_length):
            out = student_model(src_tensor, tgt_tensor)  # transformer计算
            predict = student_model.predictor(out[:, -1])  # 只取最后一个词的预测
            next_token = torch.argmax(predict, dim=1)  # 选择概率最高的词

            # 将新词添加到目标序列
            tgt_tensor = torch.cat([tgt_tensor, next_token.unsqueeze(0)], dim=1)

            # 遇到<eos>（索引1）则停止生成
            if next_token.item() == 1:
                break

    # 将token索引转换为中文句子
    tgt_tokens = tgt_tensor.squeeze().tolist()
    translated = ' '.join(dataset.zh_vocab.lookup_tokens(tgt_tokens))

    # 清理特殊标记并返回结果
    return translated.replace("<s>", "").replace("</s>", "").strip()


def translate_with_teacher_model(src: str) -> str:
    """
    使用原始教师模型进行德中翻译（用于对比）
    :param src: 德语句子
    :return: 翻译后的中文句子
    """
    # 加载教师模型
    teacher_model = torch.load("./train_process/transformer-dezh/transformer_checkpoints/best.pt", map_location=device)
    teacher_model.to(device)
    teacher_model.eval()

    # 预处理输入句子
    src_tokens = [0] + dataset.de_vocab(dataset.de_tokenizer(src)) + [1]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

    # 初始化目标序列（以<bos>开头）
    tgt_tensor = torch.tensor([[0]]).to(device)

    # 自回归生成翻译结果
    with torch.no_grad():
        for _ in range(max_seq_length):
            out = teacher_model(src_tensor, tgt_tensor)
            predict = teacher_model.predictor(out[:, -1])
            next_token = torch.argmax(predict, dim=1)

            # 将新词添加到目标序列
            tgt_tensor = torch.cat([tgt_tensor, next_token.unsqueeze(0)], dim=1)

            # 遇到<eos>（索引1）则停止生成
            if next_token.item() == 1:
                break

    # 将token索引转换为中文句子
    tgt_tokens = tgt_tensor.squeeze().tolist()
    translated = ' '.join(dataset.zh_vocab.lookup_tokens(tgt_tokens))

    # 清理特殊标记并返回结果
    return translated.replace("<s>", "").replace("</s>", "").strip()


def compare_models(src: str):
    """
    比较教师模型和学生模型的翻译结果和推理时间
    :param src: 德语句子
    """
    print(f"原始德语句子: {src}")
    print("-" * 80)
    
    # 测试教师模型
    start_time = time.time()
    teacher_translation = translate_with_teacher_model(src)
    teacher_time = time.time() - start_time
    
    print(f"教师模型翻译: {teacher_translation}")
    print(f"教师模型推理时间: {teacher_time:.4f}秒")
    print("-" * 40)
    
    # 测试学生模型
    start_time = time.time()
    student_translation = translate_with_student_model(src)
    student_time = time.time() - start_time
    
    print(f"学生模型翻译: {student_translation}")
    print(f"学生模型推理时间: {student_time:.4f}秒")
    print("-" * 40)
    
    # 计算加速比
    speedup = teacher_time / student_time if student_time > 0 else 0
    print(f"推理加速比: {speedup:.2f}x")
    print("=" * 80)


def get_model_size(model_path):
    """获取模型文件大小"""
    try:
        size_bytes = Path(model_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except:
        return 0


def model_comparison():
    """比较教师模型和学生模型的大小"""
    teacher_path = "./train_process/transformer-dezh/transformer_checkpoints/best.pt"
    student_path = model_dir / 'best_student.pt'
    
    teacher_size = get_model_size(teacher_path)
    student_size = get_model_size(student_path)
    
    print("模型大小比较:")
    print(f"教师模型大小: {teacher_size:.2f} MB")
    print(f"学生模型大小: {student_size:.2f} MB")
    if teacher_size > 0:
        compression_ratio = teacher_size / student_size if student_size > 0 else 0
        print(f"模型压缩比: {compression_ratio:.2f}x")
    print("-" * 80)


# 测试翻译
if __name__ == '__main__':
    print("知识蒸馏模型测试")
    print("=" * 80)
    
    # 显示模型大小比较
    model_comparison()
    
    # 使用德语圣经句子进行测试
    test_sentences = [
        "Am Anfang schuf Gott Himmel und Erde.",  # 创世纪 1:1
        "Und Gott sprach: Es werde Licht! Und es ward Licht.",  # 创世纪 1:3
        "Du sollst nicht töten."
        "Denn also hat Gott die Welt geliebt, dass er seinen eingeborenen Sohn gab.",  # 约翰福音 3:16
        "Ich bin der Weg und die Wahrheit und das Leben.",  # 约翰福音 14:6
        "Liebe deinen Nächsten wie dich selbst."  # 马太福音 22:39
    ]
    
    for sentence in test_sentences:
        compare_models(sentence)
        print()
    
    print("测试完成！")
    print("知识蒸馏模型优势:")
    print("1. 模型更小，占用更少存储空间")
    print("2. 推理速度更快，适合部署")
    print("3. 保持了教师模型的大部分性能")
