import torch
from pathlib import Path
from dataset.dezh import TranslationDataset
#学生模型测试
# 配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = "data/de-zh/de-zh.txt"
student_model_path = "./train_process/distillation-dezh/distillation_checkpoints/best_student.pt"

# 初始化数据集
dataset = TranslationDataset(data_dir)
max_seq_length = 42


def translate(src: str) -> str:
    """
    使用知识蒸馏学生模型进行德中翻译
    :param src: 德语句子
    :return: 翻译后的中文句子
    """
    # 加载学生模型
    student_model = torch.load(student_model_path, map_location=device)
    student_model.to(device)
    student_model.eval()

    # 预处理输入句子
    src_tokens = [0] + dataset.de_vocab(dataset.de_tokenizer(src)) + [1]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

    # 初始化目标序列（以<bos>开头）
    tgt_tensor = torch.tensor([[0]]).to(device)

    # 自回归生成翻译结果
    with torch.no_grad():
        for _ in range(max_seq_length):
            out = student_model(src_tensor, tgt_tensor)
            predict = student_model.predictor(out[:, -1])
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


# 测试翻译
if __name__ == '__main__':
    print("知识蒸馏学生模型 - 德中翻译测试")
    print("=" * 60)
    
    # 使用德语圣经句子进行测试
    test_sentences = [
        "Am Anfang schuf Gott Himmel und Erde.",  # 创世纪 1:1
        "Und Gott sprach: Es werde Licht! Und es ward Licht.",  # 创世纪 1:3
        "Denn also hat Gott die Welt geliebt, dass er seinen eingeborenen Sohn gab.",  # 约翰福音 3:16
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i}. 德语: {sentence}")
        translation = translate(sentence)
        print(f"   中文: {translation}")
        print("-" * 60)
    
    print("测试完成！")
