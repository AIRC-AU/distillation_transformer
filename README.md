# 德中圣经翻译 - 知识蒸馏模型

## 项目概述

本项目实现了基于知识蒸馏（Knowledge Distillation）的德中圣经翻译模型。通过使用已训练的大型Transformer模型作为教师模型，训练一个更小、更快的学生模型，在保持翻译质量的同时显著提升推理速度和降低模型大小。

## 模型架构

### 教师模型
- **模型**: 已训练的Transformer模型
- **路径**: `train_process/transformer-dezh/transformer_checkpoints/best.pt`
- **维度**: d_model=256
- **特点**: 大模型，翻译质量高，但推理速度慢

### 学生模型
- **模型**: 新训练的小型Transformer模型
- **维度**: d_model=128 (教师模型的一半)
- **特点**: 小模型，推理速度快，模型大小小

### 知识蒸馏原理
1. **软标签学习**: 学生模型学习教师模型的输出概率分布
2. **温度缩放**: 使用温度参数T=4.0软化概率分布
3. **损失函数组合**: 
   - 蒸馏损失 (α=0.7): KL散度损失，学习教师模型的知识
   - 任务损失 (1-α=0.3): 交叉熵损失，学习真实标签

## 文件结构

```
my-pytorch-deeplearning/
├── model/
│   └── transformer.py                    # 包含知识蒸馏模型类
├── train_distillation_transformer.py    # 知识蒸馏训练脚本
├── interface_distillation_transformer.py # 知识蒸馏接口（对比测试）
├── translate_student.py                 # 简化的学生模型接口
├── start_distillation_training.py       # 训练启动器
└── train_process/
    └── distillation-dezh/               # 知识蒸馏训练输出目录
        ├── distillation_checkpoints/    # 模型检查点
        └── logs/                        # TensorBoard日志
```

## 使用方法

### 1. 环境检查和训练启动

```bash
python start_distillation_training.py
```

这个脚本会：
- 检查CUDA环境
- 验证教师模型存在
- 检查数据集
- 显示训练配置
- 启动知识蒸馏训练

### 2. 直接启动训练

```bash
python train_distillation_transformer.py
```

### 3. 使用学生模型进行翻译

```bash
python translate_student.py
```

### 4. 对比教师和学生模型

```bash
python interface_distillation_transformer.py
```

## 训练配置

| 参数 | 值 | 说明 |
|------|----|----|
| 教师模型维度 | 256 | 原始大模型 |
| 学生模型维度 | 128 | 压缩后小模型 |
| 蒸馏温度 | 4.0 | 软化概率分布 |
| 蒸馏权重α | 0.7 | 蒸馏损失权重 |
| 任务权重(1-α) | 0.3 | 真实标签损失权重 |
| 批次大小 | 128 | 训练批次大小 |
| 学习率 | 0.0001 | Adam优化器学习率 |
| 训练轮数 | 50 | 最大训练轮数 |

## 预期效果

### 模型压缩
- **模型大小**: 减少约50%
- **参数数量**: 显著减少
- **存储需求**: 更少的磁盘空间

### 推理加速
- **推理速度**: 提升1.5-3倍
- **内存占用**: 减少约50%
- **部署友好**: 适合移动端和边缘设备

### 翻译质量
- **质量保持**: 保持教师模型85-95%的翻译质量
- **圣经翻译**: 专门针对德中圣经文本优化

## 测试句子

项目包含以下德语圣经句子用于测试：

1. "Am Anfang schuf Gott Himmel und Erde." (创世纪 1:1)
2. "Und Gott sprach: Es werde Licht! Und es ward Licht." (创世纪 1:3)
3. "Denn also hat Gott die Welt geliebt, dass er seinen eingeborenen Sohn gab." (约翰福音 3:16)
4. "Ich bin der Weg und die Wahrheit und das Leben." (约翰福音 14:6)
5. "Liebe deinen Nächsten wie dich selbst." (马太福音 22:39)

## 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir=train_process/distillation-dezh/logs
```

可以观察：
- 总损失变化
- 任务损失变化  
- 蒸馏损失变化
- 训练进度

## 注意事项

1. **教师模型**: 确保已完成基础Transformer模型的训练
2. **GPU内存**: 知识蒸馏需要同时加载教师和学生模型，需要足够的GPU内存
3. **训练时间**: 知识蒸馏训练时间比普通训练稍长
4. **模型保存**: 系统会自动保存最佳模型和定期检查点

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少batch_size
2. **教师模型加载失败**: 检查教师模型路径
3. **数据集加载错误**: 确认数据集格式正确
4. **训练中断**: 可以从检查点恢复训练

### 性能优化

1. **使用混合精度训练**: 可以进一步加速训练
2. **调整蒸馏参数**: 根据需要调整温度和权重
3. **学生模型大小**: 可以进一步减小学生模型维度
