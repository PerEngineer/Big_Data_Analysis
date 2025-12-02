"""
实验四：猫狗大战 - 深度学习图像分类实验
=====================================
实验目的：采用深度学习算法解决猫狗图片分类问题
实验内容：
    1. 数据预处理
    2. 构建CNN分类模型（使用迁移学习VGG16/ResNet）
    3. 模型训练与参数调优
    4. 结果分析与可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import warnings
warnings.filterwarnings('ignore')

# 深度学习框架
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 设置随机种子，保证实验可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 1. 数据路径配置 ====================
BASE_DIR = r'd:\Personal\Desktop\大数据分析\实验\实验四'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test1')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 2. 数据探索与分析 ====================
print("\n" + "="*50)
print("2. 数据探索与分析")
print("="*50)

# 获取训练集文件列表
train_files = os.listdir(TRAIN_DIR)
cat_files = [f for f in train_files if f.startswith('cat')]
dog_files = [f for f in train_files if f.startswith('dog')]

print(f"训练集总数: {len(train_files)}")
print(f"猫图片数量: {len(cat_files)}")
print(f"狗图片数量: {len(dog_files)}")

# 获取测试集文件列表
test_files = os.listdir(TEST_DIR)
print(f"测试集总数: {len(test_files)}")

# 分析图片尺寸分布
def analyze_image_sizes(directory, sample_size=500):
    """分析图片尺寸分布"""
    files = os.listdir(directory)
    sample_files = np.random.choice(files, min(sample_size, len(files)), replace=False)
    
    widths, heights = [], []
    for f in sample_files:
        try:
            img = Image.open(os.path.join(directory, f))
            widths.append(img.size[0])
            heights.append(img.size[1])
        except:
            continue
    
    return widths, heights

print("\n分析训练集图片尺寸分布...")
widths, heights = analyze_image_sizes(TRAIN_DIR)
print(f"宽度范围: {min(widths)} - {max(widths)}, 平均: {np.mean(widths):.1f}")
print(f"高度范围: {min(heights)} - {max(heights)}, 平均: {np.mean(heights):.1f}")

# ==================== 3. 数据预处理 ====================
print("\n" + "="*50)
print("3. 数据预处理")
print("="*50)

# 图片尺寸设置
IMG_SIZE = 224  # VGG16/ResNet标准输入尺寸

# 数据增强策略
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),           # 先放大
    transforms.RandomCrop(IMG_SIZE),          # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转
    transforms.RandomRotation(15),            # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                        std=[0.229, 0.224, 0.225])
])

# 验证集和测试集不做数据增强
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

print("数据增强策略:")
print("  - 随机裁剪到224x224")
print("  - 随机水平翻转(p=0.5)")
print("  - 随机旋转(±15°)")
print("  - 颜色抖动(亮度、对比度、饱和度)")
print("  - ImageNet标准归一化")

# ==================== 4. 自定义数据集类 ====================
class CatDogDataset(Dataset):
    """猫狗数据集类"""
    def __init__(self, file_list, directory, transform=None, is_test=False):
        self.file_list = file_list
        self.directory = directory
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.directory, img_name)
        
        # 读取图片
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图片损坏，返回一个随机图片
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            # 测试集返回图片ID
            img_id = int(img_name.split('.')[0])
            return image, img_id
        else:
            # 训练集返回标签 (cat=0, dog=1)
            label = 0 if img_name.startswith('cat') else 1
            return image, label

# ==================== 5. 准备数据加载器 ====================
print("\n" + "="*50)
print("5. 准备数据加载器")
print("="*50)

# 为加快训练速度，采样部分数据（CPU训练时使用）
SAMPLE_RATIO = 0.4  # 使用40%的数据，猫狗各5000张
# 均衡采样：确保猫狗比例一致
cat_sample = cat_files[:int(len(cat_files) * SAMPLE_RATIO)]
dog_sample = dog_files[:int(len(dog_files) * SAMPLE_RATIO)]
sampled_files = cat_sample + dog_sample
np.random.shuffle(sampled_files)  # 打乱顺序
print(f"采样后数据量: {len(sampled_files)} (猫: {len(cat_sample)}, 狗: {len(dog_sample)})")

# 划分训练集和验证集 (80% 训练, 20% 验证)
train_files_split, val_files = train_test_split(
    sampled_files, test_size=0.2, random_state=42, 
    stratify=[0 if f.startswith('cat') else 1 for f in sampled_files]
)

print(f"训练集大小: {len(train_files_split)}")
print(f"验证集大小: {len(val_files)}")

# 创建数据集
train_dataset = CatDogDataset(train_files_split, TRAIN_DIR, train_transform, is_test=False)
val_dataset = CatDogDataset(val_files, TRAIN_DIR, val_transform, is_test=False)
test_dataset = CatDogDataset(test_files, TEST_DIR, val_transform, is_test=True)

# 超参数设置
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows下设为0避免多进程问题

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"批次大小: {BATCH_SIZE}")
print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")
print(f"测试批次数: {len(test_loader)}")

# ==================== 6. 构建深度学习模型 ====================
print("\n" + "="*50)
print("6. 构建深度学习模型")
print("="*50)

class CatDogClassifier(nn.Module):
    """
    基于ResNet18的猫狗分类器
    使用迁移学习，加载ImageNet预训练权重
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(CatDogClassifier, self).__init__()
        
        # 加载预训练的ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 冻结前面的层（可选，用于fine-tuning）
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # 替换最后的全连接层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# 创建模型
model = CatDogClassifier(num_classes=2, pretrained=True)
model = model.to(device)

# 统计模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型架构: ResNet18 + 自定义分类头")
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ==================== 7. 定义损失函数和优化器 ====================
print("\n" + "="*50)
print("7. 定义损失函数和优化器")
print("="*50)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器 (Adam with weight decay)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

print(f"损失函数: CrossEntropyLoss")
print(f"优化器: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"学习率调度: ReduceLROnPlateau (factor=0.5, patience=3)")

# ==================== 8. 训练函数 ====================
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ==================== 9. 模型训练 ====================
print("\n" + "="*50)
print("9. 模型训练")
print("="*50)

NUM_EPOCHS = 6  # 训练轮次
best_val_acc = 0.0
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

print(f"开始训练，共 {NUM_EPOCHS} 个epoch...")
print("-" * 60)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # 验证
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # 打印结果
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"  [保存最佳模型] Val Acc: {val_acc:.2f}%")

print("\n" + "-" * 60)
print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

# ==================== 10. 训练过程可视化 ====================
print("\n" + "="*50)
print("10. 训练过程可视化")
print("="*50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
axes[1].plot(history['val_acc'], 'r-', label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Training and Validation Accuracy', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("训练曲线已保存到: output/training_curves.png")

# ==================== 11. 加载最佳模型进行测试 ====================
print("\n" + "="*50)
print("11. 测试集预测")
print("="*50)

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
model.eval()

# 预测测试集
predictions = []
image_ids = []

with torch.no_grad():
    for images, ids in tqdm(test_loader, desc='Predicting'):
        images = images.to(device)
        outputs = model(images)
        
        # 使用softmax获取概率
        probs = torch.softmax(outputs, dim=1)
        dog_probs = probs[:, 1].cpu().numpy()  # 狗的概率
        
        predictions.extend(dog_probs)
        image_ids.extend(ids.numpy())

# 创建提交文件
submission = pd.DataFrame({
    'id': image_ids,
    'label': predictions
})
submission = submission.sort_values('id')
submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
print(f"预测完成! 提交文件已保存到: output/submission.csv")

# ==================== 12. 结果分析 ====================
print("\n" + "="*50)
print("12. 结果分析")
print("="*50)

# 预测分布统计
pred_cat = sum(1 for p in predictions if p < 0.5)
pred_dog = sum(1 for p in predictions if p >= 0.5)
print(f"预测结果分布:")
print(f"  预测为猫: {pred_cat} ({100*pred_cat/len(predictions):.1f}%)")
print(f"  预测为狗: {pred_dog} ({100*pred_dog/len(predictions):.1f}%)")

# 预测概率分布可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 概率直方图
axes[0].hist(predictions, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
axes[0].set_xlabel('Dog Probability', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Prediction Probability Distribution', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 预测类别饼图
labels = ['Cat', 'Dog']
sizes = [pred_cat, pred_dog]
colors = ['#ff9999', '#66b3ff']
explode = (0.05, 0.05)
axes[1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Prediction Class Distribution', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()
print("预测分析图已保存到: output/prediction_analysis.png")

# ==================== 13. 样本可视化 ====================
print("\n" + "="*50)
print("13. 样本预测可视化")
print("="*50)

# 随机选择一些测试图片进行可视化
def visualize_predictions(test_dir, predictions_dict, num_samples=12):
    """可视化预测结果"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # 随机选择样本
    sample_ids = np.random.choice(list(predictions_dict.keys()), num_samples, replace=False)
    
    for idx, img_id in enumerate(sample_ids):
        img_path = os.path.join(test_dir, f"{img_id}.jpg")
        img = Image.open(img_path)
        
        prob = predictions_dict[img_id]
        pred_class = 'Dog' if prob >= 0.5 else 'Cat'
        confidence = prob if prob >= 0.5 else 1 - prob
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'ID: {img_id}\n{pred_class} ({confidence:.2%})', fontsize=11)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()

# 创建预测字典
pred_dict = dict(zip(image_ids, predictions))
visualize_predictions(TEST_DIR, pred_dict)
print("样本预测可视化已保存到: output/sample_predictions.png")

# ==================== 14. 实验总结 ====================
print("\n" + "="*50)
print("14. 实验总结")
print("="*50)

summary = f"""
==================== 猫狗大战实验报告 ====================

1. 数据集概况:
   - 训练集: 25000张图片 (猫狗各12500张)
   - 测试集: 12500张图片
   - 图片格式: JPEG

2. 数据预处理:
   - 图片缩放: 224x224 (适配ResNet输入)
   - 数据增强: 随机裁剪、水平翻转、旋转、颜色抖动
   - 标准化: ImageNet均值和标准差

3. 模型架构:
   - 基础模型: ResNet18 (预训练ImageNet权重)
   - 分类头: Dropout(0.5) -> Linear(512,256) -> ReLU -> Dropout(0.3) -> Linear(256,2)
   - 总参数量: {total_params:,}

4. 训练配置:
   - 优化器: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})
   - 损失函数: CrossEntropyLoss
   - 学习率调度: ReduceLROnPlateau
   - 训练轮次: {NUM_EPOCHS} epochs
   - 批次大小: {BATCH_SIZE}

5. 训练结果:
   - 最佳验证准确率: {best_val_acc:.2f}%
   - 最终训练准确率: {history['train_acc'][-1]:.2f}%
   - 最终验证准确率: {history['val_acc'][-1]:.2f}%

6. 测试预测:
   - 预测为猫: {pred_cat}张 ({100*pred_cat/len(predictions):.1f}%)
   - 预测为狗: {pred_dog}张 ({100*pred_dog/len(predictions):.1f}%)

7. 输出文件:
   - 最佳模型: output/best_model.pth
   - 预测结果: output/submission.csv
   - 训练曲线: output/training_curves.png
   - 预测分析: output/prediction_analysis.png
   - 样本可视化: output/sample_predictions.png

===========================================================
"""

print(summary)

# 保存实验报告
with open(os.path.join(OUTPUT_DIR, 'experiment_report.txt'), 'w', encoding='utf-8') as f:
    f.write(summary)
print("实验报告已保存到: output/experiment_report.txt")

print("\n实验完成!")
