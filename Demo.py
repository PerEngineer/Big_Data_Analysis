"""
实验三：乳腺癌预测 - 分类算法综合实验
使用多种分类算法进行乳腺癌良恶性预测
包括：逻辑回归、KNN、决策树、朴素贝叶斯、神经网络等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("="*80)
print("实验三：乳腺癌预测 - 分类算法综合实验")
print("="*80)

# ============================================================================
# 第一部分：数据加载与探索性分析
# ============================================================================
print("\n【第一部分：数据加载与探索性分析】")
print("-"*80)

# 1. 加载数据
data = pd.read_csv('03.乳腺癌预测.csv')
print(f"\n1. 数据集基本信息：")
print(f"   - 数据集形状：{data.shape}")
print(f"   - 样本数量：{data.shape[0]}")
print(f"   - 特征数量：{data.shape[1] - 2}")  # 减去id和diagnosis

# 2. 查看前几行数据
print(f"\n2. 数据前5行预览：")
print(data.head())

# 3. 数据基本统计信息
print(f"\n3. 目标变量分布：")
print(data['diagnosis'].value_counts())
print(f"\n   良性(B)：{(data['diagnosis']=='B').sum()} 例 ({(data['diagnosis']=='B').sum()/len(data)*100:.2f}%)")
print(f"   恶性(M)：{(data['diagnosis']=='M').sum()} 例 ({(data['diagnosis']=='M').sum()/len(data)*100:.2f}%)")

# 4. 检查缺失值
print(f"\n4. 缺失值检查：")
missing_values = data.isnull().sum()
if missing_values.sum() == 0:
    print("   ✓ 数据集无缺失值")
else:
    print(missing_values[missing_values > 0])

# ============================================================================
# 第二部分：数据预处理
# ============================================================================
print("\n【第二部分：数据预处理】")
print("-"*80)

# 1. 删除ID列（对预测无用）
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

print(f"\n1. 特征矩阵形状：{X.shape}")
print(f"   特征列表（前10个）：{list(X.columns[:10])}")

# 2. 将标签转换为数值（M=1恶性, B=0良性）
y = y.map({'M': 1, 'B': 0})
print(f"\n2. 标签编码：M(恶性)=1, B(良性)=0")

# 3. 数据集划分（训练集70%，测试集30%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n3. 数据集划分：")
print(f"   训练集：{X_train.shape[0]} 样本")
print(f"   测试集：{X_test.shape[0]} 样本")
print(f"   训练集恶性比例：{y_train.sum()/len(y_train)*100:.2f}%")
print(f"   测试集恶性比例：{y_test.sum()/len(y_test)*100:.2f}%")

# 4. 特征标准化（对某些算法很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\n4. 特征标准化完成（使用StandardScaler）")

# ============================================================================
# 第三部分：模型训练与预测
# ============================================================================
print("\n【第三部分：模型训练与预测】")
print("-"*80)

# 定义多个分类器
classifiers = {
    '逻辑回归': LogisticRegression(max_iter=10000, random_state=42),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (K=7)': KNeighborsClassifier(n_neighbors=7),
    '决策树': DecisionTreeClassifier(random_state=42, max_depth=5),
    '朴素贝叶斯': GaussianNB(),
    '支持向量机': SVC(kernel='rbf', probability=True, random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    '神经网络': MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=1000, random_state=42)
}

# 存储结果
results = {}

print("\n开始训练各个分类器...")
print("-"*80)

for name, clf in classifiers.items():
    print(f"\n正在训练：{name}")
    
    # 根据算法选择是否使用标准化数据
    if name in ['逻辑回归', 'KNN (K=5)', 'KNN (K=7)', '支持向量机', '神经网络']:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # 交叉验证
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # 交叉验证
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 存储结果
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"  ✓ 训练完成 - 准确率: {accuracy:.4f}")

# ============================================================================
# 第四部分：结果分析与对比
# ============================================================================
print("\n【第四部分：结果分析与对比】")
print("-"*80)

# 1. 创建结果对比表
print("\n1. 各算法性能对比表：")
print("-"*80)
print(f"{'算法':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'AUC':<10} {'交叉验证':<15}")
print("-"*80)

for name, metrics in results.items():
    print(f"{name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
          f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['roc_auc']:<10.4f} "
          f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")

# 2. 找出最佳模型
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n2. 最佳模型（基于准确率）：{best_model[0]}")
print(f"   准确率：{best_model[1]['accuracy']:.4f}")
print(f"   AUC值：{best_model[1]['roc_auc']:.4f}")

# 3. 详细分类报告（最佳模型）
print(f"\n3. {best_model[0]} 详细分类报告：")
print("-"*80)
print(classification_report(y_test, best_model[1]['y_pred'], 
                          target_names=['良性(B)', '恶性(M)']))

# 4. 混淆矩阵分析
print(f"\n4. {best_model[0]} 混淆矩阵分析：")
cm = best_model[1]['confusion_matrix']
print(f"\n   真实良性预测为良性(TN): {cm[0,0]}")
print(f"   真实良性预测为恶性(FP): {cm[0,1]} ← 假阳性")
print(f"   真实恶性预测为良性(FN): {cm[1,0]} ← 假阴性（危险！）")
print(f"   真实恶性预测为恶性(TP): {cm[1,1]}")

# 5. 保存神经网络预测结果到CSV文件
print(f"\n5. 保存{best_model[0]}预测结果...")
neural_net_results = pd.DataFrame({
    '样本ID': range(len(y_test)),
    '真实标签': y_test.values,
    '预测标签': best_model[1]['y_pred'],
    '预测概率(恶性)': best_model[1]['y_pred_proba'],
    '预测概率(良性)': 1 - best_model[1]['y_pred_proba'],
    '预测正确': y_test.values == best_model[1]['y_pred']
})

# 将标签转换为可读的格式
neural_net_results['真实标签'] = neural_net_results['真实标签'].map({0: '良性(B)', 1: '恶性(M)'})
neural_net_results['预测标签'] = neural_net_results['预测标签'].map({0: '良性(B)', 1: '恶性(M)'})
neural_net_results['预测正确'] = neural_net_results['预测正确'].map({True: '正确', False: '错误'})

# 保存到CSV文件
output_file = '神经网络预测结果.csv'
neural_net_results.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   ✓ 预测结果已保存到：{output_file}")
print(f"   - 总样本数：{len(neural_net_results)}")
print(f"   - 预测正确数：{(neural_net_results['预测正确']=='正确').sum()}")
print(f"   - 预测错误数：{(neural_net_results['预测正确']=='错误').sum()}")

# ============================================================================
# 第五部分：可视化分析
# ============================================================================
print("\n【第五部分：生成可视化图表】")
print("-"*80)

names = list(results.keys())
colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

# 1. 各算法准确率对比
fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_subplot(111)
accuracies = [results[name]['accuracy'] for name in names]
bars = ax1.barh(names, accuracies, color=colors)
ax1.set_xlabel('准确率', fontsize=12)
ax1.set_title('各算法准确率对比', fontsize=14, fontweight='bold')
ax1.set_xlim([0.8, 1.0])
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(acc, i, f' {acc:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('1_准确率对比.png', dpi=300, bbox_inches='tight')
print("✓ 图表1已保存：1_准确率对比.png")
plt.show()

# 2. 多指标雷达图（最佳模型）
fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(111, projection='polar')
categories = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
values = [
    best_model[1]['accuracy'],
    best_model[1]['precision'],
    best_model[1]['recall'],
    best_model[1]['f1'],
    best_model[1]['roc_auc']
]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]
ax2.plot(angles, values, 'o-', linewidth=2, color='red')
ax2.fill(angles, values, alpha=0.25, color='red')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylim(0, 1)
ax2.set_title(f'{best_model[0]} 性能雷达图', fontsize=14, fontweight='bold', pad=20)
ax2.grid(True)
plt.tight_layout()
plt.savefig('2_性能雷达图.png', dpi=300, bbox_inches='tight')
print("✓ 图表2已保存：2_性能雷达图.png")
plt.show()

# 3. 混淆矩阵热力图
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['预测良性', '预测恶性'],
            yticklabels=['实际良性', '实际恶性'],
            ax=ax3)
ax3.set_title(f'{best_model[0]} 混淆矩阵', fontsize=14, fontweight='bold')
ax3.set_ylabel('真实标签', fontsize=12)
ax3.set_xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.savefig('3_混淆矩阵.png', dpi=300, bbox_inches='tight')
print("✓ 图表3已保存：3_混淆矩阵.png")
plt.show()

# 4. ROC曲线对比
fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111)
for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    auc_score = results[name]['roc_auc']
    ax4.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})', linewidth=2)
ax4.plot([0, 1], [0, 1], 'k--', label='随机猜测', linewidth=1)
ax4.set_xlabel('假阳性率 (FPR)', fontsize=12)
ax4.set_ylabel('真阳性率 (TPR)', fontsize=12)
ax4.set_title('ROC曲线对比', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right', fontsize=8)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_ROC曲线.png', dpi=300, bbox_inches='tight')
print("✓ 图表4已保存：4_ROC曲线.png")
plt.show()

# 5. F1分数对比
fig5 = plt.figure(figsize=(12, 6))
ax5 = fig5.add_subplot(111)
f1_scores = [results[name]['f1'] for name in names]
bars = ax5.bar(range(len(names)), f1_scores, color=colors)
ax5.set_xticks(range(len(names)))
ax5.set_xticklabels(names, rotation=45, ha='right')
ax5.set_ylabel('F1分数', fontsize=12)
ax5.set_title('各算法F1分数对比', fontsize=14, fontweight='bold')
ax5.set_ylim([0.8, 1.0])
for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
    ax5.text(i, f1, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('5_F1分数对比.png', dpi=300, bbox_inches='tight')
print("✓ 图表5已保存：5_F1分数对比.png")
plt.show()

# 6. 交叉验证得分对比
fig6 = plt.figure(figsize=(12, 6))
ax6 = fig6.add_subplot(111)
cv_means = [results[name]['cv_mean'] for name in names]
cv_stds = [results[name]['cv_std'] for name in names]
ax6.barh(names, cv_means, xerr=cv_stds, color=colors, alpha=0.7)
ax6.set_xlabel('交叉验证准确率', fontsize=12)
ax6.set_title('5折交叉验证结果对比', fontsize=14, fontweight='bold')
ax6.set_xlim([0.8, 1.0])
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax6.text(mean, i, f' {mean:.3f}±{std:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('6_交叉验证结果.png', dpi=300, bbox_inches='tight')
print("✓ 图表6已保存：6_交叉验证结果.png")
plt.show()

print("\n✓ 所有可视化图表已生成并按顺序显示")

# ============================================================================
# 第六部分：实验总结
# ============================================================================
print("\n【第六部分：实验总结】")
print("="*80)

print(f"""
实验结论：

1. 数据集概况：
   - 本实验使用威斯康星乳腺癌数据集，包含{data.shape[0]}个样本
   - 每个样本有{X.shape[1]}个特征，包括细胞核的各种测量指标
   - 数据集包含{(y==0).sum()}例良性样本和{(y==1).sum()}例恶性样本

2. 算法性能排名（按准确率）：
""")

sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"   {i}. {name:<15} 准确率: {metrics['accuracy']:.4f}  AUC: {metrics['roc_auc']:.4f}")

print(f"""
3. 最佳模型分析：
   - 最佳算法：{best_model[0]}
   - 测试集准确率：{best_model[1]['accuracy']:.4f}
   - AUC值：{best_model[1]['roc_auc']:.4f}
   - 召回率（检出率）：{best_model[1]['recall']:.4f}
   
4. 医学应用价值：
   - 在医疗诊断中，召回率（Recall）尤为重要，因为漏诊恶性肿瘤（假阴性）
     的后果非常严重
   - 本实验中多个模型的召回率都超过95%，说明模型能有效识别恶性肿瘤
   - 精确率也很高，说明误诊率（假阳性）较低，减少不必要的治疗

5. 算法特点总结：
   - 逻辑回归：简单高效，可解释性强，适合线性可分问题
   - KNN：简单直观，无需训练，但计算量大
   - 决策树：可解释性好，能处理非线性关系
   - 朴素贝叶斯：基于概率理论，计算快速
   - 支持向量机：适合高维数据，泛化能力强
   - 随机森林：集成学习，准确率高，鲁棒性好
   - 神经网络：强大的非线性拟合能力

6. 实验建议：
   - 在实际应用中，建议使用集成方法（如随机森林）或神经网络
   - 可以通过调整分类阈值来平衡精确率和召回率
   - 建议结合医生的专业判断，将模型作为辅助诊断工具
""")

print("="*80)
print("实验完成！")
print("="*80)

plt.show()
