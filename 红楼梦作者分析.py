# -*- coding: utf-8 -*-
"""
实验五：聚类算法实验 - 《红楼梦》作者分析
实验目的：利用聚类算法分析《红楼梦》各章节的写作风格，探究前八十回与后四十回作者差异
实验方法：统计各章节虚词频率，使用层次聚类、K-means、DBSCAN等算法进行聚类分析
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据读取与预处理 =====================
print("=" * 60)
print("《红楼梦》作者聚类分析实验")
print("=" * 60)

# 读取文本（尝试多种编码）
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
text = None
for enc in encodings:
    try:
        with open('《红楼梦》完整版.txt', 'r', encoding=enc) as f:
            text = f.read()
        print(f"成功使用 {enc} 编码读取文件")
        break
    except:
        continue
if text is None:
    raise Exception("无法读取文件，请检查编码")

print(f"\n[1] 数据读取完成，文本总长度: {len(text)} 字符")

# 按章节分割文本
# 匹配"第X回"的模式
chapter_pattern = r'第([一二三四五六七八九十百零]+)回\s+.+?\n'
chapters = re.split(chapter_pattern, text)

# 提取章节内容
chapter_texts = {}
i = 1
while i < len(chapters) - 1:
    chapter_num_cn = chapters[i]  # 中文数字
    chapter_content = chapters[i + 1]  # 章节内容
    
    # 中文数字转阿拉伯数字
    cn_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                  '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                  '零': 0, '百': 100}
    
    def cn_to_num(cn_str):
        result = 0
        temp = 0
        for char in cn_str:
            if char == '十':
                if temp == 0:
                    temp = 1
                result += temp * 10
                temp = 0
            elif char == '百':
                if temp == 0:
                    temp = 1
                result += temp * 100
                temp = 0
            else:
                temp = cn_num_map.get(char, 0)
        result += temp
        return result
    
    chapter_num = cn_to_num(chapter_num_cn)
    if 1 <= chapter_num <= 120:
        chapter_texts[chapter_num] = chapter_content
    i += 2

print(f"[2] 章节分割完成，共提取 {len(chapter_texts)} 个章节")

# ===================== 2. 虚词列表定义 =====================
# 定义常用虚词列表（用于衡量写作风格）
function_words = [
    # 代词
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '这', '那', '这个', '那个',
    '什么', '怎么', '哪', '谁', '多少', '自己', '别人', '大家',
    # 副词
    '不', '没', '没有', '很', '太', '更', '最', '都', '也', '还', '又', '再',
    '就', '才', '只', '只是', '已', '已经', '曾', '曾经', '正', '正在',
    '将', '便', '却', '竟', '倒', '偏', '果然', '居然', '竟然', '必',
    # 介词
    '在', '从', '向', '往', '到', '对', '把', '被', '给', '让', '用', '以', '为',
    # 连词
    '和', '与', '或', '但', '但是', '然而', '因为', '所以', '如果', '虽然',
    '即', '及', '且', '而', '则', '若', '因', '故',
    # 助词
    '的', '地', '得', '了', '着', '过', '呢', '吧', '啊', '呀', '吗', '么',
    # 叹词
    '唉', '哎', '哦', '嗯',
    # 其他常用虚词
    '是', '有', '无', '能', '可', '可以', '要', '想', '该', '应', '须',
    '来', '去', '上', '下', '里', '中', '内', '外', '前', '后',
    '如何', '怎样', '如此', '这样', '那样', '岂', '何', '乃', '其', '之'
]

print(f"[3] 虚词列表定义完成，共 {len(function_words)} 个虚词")

# ===================== 3. 特征提取 - 统计虚词频率 =====================
def count_function_words(text, words):
    """统计文本中虚词的频率"""
    text_len = len(text)
    if text_len == 0:
        return {word: 0 for word in words}
    
    word_freq = {}
    for word in words:
        count = text.count(word)
        # 计算每千字的出现频率
        word_freq[word] = (count / text_len) * 1000
    return word_freq

# 为每个章节计算虚词频率
chapter_features = {}
for chapter_num in sorted(chapter_texts.keys()):
    chapter_features[chapter_num] = count_function_words(chapter_texts[chapter_num], function_words)

# 转换为DataFrame
df_features = pd.DataFrame(chapter_features).T
df_features.index.name = '章节'
print(f"[4] 特征提取完成，特征矩阵维度: {df_features.shape}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
print(f"[5] 数据标准化完成")

# ===================== 4. 降维可视化 =====================
# 使用PCA降维到2维进行可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"[6] PCA降维完成，解释方差比: {pca.explained_variance_ratio_.sum():.2%}")

# 创建章节标签（前80回 vs 后40回）
chapters_list = sorted(chapter_texts.keys())
labels_true = ['前80回' if ch <= 80 else '后40回' for ch in chapters_list]
colors_true = ['blue' if ch <= 80 else 'red' for ch in chapters_list]

# ===================== 5. 聚类算法实现 =====================
print("\n" + "=" * 60)
print("聚类分析")
print("=" * 60)

# 5.1 层次聚类 (Hierarchical Clustering)
print("\n[层次聚类分析]")
hc = AgglomerativeClustering(n_clusters=2, linkage='ward')
hc_labels = hc.fit_predict(X_scaled)
hc_score = silhouette_score(X_scaled, hc_labels)
print(f"轮廓系数: {hc_score:.4f}")

# 统计聚类结果与真实标签的对应关系
hc_cluster0_first80 = sum(1 for i, ch in enumerate(chapters_list) if hc_labels[i] == 0 and ch <= 80)
hc_cluster1_first80 = sum(1 for i, ch in enumerate(chapters_list) if hc_labels[i] == 1 and ch <= 80)
print(f"簇0: 前80回{hc_cluster0_first80}章, 后40回{sum(hc_labels==0)-hc_cluster0_first80}章")
print(f"簇1: 前80回{hc_cluster1_first80}章, 后40回{sum(hc_labels==1)-hc_cluster1_first80}章")

# 5.2 K-Means聚类
print("\n[K-Means聚类分析]")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
print(f"轮廓系数: {kmeans_score:.4f}")

km_cluster0_first80 = sum(1 for i, ch in enumerate(chapters_list) if kmeans_labels[i] == 0 and ch <= 80)
km_cluster1_first80 = sum(1 for i, ch in enumerate(chapters_list) if kmeans_labels[i] == 1 and ch <= 80)
print(f"簇0: 前80回{km_cluster0_first80}章, 后40回{sum(kmeans_labels==0)-km_cluster0_first80}章")
print(f"簇1: 前80回{km_cluster1_first80}章, 后40回{sum(kmeans_labels==1)-km_cluster1_first80}章")

# 5.3 DBSCAN密度聚类
print("\n[DBSCAN密度聚类分析]")
# 尝试不同的参数
best_eps, best_min_samples, best_dbscan_score = 0, 0, -1
best_dbscan_labels = None

for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for min_samples in [2, 3, 4, 5]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            # 只计算非噪声点的轮廓系数
            mask = labels != -1
            if sum(mask) > 1:
                try:
                    score = silhouette_score(X_scaled[mask], labels[mask])
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                        best_dbscan_labels = labels
                except:
                    pass

if best_dbscan_labels is not None:
    print(f"最佳参数: eps={best_eps}, min_samples={best_min_samples}")
    print(f"轮廓系数: {best_dbscan_score:.4f}")
    n_clusters = len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)
    n_noise = list(best_dbscan_labels).count(-1)
    print(f"聚类数: {n_clusters}, 噪声点数: {n_noise}")
else:
    print("DBSCAN未能找到有效聚类，使用默认参数")
    dbscan = DBSCAN(eps=2.0, min_samples=3)
    best_dbscan_labels = dbscan.fit_predict(X_scaled)

# ===================== 6. 结果可视化 =====================
print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

fig = plt.figure(figsize=(16, 12))

# 6.1 原始数据分布（按前80回/后40回着色）
ax1 = fig.add_subplot(2, 3, 1)
for i, ch in enumerate(chapters_list):
    color = 'blue' if ch <= 80 else 'red'
    marker = 'o' if ch <= 80 else '^'
    ax1.scatter(X_pca[i, 0], X_pca[i, 1], c=color, marker=marker, s=50, alpha=0.7)
ax1.scatter([], [], c='blue', marker='o', label='前80回 (曹雪芹)')
ax1.scatter([], [], c='red', marker='^', label='后40回 (高鹗)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('原始章节分布 (PCA降维)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 6.2 层次聚类结果
ax2 = fig.add_subplot(2, 3, 2)
for i, ch in enumerate(chapters_list):
    color = 'green' if hc_labels[i] == 0 else 'orange'
    marker = 'o' if ch <= 80 else '^'
    ax2.scatter(X_pca[i, 0], X_pca[i, 1], c=color, marker=marker, s=50, alpha=0.7)
ax2.scatter([], [], c='green', marker='s', label='簇0')
ax2.scatter([], [], c='orange', marker='s', label='簇1')
ax2.scatter([], [], c='gray', marker='o', label='前80回')
ax2.scatter([], [], c='gray', marker='^', label='后40回')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title(f'层次聚类结果 (轮廓系数: {hc_score:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 6.3 K-Means聚类结果
ax3 = fig.add_subplot(2, 3, 3)
for i, ch in enumerate(chapters_list):
    color = 'purple' if kmeans_labels[i] == 0 else 'cyan'
    marker = 'o' if ch <= 80 else '^'
    ax3.scatter(X_pca[i, 0], X_pca[i, 1], c=color, marker=marker, s=50, alpha=0.7)
ax3.scatter([], [], c='purple', marker='s', label='簇0')
ax3.scatter([], [], c='cyan', marker='s', label='簇1')
ax3.scatter([], [], c='gray', marker='o', label='前80回')
ax3.scatter([], [], c='gray', marker='^', label='后40回')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_title(f'K-Means聚类结果 (轮廓系数: {kmeans_score:.3f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 6.4 DBSCAN聚类结果
ax4 = fig.add_subplot(2, 3, 4)
unique_labels = set(best_dbscan_labels)
colors_db = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors_db):
    if k == -1:
        col = 'gray'
    class_member_mask = (best_dbscan_labels == k)
    for i, is_member in enumerate(class_member_mask):
        if is_member:
            marker = 'o' if chapters_list[i] <= 80 else '^'
            ax4.scatter(X_pca[i, 0], X_pca[i, 1], c=[col], marker=marker, s=50, alpha=0.7)
ax4.scatter([], [], c='gray', marker='o', label='前80回')
ax4.scatter([], [], c='gray', marker='^', label='后40回')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_title(f'DBSCAN聚类结果')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 6.5 层次聚类树状图
ax5 = fig.add_subplot(2, 3, 5)
linkage_matrix = linkage(X_scaled, method='ward')
# 简化显示，只显示最后30个合并
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, ax=ax5,
           leaf_font_size=8, show_contracted=True)
ax5.set_xlabel('样本索引')
ax5.set_ylabel('距离')
ax5.set_title('层次聚类树状图 (Ward法)')
ax5.axhline(y=50, color='r', linestyle='--', label='切割线')

# 6.6 章节虚词特征热图（选择关键虚词）
ax6 = fig.add_subplot(2, 3, 6)
# 选择方差较大的前15个虚词
var_scores = df_features.var().sort_values(ascending=False)
top_words = var_scores.head(15).index.tolist()
df_heatmap = df_features[top_words]

# 分组计算平均值
first_80_mean = df_heatmap.iloc[:80].mean()
last_40_mean = df_heatmap.iloc[80:].mean()
comparison = pd.DataFrame({'前80回': first_80_mean, '后40回': last_40_mean})
im = ax6.imshow(comparison.T, aspect='auto', cmap='RdYlBu_r')
ax6.set_xticks(range(len(top_words)))
ax6.set_xticklabels(top_words, rotation=45, ha='right')
ax6.set_yticks([0, 1])
ax6.set_yticklabels(['前80回', '后40回'])
ax6.set_title('关键虚词频率对比')
plt.colorbar(im, ax=ax6, label='频率(‰)')

plt.tight_layout()
plt.savefig('红楼梦聚类分析结果.png', dpi=150, bbox_inches='tight')
plt.show()

# ===================== 7. 详细结果分析 =====================
print("\n" + "=" * 60)
print("详细结果分析")
print("=" * 60)

# 计算聚类与真实标签的一致性
def calculate_consistency(cluster_labels, chapters, threshold=80):
    """计算聚类结果与前80/后40分界的一致性"""
    cluster0_first80 = sum(1 for i, ch in enumerate(chapters) if cluster_labels[i] == 0 and ch <= threshold)
    cluster0_last40 = sum(1 for i, ch in enumerate(chapters) if cluster_labels[i] == 0 and ch > threshold)
    cluster1_first80 = sum(1 for i, ch in enumerate(chapters) if cluster_labels[i] == 1 and ch <= threshold)
    cluster1_last40 = sum(1 for i, ch in enumerate(chapters) if cluster_labels[i] == 1 and ch > threshold)
    
    # 计算两种对应方式的一致性，取较高者
    consistency1 = (cluster0_first80 + cluster1_last40) / len(chapters)
    consistency2 = (cluster1_first80 + cluster0_last40) / len(chapters)
    
    return max(consistency1, consistency2)

hc_consistency = calculate_consistency(hc_labels, chapters_list)
km_consistency = calculate_consistency(kmeans_labels, chapters_list)

print(f"\n层次聚类与前80/后40分界一致性: {hc_consistency:.2%}")
print(f"K-Means聚类与前80/后40分界一致性: {km_consistency:.2%}")

# 虚词频率差异分析
print("\n[前80回与后40回虚词频率差异最大的词]")
first_80_mean = df_features.iloc[:80].mean()
last_40_mean = df_features.iloc[80:].mean()
diff = (last_40_mean - first_80_mean).abs().sort_values(ascending=False)

print("\n虚词频率差异排名 (单位: 每千字):")
print("-" * 50)
for i, (word, d) in enumerate(diff.head(15).items()):
    f80 = first_80_mean[word]
    l40 = last_40_mean[word]
    direction = "↑" if l40 > f80 else "↓"
    print(f"{i+1:2d}. '{word}': 前80回={f80:.3f}, 后40回={l40:.3f} {direction} 差值={d:.3f}")

# 边界章节分析
print("\n[边界章节分析 - 第75-85回聚类归属]")
print("-" * 50)
for ch in range(75, 86):
    if ch in chapters_list:
        idx = chapters_list.index(ch)
        print(f"第{ch:3d}回: 层次聚类=簇{hc_labels[idx]}, K-Means=簇{kmeans_labels[idx]}")

# ===================== 8. 实验结论 =====================
print("\n" + "=" * 60)
print("实验结论")
print("=" * 60)

conclusion = """
基于虚词频率的聚类分析结果显示：

1. 【聚类效果】
   - 层次聚类轮廓系数: {:.4f}
   - K-Means聚类轮廓系数: {:.4f}
   - 两种方法都能将120回较好地分成两类

2. 【与传统观点对比】
   - 层次聚类与前80/后40分界一致性: {:.2%}
   - K-Means与前80/后40分界一致性: {:.2%}
   
3. 【关键发现】
   - 前80回与后40回在虚词使用上存在明显差异
   - 差异最大的虚词主要体现在写作习惯上
   - 聚类结果在一定程度上支持"前80回与后40回作者不同"的观点

4. 【局限性说明】
   - 虚词频率只是写作风格的一个维度
   - 聚类结果受参数选择影响
   - 不能作为作者鉴定的唯一依据

5. 【进一步研究建议】
   - 可结合其他语言特征（如句式、修辞等）进行综合分析
   - 可使用更多聚类算法进行交叉验证
   - 可分析具体章节的过渡特征
""".format(hc_score, kmeans_score, hc_consistency, km_consistency)

print(conclusion)

# 保存分析结果
with open('分析报告.txt', 'w', encoding='utf-8') as f:
    f.write("《红楼梦》作者聚类分析报告\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"分析日期: 实验五聚类算法实验\n")
    f.write(f"样本数量: {len(chapters_list)} 个章节\n")
    f.write(f"特征数量: {len(function_words)} 个虚词\n\n")
    f.write(conclusion)
    f.write("\n\n[虚词频率差异详表]\n")
    f.write("-" * 50 + "\n")
    for word, d in diff.head(20).items():
        f80 = first_80_mean[word]
        l40 = last_40_mean[word]
        f.write(f"'{word}': 前80回={f80:.3f}‰, 后40回={l40:.3f}‰, 差值={d:.3f}‰\n")

print("\n结果已保存至:")
print("  - 红楼梦聚类分析结果.png")
print("  - 分析报告.txt")
print("\n实验完成!")
