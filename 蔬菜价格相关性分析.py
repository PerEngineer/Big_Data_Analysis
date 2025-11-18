import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：数据预处理
# ============================================================================

def load_and_preprocess_data(filepath):
    """
    加载和预处理蔬菜价格数据
    """
    print("=" * 80)
    print("第一部分：数据预处理")
    print("=" * 80)
    
    # 读取CSV文件
    data = pd.read_csv(filepath, encoding='gbk')
    print(f"\n原始数据形状: {data.shape}")
    print(f"原始数据列: {data.columns.tolist()}")
    
    # 去除重复记录（保留最后一条）
    data = data.drop_duplicates(['日期', '蔬菜名'], keep='last')
    data = data.drop_duplicates(['日期', '肉食禽蛋'], keep='last')
    
    # 去除含有缺失值的行
    data = data.dropna()
    
    # 提取蔬菜类
    k1 = data['蔬菜名'] != '蔬菜类'
    vegetable = data[k1]
    
    # 提取肉蛋类
    k2 = data['肉食禽蛋'] != '肉食禽蛋类'
    meat = data[k2]
    
    # 透视表：行为日期，列为蔬菜名，值为价格
    dataveg = vegetable.pivot_table(index='日期', columns='蔬菜名', values='价格', aggfunc='first')
    datam = meat.pivot_table(index='日期', columns='肉食禽蛋', values='批发价格', aggfunc='first')
    
    # 合并蔬菜和肉蛋数据
    data = pd.merge(dataveg, datam, left_index=True, right_index=True)
    
    # 用NaN代替空格以便后续的dropna操作
    for i in data.columns:
        data[i] = data[i].apply(lambda x: np.nan if isinstance(x, str) and x.isspace() else x)
    
    # 去除有缺失值的列（没有价格的蔬菜/肉蛋）
    data = data.dropna(axis=1)
    
    # 转换所有列为数值类型
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    print(f"\n清洗后数据形状: {data.shape}")
    print(f"数据时间范围: {data.index.min()} 到 {data.index.max()}")
    print(f"商品种类数: {len(data.columns)}")
    print(f"记录日期数: {len(data)}")
    
    print(f"\n清洗后数据样本:")
    print(data.head(10))
    
    return data


def create_price_trend_transactions(data):
    """
    创建价格趋势事务数据（用于发现同涨、同跌现象）
    """
    print("\n" + "=" * 80)
    print("第二部分：价格趋势事务生成")
    print("=" * 80)
    
    # 创建事务：比较相邻两个日期的价格变化
    transactions = []
    
    # 获取所有日期
    all_dates = data.index.tolist()
    
    # 对相邻的日期对进行比较
    for i in range(len(all_dates) - 1):
        date1 = all_dates[i]
        date2 = all_dates[i + 1]
        
        transaction = set()
        
        for product in data.columns:
            price1 = data.loc[date1, product]
            price2 = data.loc[date2, product]
            
            # 计算价格变化
            if price2 > price1 * 1.05:  # 上升5%以上
                transaction.add(f"{product}_上升")
            elif price2 < price1 * 0.95:  # 下降5%以上
                transaction.add(f"{product}_下降")
            else:  # 持平
                transaction.add(f"{product}_持平")
        
        if transaction:
            transactions.append(transaction)
    
    print(f"\n生成的趋势事务数: {len(transactions)}")
    print(f"\n前5个趋势事务示例:")
    for i, trans in enumerate(transactions[:5]):
        print(f"事务 {i+1}: {sorted(list(trans))[:5]}... (共{len(trans)}项)")
    
    return transactions


# ============================================================================
# 第三部分：简化的Apriori算法（仅计算2-项集）
# ============================================================================

class SimpleApriori:
    """
    简化的Apriori算法 - 仅计算1-项集和2-项集
    """
    
    def __init__(self, min_support=0.2, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.frequent_1itemsets = {}
        self.frequent_2itemsets = {}
        self.rules = []
    
    def fit(self, transactions):
        """
        运行简化的Apriori算法
        """
        self.transactions = transactions
        total_transactions = len(transactions)
        min_support_count = int(np.ceil(self.min_support * total_transactions))
        
        print(f"\n最小支持度计数: {min_support_count} (总事务数: {total_transactions})")
        
        # 第一步：找出1-项集
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # 过滤满足最小支持度的1-项集
        for item, count in item_counts.items():
            if count >= min_support_count:
                self.frequent_1itemsets[item] = count / total_transactions
        
        print(f"1-项集数: {len(self.frequent_1itemsets)}")
        
        # 第二步：找出2-项集
        pair_counts = defaultdict(int)
        for transaction in transactions:
            items = list(transaction)
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    pair = tuple(sorted([items[i], items[j]]))
                    pair_counts[pair] += 1
        
        # 过滤满足最小支持度的2-项集
        for pair, count in pair_counts.items():
            if count >= min_support_count:
                self.frequent_2itemsets[pair] = count / total_transactions
        
        print(f"2-项集数: {len(self.frequent_2itemsets)}")
        
        return self
    
    def generate_rules(self):
        """
        从2-项集生成关联规则
        """
        self.rules = []
        
        # 从2-项集生成规则
        for (item1, item2), support in self.frequent_2itemsets.items():
            # 规则1: item1 => item2
            support1 = self.frequent_1itemsets.get(item1, 0)
            if support1 > 0:
                confidence = support / support1
                if confidence >= self.min_confidence:
                    lift = confidence / self.frequent_1itemsets.get(item2, 1)
                    self.rules.append({
                        'antecedent': item1,
                        'consequent': item2,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
            
            # 规则2: item2 => item1
            support2 = self.frequent_1itemsets.get(item2, 0)
            if support2 > 0:
                confidence = support / support2
                if confidence >= self.min_confidence:
                    lift = confidence / self.frequent_1itemsets.get(item1, 1)
                    self.rules.append({
                        'antecedent': item2,
                        'consequent': item1,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
        
        # 按置信度排序
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        return self.rules
    
    def print_frequent_itemsets(self):
        """
        打印频繁项集
        """
        print("\n" + "=" * 80)
        print("第三部分：频繁项集")
        print("=" * 80)
        
        print(f"\n1-项集 (共{len(self.frequent_1itemsets)}个):")
        for i, (item, support) in enumerate(sorted(
            self.frequent_1itemsets.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]):
            print(f"  {item}: 支持度={support:.4f}")
        
        if len(self.frequent_1itemsets) > 15:
            print(f"  ... 还有 {len(self.frequent_1itemsets) - 15} 个项集")
        
        print(f"\n2-项集 (共{len(self.frequent_2itemsets)}个):")
        for i, (pair, support) in enumerate(sorted(
            self.frequent_2itemsets.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]):
            print(f"  {pair[0]}, {pair[1]}: 支持度={support:.4f}")
        
        if len(self.frequent_2itemsets) > 15:
            print(f"  ... 还有 {len(self.frequent_2itemsets) - 15} 个项集")
    
    def print_rules(self, top_n=20):
        """
        打印关联规则
        """
        print("\n" + "=" * 80)
        print("第四部分：关联规则")
        print("=" * 80)
        print(f"\n发现的关联规则总数: {len(self.rules)}")
        print(f"最小支持度: {self.min_support}, 最小置信度: {self.min_confidence}\n")
        
        if len(self.rules) == 0:
            print("未发现满足条件的关联规则")
            return
        
        print(f"前{min(top_n, len(self.rules))}个规则 (按置信度排序):\n")
        print(f"{'序号':<5} {'前件':<35} {'后件':<35} {'支持度':<10} {'置信度':<10} {'提升度':<10}")
        print("-" * 105)
        
        for i, rule in enumerate(self.rules[:top_n]):
            ant_str = rule['antecedent'][:33]
            cons_str = rule['consequent'][:33]
            
            print(f"{i+1:<5} {ant_str:<35} {cons_str:<35} "
                  f"{rule['support']:<10.4f} {rule['confidence']:<10.4f} {rule['lift']:<10.4f}")


# ============================================================================
# 第五部分：结果分析
# ============================================================================

def analyze_results(apriori, transactions):
    """
    对关联规则结果进行分析
    """
    print("\n" + "=" * 80)
    print("第五部分：结果分析")
    print("=" * 80)
    
    print(f"\n1. 数据统计:")
    print(f"   - 事务总数: {len(transactions)}")
    print(f"   - 频繁项集总数: {len(apriori.frequent_1itemsets) + len(apriori.frequent_2itemsets)}")
    print(f"   - 关联规则总数: {len(apriori.rules)}")
    
    if len(apriori.rules) > 0:
        confidences = [rule['confidence'] for rule in apriori.rules]
        supports = [rule['support'] for rule in apriori.rules]
        lifts = [rule['lift'] for rule in apriori.rules]
        
        print(f"\n2. 规则质量指标:")
        print(f"   - 置信度范围: {min(confidences):.4f} - {max(confidences):.4f}")
        print(f"   - 平均置信度: {np.mean(confidences):.4f}")
        print(f"   - 支持度范围: {min(supports):.4f} - {max(supports):.4f}")
        print(f"   - 平均支持度: {np.mean(supports):.4f}")
        print(f"   - 提升度范围: {min(lifts):.4f} - {max(lifts):.4f}")
        print(f"   - 平均提升度: {np.mean(lifts):.4f}")
        
        print(f"\n3. 高质量规则 (置信度 > 0.7):")
        high_conf_rules = [r for r in apriori.rules if r['confidence'] > 0.7]
        if high_conf_rules:
            for i, rule in enumerate(high_conf_rules[:10]):
                print(f"   {i+1}. {rule['antecedent']} => {rule['consequent']}")
                print(f"      置信度: {rule['confidence']:.4f}, 支持度: {rule['support']:.4f}, 提升度: {rule['lift']:.4f}")
        else:
            print("   未发现高置信度规则")
        
        print(f"\n4. 高提升度规则 (提升度 > 1.2):")
        high_lift_rules = [r for r in apriori.rules if r['lift'] > 1.2]
        if high_lift_rules:
            for i, rule in enumerate(sorted(high_lift_rules, key=lambda x: x['lift'], reverse=True)[:10]):
                print(f"   {i+1}. {rule['antecedent']} => {rule['consequent']}")
                print(f"      置信度: {rule['confidence']:.4f}, 支持度: {rule['support']:.4f}, 提升度: {rule['lift']:.4f}")
        else:
            print("   未发现高提升度规则")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主程序：执行完整的关联规则挖掘流程
    """
    print("\n")
    print("*" * 80)
    print("蔬菜价格相关性分析 - 关联规则挖掘实验")
    print("*" * 80)
    
    # 1. 数据预处理
    filepath = r'd:\Personal\Desktop\大数据分析\实验\实验一\01.蔬菜价格.csv'
    data = load_and_preprocess_data(filepath)
    
    # 2. 生成事务数据
    transactions = create_price_trend_transactions(data)
    
    # 3. 运行Apriori算法
    print("\n" + "=" * 80)
    print("第三部分：Apriori算法执行")
    print("=" * 80)
    
    apriori = SimpleApriori(min_support=0.15, min_confidence=0.6)
    apriori.fit(transactions)
    apriori.print_frequent_itemsets()
    
    # 4. 生成关联规则
    apriori.generate_rules()
    apriori.print_rules(top_n=30)
    
    # 5. 结果分析
    analyze_results(apriori, transactions)
    
    # 6. 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    print("""
本实验通过Apriori算法对蔬菜价格数据进行了关联规则挖掘，主要发现：

1. 数据预处理：
   - 清洗了原始数据中的缺失值和特殊字符
   - 将价格数据转换为价格趋势（上升/下降/持平）
   - 生成了事务数据集用于关联规则挖掘

2. 关联规则发现：
   - 使用Apriori算法发现了蔬菜价格之间的关联规则
   - 通过支持度、置信度和提升度等指标评估规则质量
   - 识别出具有同涨、同跌现象的蔬菜对

3. 业务意义：
   - 高置信度规则表示蔬菜价格变化的强关联性
   - 高提升度规则表示规则的实际价值
   - 这些规则可用于价格预测和市场分析

4. 后续改进方向：
   - 调整最小支持度和置信度阈值以获得更多/更少规则
   - 尝试其他算法如Eclat或FP-Growth
   - 结合时间序列分析进行更深入的研究
    """)
    
    print("=" * 80)
    print("实验完成")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
