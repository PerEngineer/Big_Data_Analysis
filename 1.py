import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（尝试多个字体选项）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'STSong']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 验证字体设置
from matplotlib.font_manager import FontProperties
print(f"当前可用中文字体: {[f.name for f in matplotlib.font_manager.fontManager.ttflist if 'Microsoft' in f.name or 'SimHei' in f.name or 'KaiTi' in f.name][:5]}")

print("=" * 80)
print("美国新冠肺炎总确诊量预测实验")
print("=" * 80)

# 1. 数据加载与预处理
print("\n【步骤1：数据加载与预处理】")
print("-" * 80)

# 读取数据
df = pd.read_csv('02.美国新冠肺炎疫情历史总数据9.9.csv', encoding='gbk')
print(f"原始数据形状: {df.shape}")
print(f"数据列名: {df.columns.tolist()}")
print(f"\n前5行数据:\n{df.head()}")

# 数据清洗
df = df.dropna()  # 删除空行
print(f"\n删除空行后数据形状: {df.shape}")

# 将日期转换为数值型特征（天数）
df['天数'] = range(1, len(df) + 1)

# 显示数据统计信息
print(f"\n数据统计信息:")
print(df.describe())

# 2. 特征工程（时间序列特征构建）
print("\n【步骤2：特征工程】")
print("-" * 80)
print("构建时间序列特征，避免数据泄露...")

# 创建滞后特征（使用历史数据预测未来）
def create_time_series_features(data, lags=[1, 3, 7]):
    """创建时间序列特征 (已修正数据泄露问题)"""
    df_features = data.copy()
    
    # 1. 滞后特征（前N天的数据） - 这部分本身是正确的
    for lag in lags:
        df_features[f'总确诊_lag{lag}'] = df_features['总确诊'].shift(lag)
        df_features[f'新增_lag{lag}'] = df_features['新增'].shift(lag)
        df_features[f'死亡_lag{lag}'] = df_features['死亡'].shift(lag)
    
    # 2. 移动平均特征（平滑趋势）- 修正：使用 .shift(1)
    df_features['总确诊_ma3'] = df_features['总确诊'].rolling(window=3, min_periods=1).mean().shift(1)
    df_features['总确诊_ma7'] = df_features['总确诊'].rolling(window=7, min_periods=1).mean().shift(1)
    df_features['新增_ma3'] = df_features['新增'].rolling(window=3, min_periods=1).mean().shift(1)
    df_features['新增_ma7'] = df_features['新增'].rolling(window=7, min_periods=1).mean().shift(1)
    
    # 3. 增长率特征 - 修正：使用 .shift(1)
    df_features['总确诊_增长率'] = df_features['总确诊'].pct_change().shift(1).fillna(0)
    df_features['新增_增长率'] = df_features['新增'].pct_change().shift(1).fillna(0)
    
    # 4. 差分特征（变化量）- 修正：使用 .shift(1)
    df_features['总确诊_diff1'] = df_features['总确诊'].diff().shift(1).fillna(0)
    df_features['总确诊_diff7'] = df_features['总确诊'].diff(7).shift(1).fillna(0)
    
    return df_features

# 应用特征工程
df_engineered = create_time_series_features(df)

# 处理NaN和无穷大值 (shift操作会产生NaN，需要填充)
df_engineered = df_engineered.fillna(0)
df_engineered = df_engineered.replace([np.inf, -np.inf], 0)

print(f"原始特征数: 5")
print(f"工程化后特征数: {df_engineered.shape[1]}")
print(f"新增特征: 滞后特征、移动平均、增长率、差分等")

# 3. 数据集划分
print("\n【步骤3：数据集划分】")
print("-" * 80)

train_size = 191
test_size = 9

train_data = df_engineered[:train_size]
test_data = df_engineered[train_size:train_size+test_size]

print(f"训练集大小: {len(train_data)} 天 (2020.1.28-2020.8.31)")
print(f"测试集大小: {len(test_data)} 天 (2020.9.1-2020.9.9)")
print(f"训练集日期范围: {df.iloc[0, 0]} 至 {df.iloc[train_size-1, 0]}")
print(f"测试集日期范围: {df.iloc[train_size, 0]} 至 {df.iloc[train_size+test_size-1, 0]}")

# 选择特征（排除目标变量和辅助列）- 这部分本身是正确的
exclude_cols = ['时间', '总确诊', '治愈', '新增', '死亡']
feature_cols = [col for col in df_engineered.columns if col not in exclude_cols]
target_col = '总确诊'

X_train = train_data[feature_cols].values
y_train = train_data[target_col].values

X_test = test_data[feature_cols].values
y_test = test_data[target_col].values

print(f"\n" + "="*80)
print("【变量分析说明】")
print("="*80)
print(f"\n[因变量（目标变量Y）]: {target_col}")
# ... (省略与之前相同的打印信息) ...
print(f"\n最终使用的特征变量:")
print(f"  {feature_cols}")
print()
print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
print("="*80)

# 4. 特征标准化
print("\n【步骤4：特征标准化】")
print("-" * 80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[完成] 特征已标准化（均值=0，标准差=1）")
print(f"标准化后训练集形状: {X_train_scaled.shape}")

# 5. 超参数调优
print("\n【步骤5：超参数调优（网格搜索）】")
print("=" * 80)
print("正在为每个模型寻找最佳参数...")

param_grids = {
    'Lasso': {
        'model': Lasso(random_state=42, max_iter=10000),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=42, max_iter=10000),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    },
    'SVR': {
        'model': SVR(kernel='rbf'),
        'params': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
    }
}

# 修正：使用 TimeSeriesSplit 进行交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 执行网格搜索
best_models = {}
for name, config in param_grids.items():
    print(f"\n调优 {name}...")
    grid_search = GridSearchCV(
        config['model'], 
        config['params'],
        cv=tscv,  # <--- 应用修正后的交叉验证策略
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"  最佳参数: {grid_search.best_params_}")
    print(f"  最佳CV分数: {-grid_search.best_score_:.2f}")

print("\n[完成] 超参数调优完成！")

# 6. 模型训练与预测（使用优化后的参数）
print("\n【步骤6：模型训练与预测（超参数优化版）】")
print("=" * 80)

models = {
    '岭回归(Ridge)': Ridge(alpha=1.0),
    'Lasso回归(调优)': best_models['Lasso'],
    'ElasticNet回归(调优)': best_models['ElasticNet'],
    'SVR回归(RBF核)': best_models['SVR']
}

results = {}

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"模型: {model_name}")
    print('-' * 80)
    
    # 直接使用标准化数据训练
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    results[model_name] = {
        'y_test_pred': y_test_pred, 'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae, 'train_r2': train_r2,
        'test_r2': test_r2, 'test_mape': test_mape
    }
    
    print(f"训练集性能:")
    print(f"  RMSE (均方根误差): {train_rmse:,.2f}")
    print(f"  MAE (平均绝对误差): {train_mae:,.2f}")
    print(f"  R² (决定系数): {train_r2:.4f}")
    
    print(f"\n测试集性能:")
    print(f"  RMSE (均方根误差): {test_rmse:,.2f}")
    print(f"  MAE (平均绝对误差): {test_mae:,.2f}")
    print(f"  R² (决定系数): {test_r2:.4f}")
    print(f"  MAPE (平均相对误差): {test_mape:.2f}%")
    
    print(f"\n测试集预测结果对比:")
    print(f"{'日期':<15} {'实际值':>12} {'预测值':>12} {'误差':>12} {'相对误差%':>12}")
    print('-' * 65)
    for i in range(len(test_data)):
        date = test_data.iloc[i, 0]
        actual = y_test[i]
        pred = y_test_pred[i]
        error = pred - actual
        rel_error = (error / actual) * 100
        print(f"{date:<15} {actual:>12,} {pred:>12,.0f} {error:>12,.0f} {rel_error:>11.2f}%")

# 7. 模型对比总结
print("\n\n【步骤7：模型性能对比总结】")
print("=" * 80)
print(f"{'模型名称':<25} {'测试RMSE':>15} {'测试MAE':>15} {'测试R²':>12} {'测试MAPE':>12}")
print('-' * 85)
for model_name, result in results.items():
    print(f"{model_name:<25} {result['test_rmse']:>15,.2f} {result['test_mae']:>15,.2f} "
          f"{result['test_r2']:>12.4f} {result['test_mape']:>11.2f}%")

best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
print(f"\n★ 最佳模型: {best_model[0]} (测试RMSE最小: {best_model[1]['test_rmse']:,.2f})")

# 8. 可视化结果
print("\n【步骤8：结果可视化】")
print("=" * 80)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('美国新冠肺炎总确诊量预测模型对比（无数据泄露版）', fontsize=16, fontweight='bold', y=0.99)

for idx, (model_name, result) in enumerate(results.items()):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    test_dates = test_data['天数'].values
    ax.plot(test_dates, y_test, 'o-', label='实际值', linewidth=2, markersize=8, color='blue')
    ax.plot(test_dates, result['y_test_pred'], 's--', label='预测值', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('天数', fontsize=10)
    ax.set_ylabel('总确诊数', fontsize=10)
    ax.set_title(f'{model_name}\nR²={result["test_r2"]:.4f}, MAPE={result["test_mape"]:.2f}%', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    for i, (date, actual, pred) in enumerate(zip(test_dates, y_test, result['y_test_pred'])):
        if i % 2 == 0:
            ax.annotate(f'{int(actual):,}', (date, actual), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, color='blue')

plt.tight_layout(pad=3.0, h_pad=12.0, w_pad=3.0)
plt.savefig('预测结果对比.png', dpi=300, bbox_inches='tight')
print("图表已保存为: 预测结果对比.png")
plt.show()

# 绘制整体趋势图
fig2, ax2 = plt.subplots(figsize=(14, 7))
all_days = df['天数'].values
all_confirmed = df[target_col].values
ax2.plot(all_days[:train_size], all_confirmed[:train_size], 'o-', label='训练集数据', linewidth=2, markersize=4, color='green', alpha=0.7)
ax2.plot(all_days[train_size:train_size+test_size], y_test, 'o-', label='测试集实际值', linewidth=2, markersize=8, color='blue')
ax2.plot(all_days[train_size:train_size+test_size], best_model[1]['y_test_pred'], 's--', label=f'测试集预测值({best_model[0]})', linewidth=2, markersize=8, color='red')
ax2.axvline(x=train_size, color='gray', linestyle='--', linewidth=2, label='训练/测试分界线')
ax2.set_xlabel('天数（自2020.1.28起）', fontsize=12)
ax2.set_ylabel('总确诊数', fontsize=12)
ax2.set_title('美国新冠肺炎总确诊数趋势及预测', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('整体趋势图.png', dpi=300, bbox_inches='tight')
print("图表已保存为: 整体趋势图.png")
plt.show()

# 9. 实验结论
print("\n\n【步骤9：实验结论】")
print("=" * 80)
print("\n实验完成!")
print("=" * 80)