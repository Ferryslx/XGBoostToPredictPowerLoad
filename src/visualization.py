# -*- coding: utf-8 -*-
"""
电力负荷预测 - 可视化模块
生成 5 类高质量分析图表：
  1. 滞后特征自相关分析图 (ACF)
  2. 特征相关性热力图
  3. XGBoost 训练损失下降曲线
  4. XGBoost 内置特征重要性排序 (Weight / Gain / Cover)
  5. SHAP 全局特征重要性图
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import data_preprocessing

# ============================================================================
# 全局绘图风格与字体设置
# 注意：font.sans-serif 必须放在 sns.set_style 之后，否则 seaborn 会覆盖字体
# ============================================================================
sns.set_style('whitegrid')
sns.set_palette('Set2')

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'lines.linewidth': 2.0,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'legend.fancybox': False,
})

# 论文级配色
C_DARK    = '#2C3E50'
C_PRIMARY = '#2980B9'
C_ACCENT  = '#E67E22'
C_GREEN   = '#27AE60'
C_RED     = '#C0392B'
C_PURPLE  = '#8E44AD'
C_GRAY    = '#7F8C8D'
C_LIGHT   = '#BDC3C7'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'diagrams')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 数据加载与特征工程
# ============================================================================
print("=" * 60)
print("  电力负荷预测 - 论文级可视化")
print("=" * 60)

print("\n[加载] 数据集...")
data = data_preprocessing(os.path.join(BASE_DIR, 'data', 'train.csv'))

print("[处理] 特征工程...")
feature_data = data.copy()
feature_data['hour'] = feature_data['time'].str[11:13]
feature_data['month'] = feature_data['time'].str[5:7]
feature_data = pd.get_dummies(feature_data, columns=['hour', 'month'])
dummy_cols = [col for col in feature_data.columns
              if col.startswith('hour_') or col.startswith('month_')]
feature_data[dummy_cols] = feature_data[dummy_cols].astype(int)

# 滞后特征
feature_data['Lag_1h'] = feature_data['power_load'].shift(1)
feature_data['Lag_2h'] = feature_data['power_load'].shift(2)
feature_data['Lag_3h'] = feature_data['power_load'].shift(3)
feature_data['Lag_24h'] = feature_data['power_load'].shift(24)
feature_data = feature_data.dropna()

feature_columns = [col for col in feature_data.columns
                   if col not in ['time', 'power_load']]


def simplify_feature_name(name):
    """将 one-hot 编码的特征名简化为论文中更易读的形式"""
    if name.startswith('hour_'):
        h = name.split('_')[1]
        return f'{h}:00'
    if name.startswith('month_'):
        m = name.split('_')[1]
        return f'{int(m)}月'
    mapping = {
        'Lag_1h': '前1小时负荷',
        'Lag_2h': '前2小时负荷',
        'Lag_3h': '前3小时负荷',
        'Lag_24h': '昨日同时刻负荷',
    }
    return mapping.get(name, name)


# 数据划分
split_idx = int(len(feature_data) * 0.8)
train = feature_data.iloc[:split_idx]
test = feature_data.iloc[split_idx:]
x_train = train[feature_columns]
y_train = train['power_load']
x_test = test[feature_columns]
y_test = test['power_load']

print(f"       训练集: {x_train.shape[0]} 样本 x {x_train.shape[1]} 特征")
print(f"       验证集: {x_test.shape[0]} 样本 x {x_test.shape[1]} 特征")

# ============================================================================
# 训练模型（用于损失曲线、特征重要性、SHAP）
# ============================================================================
print("[训练] XGBoost...")
eval_set = [(x_train, y_train), (x_test, y_test)]
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='rmse',
    early_stopping_rounds=50,
    random_state=42
)
xgb_model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
results = xgb_model.evals_result()

best_iteration = xgb_model.best_iteration
best_rmse = results['validation_1']['rmse'][best_iteration]

print(f"       最优迭代轮次: {best_iteration},  验证集 RMSE: {best_rmse:.4f}")

# ============================================================================
# 图表 1 —— 负荷序列自相关分析 (ACF)
# ============================================================================
print("\n[图表 1/5] 负荷序列自相关分析 (ACF)...")

fig, ax = plt.subplots(figsize=(10, 5))

max_lags = 168
acf_values = [1.0]
for lag in range(1, max_lags + 1):
    acf_values.append(data['power_load'].autocorr(lag=lag))

lags = range(0, max_lags + 1)
ci = 1.96 / np.sqrt(len(data))

ax.bar(lags, acf_values, width=0.6, color=C_PRIMARY,
       edgecolor='white', linewidth=0.3, alpha=0.85)

ax.axhline(y=ci, color=C_RED, linestyle='--', linewidth=1.0,
           alpha=0.7, label=f'95% 置信界 (±{ci:.3f})')
ax.axhline(y=-ci, color=C_RED, linestyle='--', linewidth=1.0, alpha=0.7)
ax.fill_between([0, max_lags], -ci, ci, color=C_RED, alpha=0.04)
ax.axhline(y=0, color=C_DARK, linewidth=0.8)

for key_lag, label, color in [(24, '24h (天)', C_ACCENT),
                                (48, '48h', C_GRAY),
                                (168, '168h (周)', C_GREEN)]:
    if key_lag <= max_lags:
        ax.annotate(label,
                    xy=(key_lag, acf_values[key_lag]),
                    xytext=(0, 18), textcoords='offset points',
                    fontsize=9, color=color, weight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

ax.set_xlabel('滞后步长 / h', fontsize=13, color=C_DARK)
ax.set_ylabel('自相关系数 (ACF)', fontsize=13, color=C_DARK)
ax.set_title('电力负荷序列自相关分析', fontsize=15, color=C_DARK, pad=12)
ax.set_xlim(-1, max_lags + 1)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '1-滞后特征自相关分析图.png'),
            dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("      -> 1-滞后特征自相关分析图.png")

# ============================================================================
# 图表 2 —— 特征相关性热力图
# ============================================================================
print("\n[图表 2/5] 特征相关性热力图...")

cont_feats = ['power_load', 'Lag_1h', 'Lag_2h', 'Lag_3h', 'Lag_24h']
hour_cols = [c for c in feature_data.columns if c.startswith('hour_')]
month_cols = [c for c in feature_data.columns if c.startswith('month_')]
all_candidates = cont_feats + hour_cols + month_cols

corr_target = feature_data[all_candidates].corr()['power_load'].abs().sort_values(ascending=False)
top_features = corr_target.head(20).index.tolist()
if 'power_load' in top_features:
    top_features.remove('power_load')
top_features = ['power_load'] + top_features[:19]

corr_data = feature_data[top_features].corr()
simple_labels = [simplify_feature_name(c) if c != 'power_load' else '负荷'
                 for c in corr_data.columns]

fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=256, center='light')

sns.heatmap(corr_data,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.8,
            linecolor='white',
            annot_kws={'size': 7.5, 'weight': 'bold'},
            cbar_kws={'label': 'Pearson 相关系数', 'shrink': 0.82},
            xticklabels=simple_labels,
            yticklabels=simple_labels,
            ax=ax)

ax.set_title('特征间 Pearson 相关系数矩阵', fontsize=15, color=C_DARK, pad=14)
ax.tick_params(axis='x', labelsize=8, rotation=50, labelcolor=C_DARK)
ax.tick_params(axis='y', labelsize=8, rotation=0, labelcolor=C_DARK)
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '2-特征相关性热力图.png'),
            dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("      -> 2-特征相关性热力图.png")

# ============================================================================
# 图表 3 —— XGBoost 训练损失下降曲线
# ============================================================================
print("\n[图表 3/5] XGBoost 训练损失下降曲线...")

train_rmse = results['validation_0']['rmse']
val_rmse = results['validation_1']['rmse']
n_rounds = len(train_rmse)

fig, ax = plt.subplots(figsize=(10, 5.5))
epochs = range(1, n_rounds + 1)

ax.fill_between(epochs, train_rmse, val_rmse,
                color=C_GRAY, alpha=0.08, label='泛化间隙 (Generalization Gap)')

ax.plot(epochs, train_rmse, color=C_PRIMARY, linewidth=2.2, label='训练集 RMSE')
ax.plot(epochs, val_rmse, color=C_ACCENT, linewidth=2.2, label='验证集 RMSE')

ax.axvline(x=best_iteration, color=C_GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
ax.scatter([best_iteration], [best_rmse], color=C_GREEN, s=100, zorder=5,
           edgecolors='white', linewidths=1.5)
ax.annotate(f' 最优点: iter={best_iteration}\n RMSE={best_rmse:.4f}',
            xy=(best_iteration, best_rmse),
            xytext=(best_iteration + 15, best_rmse + 50),
            fontsize=9.5, color=C_GREEN, weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=C_GREEN, alpha=0.85),
            arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.3))

ax.set_xlabel('Boosting 迭代轮次', fontsize=13, color=C_DARK)
ax.set_ylabel('RMSE / MW', fontsize=13, color=C_DARK)
ax.set_title('XGBoost 模型训练损失下降曲线', fontsize=15, color=C_DARK, pad=12)
ax.set_xlim(1, n_rounds)
ax.legend(loc='upper right', fontsize=11, ncol=1)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '3-XGBoost训练损失下降曲线.png'),
            dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("      -> 3-XGBoost训练损失下降曲线.png")

# ============================================================================
# 图表 4 —— XGBoost 内置特征重要性 (Weight / Gain / Cover)
# ============================================================================
print("\n[图表 4/5] XGBoost 内置特征重要性排序...")

importance_config = [
    ('weight', 'Weight (特征被选为分裂节点的次数)', 'Blues'),
    ('gain',   'Gain (特征带来的平均信息增益)',   'Oranges'),
    ('cover',  'Cover (特征覆盖的样本比例)',      'Greens'),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.subplots_adjust(wspace=0.35)

for i, (imp_type, title, cmap_name) in enumerate(importance_config):
    ax = axes[i]
    booster = xgb_model.get_booster()
    importance_dict = booster.get_score(importance_type=imp_type)

    if importance_dict:
        first_key = list(importance_dict.keys())[0]
        if first_key.startswith('f') and first_key[1:].isdigit():
            mapped = {}
            for fid, score in importance_dict.items():
                idx = int(fid[1:])
                if idx < len(feature_columns):
                    mapped[simplify_feature_name(feature_columns[idx])] = score
                else:
                    mapped[fid] = score
            importance_dict = mapped

    imp_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=True)
    imp_df = imp_df.tail(15)

    max_val = imp_df['importance'].max()
    norm = imp_df['importance'] / max_val if max_val > 0 else imp_df['importance']
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(0.3 + 0.7 * norm)

    bars = ax.barh(imp_df['feature'], imp_df['importance'],
                   color=colors, edgecolor='white', linewidth=0.5)

    for bar_obj, val in zip(bars, imp_df['importance']):
        ax.text(bar_obj.get_width() + max_val * 0.01,
                bar_obj.get_y() + bar_obj.get_height() / 2,
                f'{val:.0f}' if val >= 1 else f'{val:.2f}',
                va='center', fontsize=7.5, color=C_DARK)

    ax.set_title(title, fontsize=12, color=C_DARK, pad=10)
    ax.set_xlabel('重要性得分', fontsize=10, color=C_DARK)
    ax.tick_params(axis='y', labelsize=9, color=C_DARK)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

fig.suptitle('XGBoost 特征重要性排序（三种度量方式对比）', fontsize=15,
             color=C_DARK, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '4-特征重要性排序条形图.png'),
            dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("      -> 4-特征重要性排序条形图.png")

# ============================================================================
# 图表 5 —— SHAP 全局特征重要性
# ============================================================================
print("\n[图表 5/5] SHAP 全局特征重要性...")

x_sample = x_train.sample(min(1500, len(x_train)), random_state=42)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_sample)

# === 5a: SHAP Bar Plot ===
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, x_sample,
    plot_type='bar',
    show=False,
    max_display=20,
    color=C_PRIMARY,
    class_names=None
)

lbls = [simplify_feature_name(lbl.get_text()) for lbl in ax.get_yticklabels()]
ax.set_yticklabels(lbls)

ax.set_title('SHAP 全局特征重要性 (Mean |SHAP Value|)',
             fontsize=15, color=C_DARK, pad=12)
ax.set_xlabel('平均 |SHAP| 值（对模型输出的平均影响幅度）', fontsize=12, color=C_DARK)
ax.set_ylabel('')
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='x')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '5-SHAP全局特征重要性图.png'),
            dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("      -> 5-SHAP全局特征重要性图.png")

# === 5b: SHAP Beeswarm Summary Plot ===
fig = plt.figure(figsize=(12, 8))

x_sample_renamed = x_sample.copy()
rename_dict = {col: simplify_feature_name(col) for col in x_sample.columns}
x_sample_renamed = x_sample_renamed.rename(columns=rename_dict)

shap_values_renamed = explainer.shap_values(x_sample)
if hasattr(shap_values_renamed, 'feature_names'):
    shap_values_renamed.feature_names = [simplify_feature_name(f)
                                          for f in shap_values_renamed.feature_names]

shap.summary_plot(
    shap_values_renamed, x_sample_renamed,
    show=False,
    max_display=20,
    plot_size=None
)

ax = plt.gca()
ax.set_title('SHAP 特征影响概要图',
             fontsize=15, color=C_DARK, pad=12)
ax.set_xlabel('SHAP 值（对预测负荷的贡献 / MW）', fontsize=12, color=C_DARK)
ax.set_ylabel('')
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.2, linestyle='--', axis='x')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '5-SHAP特征重要性概要图.png'),
            dpi=300, facecolor='white', edgecolor='none')
plt.close()
print("      -> 5-SHAP特征重要性概要图.png")

# ============================================================================
print("\n" + "=" * 60)
print("  [完成] 所有论文级图表已生成至:")
print(f"         {OUTPUT_DIR}")
print("=" * 60)
