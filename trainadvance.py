import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


# ---------------------------------------------------------
# 1. 核心功能：数据增强 (把 1 条变成 3 条)
# ---------------------------------------------------------
def augment_data(df):
    synthetic_rows = []
    print(f"📉 原始数据量: {len(df)} 条")

    for idx, row in df.iterrows():
        try:
            # 获取基础信息
            bw = float(row['bandwidth_mbps'])
            price = float(row['price'])

            # --- 裂变规则 1: 生成更高带宽的数据 ---
            # 假设: 带宽 x2 -> 价格 x1.2 (保守估计)
            new_row_high = row.copy()
            new_row_high['bandwidth_mbps'] = bw * 2
            new_row_high['price'] = price * 1.2
            # 标记一下这是合成数据（可选）
            synthetic_rows.append(new_row_high)

            # --- 裂变规则 2: 生成更低带宽的数据 ---
            # 假设: 带宽 x0.5 -> 价格 x0.8
            if bw >= 20:  # 太小的带宽就不减了
                new_row_low = row.copy()
                new_row_low['bandwidth_mbps'] = bw * 0.5
                new_row_low['price'] = price * 0.8
                synthetic_rows.append(new_row_low)

        except Exception as e:
            continue

    # 合并数据
    df_synthetic = pd.DataFrame(synthetic_rows)
    df_final = pd.concat([df, df_synthetic], ignore_index=True)
    print(f"📈 增强后数据量: {len(df_final)} 条 (涨了 {len(df_synthetic)} 条)")
    return df_final


# ---------------------------------------------------------
# 2. 核心功能：地区归类 (让 AI 变聪明)
# ---------------------------------------------------------
def get_zone_from_postcode(val):
    # 尝试从输入中提取数字邮编
    import re
    val_str = str(val)
    digits = re.findall(r'\b\d{4}\b', val_str)

    if not digits:
        return 'Unknown_Zone'

    postcode = int(digits[0])

    # 简单的澳洲大区划分规则 (你可以根据业务调整)
    if 2000 <= postcode <= 2234: return 'NSW_Metro'
    if 2500 <= postcode <= 2530: return 'NSW_Regional'  # 卧龙岗等
    if 2600 <= postcode <= 2620: return 'ACT_Metro'
    if 3000 <= postcode <= 3207: return 'VIC_Metro'
    if 4000 <= postcode <= 4179: return 'QLD_Metro'
    if 6000 <= postcode <= 6199: return 'WA_Metro'

    return 'Other_Regional'  # 其他偏远地区


# =========================================================
# 主程序
# =========================================================

print("正在读取数据...")
# 1. 读取
try:
    df = pd.read_csv("training_mvp_v5.csv")

    # 映射列名 (兼容你现在的 v5 文件)
    col_map = {
        'operator_manual': 'operator',
        'product_type_rule': 'product_type',
        'region_rule': 'region',
        'MRC (ex)': 'price',
        'mrc': 'price'
    }
    df = df.rename(columns=col_map)

    # 确保是数字
    df['bandwidth_mbps'] = pd.to_numeric(df['bandwidth_mbps'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['bandwidth_mbps', 'price'])

except Exception as e:
    print(f"读取失败: {e}")
    exit()

# 2. 执行数据增强
df_train = augment_data(df)

# 3. 执行地区归类 (Feature Engineering)
print("正在进行智能地区归类...")
df_train['zone'] = df_train['region'].apply(get_zone_from_postcode)
print("✅ 地区归类完成！AI 现在是按 '大区' 学习，而不是死记硬背邮编。")

# 4. 训练模型
print("开始训练增强版模型...")
# 注意：我们要把新的 'zone' 加入到特征里
X = df_train[['operator', 'product_type', 'bandwidth_mbps', 'term_months', 'zone']]
y = df_train['price']

# 填充缺失值
X = X.fillna('UNKNOWN')

model = CatBoostRegressor(
    iterations=1000,  # 数据多了，可以多练几轮
    learning_rate=0.05,
    depth=6,
    verbose=False,
    cat_features=['operator', 'product_type', 'zone']  # 注意这里用了 zone
)

model.fit(X, y)
model.save_model("telecom_model.cbm")
print("🎉 模型训练完毕！这个模型比之前的强多了。")