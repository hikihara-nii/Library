import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 0. 基本設定 ＆ 日本語フォント設定
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'TakaoPGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def find_file(filename):
    """ ファイルの自動探索 """
    candidate_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename)
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

def parse_macro_tsv(filepath, publisher_name):
    """ TSVマクロデータのパース処理 """
    if not filepath or not os.path.exists(filepath):
        return pd.DataFrame()
        
    sections = {}
    current_sec = None
    lines = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('====='):
                continue
            if stripped and not line.startswith('\t') and not line[0].isdigit():
                if current_sec and lines:
                    sections[current_sec] = lines
                    lines = []
                current_sec = stripped
            else:
                if current_sec:
                    lines.append(line)
        if current_sec and lines:
            sections[current_sec] = lines
            
    records = []
    for sec_name, sec_lines in sections.items():
        data_lines = [l for l in sec_lines if l.strip()]
        if len(data_lines) >= 3:
            upper_header = data_lines[0].split('\t')
            sub_header = data_lines[1].split('\t')
            
            curr_upper = ''
            upper_cols = []
            for u in upper_header[1:]:
                if u.strip():
                    curr_upper = u.strip()
                upper_cols.append(curr_upper)
            
            cols = ['Year']
            for u, s in zip(upper_cols, sub_header[1:]):
                cols.append(f"{u}_{s.strip()}")
                
            for dl in data_lines[2:]:
                parts = dl.strip().split('\t')
                if parts[0].isdigit():
                    row_dict = {'Publisher': publisher_name, 'Institution': sec_name, 'Year': int(parts[0])}
                    for c_idx, val in enumerate(parts[1:], start=1):
                        if c_idx < len(cols):
                            clean_val = val.replace(',', '').strip()
                            row_dict[cols[c_idx]] = float(clean_val) if clean_val.replace('.','',1).isdigit() else 0.0
                    records.append(row_dict)
                    
    return pd.DataFrame(records)

# ------------------------------------------------------------------------------
# 1. データの準備
# ------------------------------------------------------------------------------
print("📂 データを読み込んでいます...")

# マクロTSVの読み込み
df_els_m = parse_macro_tsv(find_file("集計_elsevier.tsv"), "Elsevier")
df_wiley_m = parse_macro_tsv(find_file("集計_wiley.tsv"), "Wiley")
df_macro = pd.concat([df_els_m, df_wiley_m], ignore_index=True)

df_univ = df_macro[df_macro['Institution'] != '日本全体'].copy()

univ_summary = df_univ.groupby(['Publisher', 'Institution']).agg({
    '責任著者のみ_論文数': 'sum',
    '責任著者のみ_gold': 'sum',
    '責任著者のみ_hybrid': 'sum',
    '責任著者のみ_closed': 'sum',
    '責任著者のみ_APC支払額USD': 'sum'
}).reset_index()

univ_summary['責任著者_OA論文数'] = univ_summary['責任著者のみ_gold'] + univ_summary['責任著者のみ_hybrid']
univ_summary['1論文あたり平均APC単価_USD'] = np.where(
    univ_summary['責任著者_OA論文数'] > 0,
    univ_summary['責任著者のみ_APC支払額USD'] / univ_summary['責任著者_OA論文数'],
    0.0
)
univ_summary['100%_OA上限コスト_USD'] = univ_summary['責任著者のみ_論文数'] * univ_summary['1論文あたり平均APC単価_USD']

# OpenAlex ミクロデータの読み込み
df_wiley_oa = pd.read_csv(find_file("publisher_counts_P4310320595-wiley_2024-2026_OA_analysis.csv")).assign(publisher='Wiley')
df_els_oa = pd.read_csv(find_file("publisher_counts_P4310320990-elsevier_2024-2026_OA_analysis.csv")).assign(publisher='Elsevier')
df_oa_all = pd.concat([df_wiley_oa, df_els_oa], ignore_index=True)

j_summary = df_oa_all.groupby(['publisher', 'journal_name', 'issn_l', 'eissn']).agg({
    'total_papers': 'sum',
    'publisher_oa_papers': 'sum',
    'green_only_papers': 'sum'
}).reset_index()

j_summary['paywall_rate_%'] = (1.0 - (j_summary['publisher_oa_papers'] / j_summary['total_papers'])) * 100.0

# COUNTER利用数（シミュレーションデータ）
np.random.seed(42)
j_summary['counter_dl'] = (np.random.negative_binomial(3, 0.01, size=len(j_summary)) + 
                           j_summary['total_papers'] * np.random.uniform(0.1, 1.2, size=len(j_summary))).astype(int)

def classify_action(row):
    high_read = row['counter_dl'] >= 300
    high_paywall = row['paywall_rate_%'] >= 50.0
    high_publish = row['total_papers'] >= 100
    
    if high_publish and high_paywall:
        return "【A】転換契約コア誌 (高投稿 ✕ 高Paywall)"
    elif high_read and high_paywall:
        return "【B】純粋購読維持誌 (高需要 ✕ 高Paywall)"
    elif high_read and not high_paywall:
        return "【C】OA代替解約候補 (高需要 ✕ 低Paywall)"
    else:
        return "【D】ILL/PPV補填誌 (低需要手配対応)"

j_summary['action_category'] = j_summary.apply(classify_action, axis=1)


# ------------------------------------------------------------------------------
# 📊 グラフ 1: 転換契約費用妥当性（現行APC実績 vs 100%OA理論上限コスト）
# ------------------------------------------------------------------------------
print("🎨 グラフ 1 を生成中...")
fig1, ax1 = plt.subplots(figsize=(10, 6))

pub_cost_summary = univ_summary.groupby('Publisher').agg({
    '責任著者のみ_APC支払額USD': 'sum',
    '100%_OA上限コスト_USD': 'sum'
}).reset_index()

pub_cost_melted = pd.melt(
    pub_cost_summary,
    id_vars=['Publisher'],
    value_vars=['責任著者のみ_APC支払額USD', '100%_OA上限コスト_USD'],
    var_name='Cost_Type',
    value_name='Amount_USD'
)

pub_cost_melted['Cost_Type'] = pub_cost_melted['Cost_Type'].replace({
    '責任著者のみ_APC支払額USD': '① 現行APC実支払額 (2023-2025合計)',
    '100%_OA上限コスト_USD': '② 全責任著者論文 100%OA化時の理論上限コスト'
})

sns.barplot(
    data=pub_cost_melted,
    x='Publisher',
    y='Amount_USD',
    hue='Cost_Type',
    palette=['#2ca02c', '#ff7f0e'],
    ax=ax1
)

ax1.set_title("【9大学合計】現行APC実支出 vs 100%OA化時の理論的コスト上限（転換契約交渉用ベンチマーク）", fontsize=13, fontweight='bold')
ax1.set_xlabel("出版社", fontsize=11)
ax1.set_ylabel("金額 (USD)", fontsize=11)
ax1.grid(True, axis='y', ls='--', alpha=0.3)

for p in ax1.patches:
    val = p.get_height()
    if val > 0:
        ax1.text(p.get_x() + p.get_width()/2, val + 600000, f"${val/1e6:.2f}M USD\n(約{val*155/1e8:.2f}億円)", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("viz_transformative_agreement_valuation.png", dpi=300, bbox_inches='tight')
plt.close()


# ------------------------------------------------------------------------------
# 📊 グラフ 2: 大学別・1論文あたり平均実効APC単価 ＆ 出版形態内訳
# ------------------------------------------------------------------------------
print("🎨 グラフ 2 を生成中...")
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

# 左: 大学別 APC平均単価
sns.barplot(
    data=univ_summary,
    x='Institution',
    y='1論文あたり平均APC単価_USD',
    hue='Publisher',
    palette={'Elsevier': '#d62728', 'Wiley': '#1f77b4'},
    ax=axes2[0]
)
axes2[0].set_title("【大学別】1論文あたり平均実効APC単価 (USD)", fontsize=13, fontweight='bold')
axes2[0].set_xlabel("大学・機関", fontsize=11)
axes2[0].set_ylabel("平均APC単価 (USD)", fontsize=11)
axes2[0].set_xticklabels(axes2[0].get_xticklabels(), rotation=45, ha='right')
axes2[0].grid(True, axis='y', ls='--', alpha=0.3)

for p in axes2[0].patches:
    h = p.get_height()
    if h > 0:
        axes2[0].text(p.get_x() + p.get_width()/2, h + 50, f"${h:,.0f}", ha='center', va='bottom', fontsize=8, rotation=90)

# 右: 大学別 責任著者論文の出版形態内訳
df_univ_pub = univ_summary.melt(
    id_vars=['Publisher', 'Institution'],
    value_vars=['責任著者のみ_gold', '責任著者のみ_hybrid', '責任著者のみ_closed'],
    var_name='OA_Type',
    value_name='Paper_Count'
)
df_univ_pub['OA_Type'] = df_univ_pub['OA_Type'].replace({
    '責任著者のみ_gold': 'Gold OA',
    '責任著者のみ_hybrid': 'Hybrid OA',
    '責任著者のみ_closed': 'Closed (非OA)'
})

sns.barplot(
    data=df_univ_pub,
    x='Institution',
    y='Paper_Count',
    hue='OA_Type',
    palette={'Gold OA': '#2ca02c', 'Hybrid OA': '#bcbd22', 'Closed (非OA)': '#7f7f7f'},
    ax=axes2[1]
)
axes2[1].set_title("【大学別】責任著者論文の出版形態別内訳 (2023-2025)", fontsize=13, fontweight='bold')
axes2[1].set_xlabel("大学・機関", fontsize=11)
axes2[1].set_ylabel("論文数 (本)", fontsize=11)
axes2[1].set_xticklabels(axes2[1].get_xticklabels(), rotation=45, ha='right')
axes2[1].grid(True, axis='y', ls='--', alpha=0.3)

plt.tight_layout()
plt.savefig("viz_univ_apc_breakdown.png", dpi=300, bbox_inches='tight')
plt.close()


# ------------------------------------------------------------------------------
# 📊 グラフ 3: 多次元（投稿・閲覧・Paywall）アクション分類マトリクス
# ------------------------------------------------------------------------------
print("🎨 グラフ 3 を生成中...")
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

# 左: 出版社別 アクション分類構成比
cat_counts = j_summary.groupby(['publisher', 'action_category']).size().unstack(fill_value=0)
cat_pct = cat_counts.div(cat_counts.sum(axis=1), axis=0) * 100

cat_pct.plot(
    kind='bar',
    stacked=True,
    color=['#d62728', '#1f77b4', '#bcbd22', '#7f7f7f'],
    ax=axes3[0]
)
axes3[0].set_title("【多次元評価】出版社別 契約・交渉アクション割合 (%)", fontsize=13, fontweight='bold')
axes3[0].set_xlabel("出版社", fontsize=11)
axes3[0].set_ylabel("割合 (%)", fontsize=11)
axes3[0].set_xticklabels(axes3[0].get_xticklabels(), rotation=0)
axes3[0].legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2, fontsize=9)
axes3[0].grid(True, axis='y', ls='--', alpha=0.3)

for p in axes3[0].patches:
    h = p.get_height()
    if h > 3:
        x, y = p.get_xy()
        axes3[0].text(x + p.get_width()/2, y + h/2, f"{h:.1f}%", ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# 右: 4象限＋多次元散布図
sns.scatterplot(
    data=j_summary,
    x='paywall_rate_%',
    y='counter_dl',
    hue='action_category',
    palette={
        "【A】転換契約コア誌 (高投稿 ✕ 高Paywall)": "#d62728",
        "【B】純粋購読維持誌 (高需要 ✕ 高Paywall)": "#1f77b4",
        "【C】OA代替解約候補 (高需要 ✕ 低Paywall)": "#bcbd22",
        "【D】ILL/PPV補填誌 (低需要手配対応)": "#7f7f7f"
    },
    alpha=0.6,
    s=30,
    ax=axes3[1]
)
axes3[1].axvline(x=50.0, color='black', linestyle='--', linewidth=1.5, label='Paywall Threshold (50%)')
axes3[1].axhline(y=300, color='black', linestyle=':', linewidth=1.5, label='COUNTER DL Threshold (300件)')
axes3[1].set_yscale('log')
axes3[1].set_title("【雑誌マトリクス】閲覧需要 ✕ Paywall率 ✕ アクション分類", fontsize=13, fontweight='bold')
axes3[1].set_xlabel("Paywall（非OA）率 (%) -> 右ほど有料壁依存", fontsize=11)
axes3[1].set_ylabel("COUNTER ダウンロード数 (対数軸)", fontsize=11)
axes3[1].legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2, fontsize=8)
axes3[1].grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("viz_multidimensional_action_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✨ すべてのグラフ画像 (3点) の出力が完了しました！")