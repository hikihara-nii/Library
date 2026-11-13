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
    """
    スクリプトの配置先や現在の作業ディレクトリ、data/ フォルダなどを自動探索する関数
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidate_paths = [
        os.path.join(current_script_dir, filename),                     # スクリプトと同じフォルダ
        os.path.join(current_script_dir, "data", filename),             # スクリプトフォルダ直下の data/
        os.path.join(os.getcwd(), filename),                            # 作業ディレクトリ直下
        os.path.join(os.getcwd(), "data", filename),                    # 作業ディレクトリ直下の data/
        os.path.join("/data", filename),                                # ルート直下の /data/
        os.path.join(current_script_dir, "..", filename)                # 1つ上のフォルダ
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

# ------------------------------------------------------------------------------
# 1. データの自動探索とロード
# ------------------------------------------------------------------------------
els_file = find_file("publisher_counts_P4310320990-elsevier_2024-2026_OA_analysis.csv")
wiley_file = find_file("publisher_counts_P4310320595-wiley_2024-2026_OA_analysis.csv")

if not els_file or not wiley_file:
    print("❌ データファイルが見つかりませんでした。ファイル名と配置場所を確認してください。")
    print(f"  ・Elsevier探索結果: {els_file}")
    print(f"  ・Wiley探索結果   : {wiley_file}")
    exit(1)

print(f"📂 Elsevier データをロード中: {els_file}")
print(f"📂 Wiley データをロード中   : {wiley_file}")

df_els = pd.read_csv(els_file).assign(publisher='Elsevier')
df_wiley = pd.read_csv(wiley_file).assign(publisher='Wiley')
df_all = pd.concat([df_els, df_wiley], ignore_index=True)

# ------------------------------------------------------------------------------
# 2. Green OA 攻防指標の算出（雑誌単位）
# ------------------------------------------------------------------------------
print("🔬 Green OA 攻防指標（貫通率・無償依存度）を計算中...")

journal_green_metrics = df_all.groupby(['publisher', 'journal_name', 'issn_l']).agg({
    'total_papers': 'sum',
    'publisher_oa_papers': 'sum',
    'green_only_papers': 'sum'
}).reset_index()

# ① 有料壁（Closed）論文数
journal_green_metrics['closed_papers'] = journal_green_metrics['total_papers'] - journal_green_metrics['publisher_oa_papers']

# ② 【指標1】Green貫通率 (%): 非OA(Closed)論文のうち、著者の草の根努力（リポジトリ等）で読める割合
journal_green_metrics['green_salvage_rate_%'] = np.where(
    journal_green_metrics['closed_papers'] > 0,
    (journal_green_metrics['green_only_papers'] / journal_green_metrics['closed_papers']) * 100.0,
    0.0
)

# ③ 【指標2】無償OA依存度 (%): 全OA(Gold/Hybrid/Green)のうち、APC負担ゼロでオープン化できた割合
journal_green_metrics['total_oa_papers'] = journal_green_metrics['publisher_oa_papers'] + journal_green_metrics['green_only_papers']
journal_green_metrics['cost_free_oa_share_%'] = np.where(
    journal_green_metrics['total_oa_papers'] > 0,
    (journal_green_metrics['green_only_papers'] / journal_green_metrics['total_oa_papers']) * 100.0,
    0.0
)

# 出版社レベルでの集計表示
pub_summary = journal_green_metrics.groupby('publisher').agg({
    'total_papers': 'sum',
    'publisher_oa_papers': 'sum',
    'green_only_papers': 'sum',
    'closed_papers': 'sum'
}).reset_index()

pub_summary['全体Green貫通率_%'] = (pub_summary['green_only_papers'] / pub_summary['closed_papers']) * 100.0
pub_summary['全体無償OA依存度_%'] = (pub_summary['green_only_papers'] / (pub_summary['publisher_oa_papers'] + pub_summary['green_only_papers'])) * 100.0

print("\n=== 出版社別 Green OA 評価結果サマリー ===")
for _, r in pub_summary.iterrows():
    print(f"■ {r['publisher']}")
    print(f"  ・総論文数           : {r['total_papers']:,} 本")
    print(f"  ・出版社OA論文数     : {r['publisher_oa_papers']:,} 本")
    print(f"  ・Green Only 論文数  : {r['green_only_papers']:,} 本")
    print(f"  ・【Green 貫通率】   : {r['全体Green貫通率_%']:.2f}% (有料壁を突破して救済できた割合)")
    print(f"  ・【無償OA 依存度】  : {r['全体無償OA依存度_%']:.2f}% (APCゼロで獲得できたOAの割合)")
    print("  ──────────────────────────────────────────────")

# ------------------------------------------------------------------------------
# 3. 評価グラフの可視化と画像保存
# ------------------------------------------------------------------------------
print("🎨 可視化グラフを作成中...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# 左グラフ: Green 貫通率の分布（ボックスプロット）
sns.boxplot(
    data=journal_green_metrics,
    x='publisher',
    y='green_salvage_rate_%',
    palette={'Elsevier': '#d62728', 'Wiley': '#1f77b4'},
    ax=axes[0]
)
axes[0].set_title("【Green貫通率】非OA(Closed)論文のうち Green で読める割合の分布", fontsize=11, fontweight='bold')
axes[0].set_xlabel("出版社", fontsize=10)
axes[0].set_ylabel("Closed論文に占める Green 救済率 (%)", fontsize=10)
axes[0].grid(True, axis='y', ls='--', alpha=0.3)

els_mean = journal_green_metrics[journal_green_metrics['publisher']=='Elsevier']['green_salvage_rate_%'].mean()
wiley_mean = journal_green_metrics[journal_green_metrics['publisher']=='Wiley']['green_salvage_rate_%'].mean()
axes[0].text(0, els_mean + 2, f"平均: {els_mean:.1f}%", ha='center', fontweight='bold', color='#d62728')
axes[0].text(1, wiley_mean + 2, f"平均: {wiley_mean:.1f}%", ha='center', fontweight='bold', color='#1f77b4')

# 右グラフ: 全OAにおける構成比（Gold/Hybrid vs Green Only）
pub_green_totals = journal_green_metrics.groupby('publisher').agg({
    'publisher_oa_papers': 'sum',
    'green_only_papers': 'sum'
}).reset_index()

pub_green_totals['Total_OA'] = pub_green_totals['publisher_oa_papers'] + pub_green_totals['green_only_papers']
pub_green_totals['Gold/Hybrid (有料/APC)'] = (pub_green_totals['publisher_oa_papers'] / pub_green_totals['Total_OA']) * 100
pub_green_totals['Green Only (無償/要努力)'] = (pub_green_totals['green_only_papers'] / pub_green_totals['Total_OA']) * 100

pub_green_totals.set_index('publisher')[['Gold/Hybrid (有料/APC)', 'Green Only (無償/要努力)']].plot(
    kind='bar',
    stacked=True,
    color=['#2ca02c', '#1f77b4'],
    ax=axes[1]
)
axes[1].set_title("【OA獲得コスト構造】全OA論文に占める「有料(APC) vs 完全無償(Green)」割合", fontsize=11, fontweight='bold')
axes[1].set_xlabel("出版社", fontsize=10)
axes[1].set_ylabel("OA論文全体に対する構成比 (%)", fontsize=10)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
axes[1].grid(True, axis='y', ls='--', alpha=0.3)

for p in axes[1].patches:
    h = p.get_height()
    if h > 3:
        x, y = p.get_xy()
        axes[1].text(x + p.get_width()/2, y + h/2, f"{h:.1f}%", ha='center', va='center', fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
output_img = "viz_green_resistance_evaluation.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()

# 詳細CSV結果の保存
output_csv = "green_oa_evaluation_results.csv"
journal_green_metrics.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✨ 解析完了!")
print(f"  ・可視化画像を出力しました: {output_img}")
print(f"  ・詳細集計CSVを出力しました : {output_csv}")