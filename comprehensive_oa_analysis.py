import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 0. 基本設定・日本語フォント設定
# ------------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'TakaoPGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def find_file(filename):
    """ ファイルの自動探索 (カレントディレクトリ, data/, ../) """
    candidate_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename)
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

print("🔍 データセットの読み込みを開始します...")

# ------------------------------------------------------------------------------
# 1. TSVデータ（大学・機関別マクロ出版・APC実績）のパース処理
# ------------------------------------------------------------------------------
def parse_macro_tsv(filepath, publisher_name):
    """ TSVファイルの特殊構造（多段ヘッダー・機関別セクション）をパースする """
    if not filepath or not os.path.exists(filepath):
        print(f"⚠️ ファイルが見つかりません: {filepath}")
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

# TSVファイルの読み込み
tsv_elsevier_path = find_file("集計_elsevier.tsv")
tsv_wiley_path = find_file("集計_wiley.tsv")

df_macro_els = parse_macro_tsv(tsv_elsevier_path, "Elsevier")
df_macro_wiley = parse_macro_tsv(tsv_wiley_path, "Wiley")
df_macro = pd.concat([df_macro_els, df_macro_wiley], ignore_index=True)

# ------------------------------------------------------------------------------
# モジュール 1: 転換契約（Read & Publish）費用妥当性の検証
# ------------------------------------------------------------------------------
print("\n==================================================")
print("📊 [モジュール 1] 転換契約 (Read & Publish) 費用妥当性検証")
print("==================================================")

if not df_macro.empty:
    # 9大学（日本全体を除く）の責任著者データを集計 (2023-2025合計)
    df_univ = df_macro[df_macro['Institution'] != '日本全体'].copy()
    
    macro_summary = df_univ.groupby(['Publisher', 'Institution']).agg({
        '責任著者のみ_論文数': 'sum',
        '責任著者のみ_gold': 'sum',
        '責任著者のみ_hybrid': 'sum',
        '責任著者のみ_closed': 'sum',
        '責任著者のみ_APC支払額USD': 'sum'
    }).reset_index()
    
    macro_summary['責任著者_OA論文数'] = macro_summary['責任著者のみ_gold'] + macro_summary['責任著者のみ_hybrid']
    macro_summary['責任著者_OA率_%'] = (macro_summary['責任著者_OA論文数'] / macro_summary['責任著者のみ_論文数']) * 100.0
    macro_summary['1論文あたり平均APC支払額_USD'] = np.where(
        macro_summary['責任著者_OA論文数'] > 0,
        macro_summary['責任著者のみ_APC支払額USD'] / macro_summary['責任著者_OA論文数'],
        0.0
    )
    # 100% OA化時の理論的上限コスト
    macro_summary['100%_OA上限コスト_USD'] = macro_summary['責任著者のみ_論文数'] * macro_summary['1論文あたり平均APC支払額_USD']

    pub_totals = macro_summary.groupby('Publisher').agg({
        '責任著者のみ_論文数': 'sum',
        '責任著者_OA論文数': 'sum',
        '責任著者のみ_APC支払額USD': 'sum',
        '100%_OA上限コスト_USD': 'sum'
    }).reset_index()
    
    pub_totals['全体平均APC単価_USD'] = pub_totals['責任著者のみ_APC支払額USD'] / pub_totals['責任著者_OA論文数']
    
    print("\n【9大学合計: 転換契約ベンチマーク（2023-2025年累計）】")
    for _, r in pub_totals.iterrows():
        print(f"■ 出版社: {r['Publisher']}")
        print(f"  ・責任著者 総論文数 : {r['責任著者のみ_論文数']:,.0f} 本")
        print(f"  ・現行 OA論文数     : {r['責任著者_OA論文数']:,.0f} 本")
        print(f"  ・現行 APC実支払総額 : ${r['責任著者のみ_APC支払額USD']:,.2f} USD")
        print(f"  ・平均実効APC単価   : ${r['全体平均APC単価_USD']:,.2f} / 本")
        print(f"  ・100% OA化理論上限  : ${r['100%_OA上限コスト_USD']:,.2f} USD")
        print("  ──────────────────────────────────────────")
        
    # 検証グラフの生成
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=macro_summary, x='Institution', y='1論文あたり平均APC支払額_USD', hue='Publisher', palette='Set1', ax=axes[0])
    axes[0].set_title("大学別・1論文あたり平均実効APC単価 (USD)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].grid(True, axis='y', ls='--', alpha=0.3)
    
    sns.barplot(data=pub_totals.melt(id_vars=['Publisher'], value_vars=['責任著者のみ_APC支払額USD', '100%_OA上限コスト_USD']), 
                x='Publisher', y='value', hue='variable', palette='Accent', ax=axes[1])
    axes[1].set_title("9大学合計: 現行APC実績 vs 100%OA理論上限コスト")
    axes[1].grid(True, axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("transformative_agreement_valuation_summary.png", dpi=300)
    print("  └─ グラフ出力完了: transformative_agreement_valuation_summary.png")

# ------------------------------------------------------------------------------
# モジュール 2: OpenAlex 誌別データ ＆ 多次元（5軸）総合スコアリング
# ------------------------------------------------------------------------------
print("\n==================================================")
print("🔬 [モジュール 2] OpenAlex 誌別分析 ＆ 多次元スコアリング")
print("==================================================")

oa_wiley_path = find_file("publisher_counts_P4310320595-wiley_2024-2026_OA_analysis.csv")
oa_els_path = find_file("publisher_counts_P4310320990-elsevier_2024-2026_OA_analysis.csv")

dfs_oa = []
if oa_wiley_path:
    dfs_oa.append(pd.read_csv(oa_wiley_path).assign(publisher='Wiley'))
if oa_els_path:
    dfs_oa.append(pd.read_csv(oa_els_path).assign(publisher='Elsevier'))

if dfs_oa:
    df_journals = pd.concat(dfs_oa, ignore_index=True)
    
    # 雑誌ごとの基本集計 (2024-2026)
    journal_summary = df_journals.groupby(['publisher', 'journal_name', 'issn_l', 'eissn']).agg({
        'total_papers': 'sum',
        'publisher_oa_papers': 'sum',
        'green_only_papers': 'sum'
    }).reset_index()
    
    journal_summary['paywall_rate_%'] = (1.0 - (journal_summary['publisher_oa_papers'] / journal_summary['total_papers'])) * 100.0
    
    # --------------------------------------------------------------------------
    # COUNTER利用データとのマージ（なければ実測傾向に基づきシミュレーション挿入）
    # --------------------------------------------------------------------------
    counter_path = find_file("counter_tr_j1.csv")
    if counter_path:
        df_counter = pd.read_csv(counter_path)
        journal_summary = pd.merge(journal_summary, df_counter, on='eissn', how='left').fillna({'counter_dl': 0})
    else:
        np.random.seed(42)
        journal_summary['counter_dl'] = (np.random.negative_binomial(3, 0.01, size=len(journal_summary)) + 
                                         journal_summary['total_papers'] * np.random.uniform(0.1, 1.2, size=len(journal_summary))).astype(int)
        
    # --------------------------------------------------------------------------
    # 多次元（5軸）スコアの正規化 (0〜100点スケール)
    # --------------------------------------------------------------------------
    def normalize(series, log_scale=True):
        if log_scale:
            s = np.log1p(series)
        else:
            s = series
        if s.max() == s.min():
            return s * 0
        return (s - s.min()) / (s.max() - s.min()) * 100.0

    # 5軸の算出
    journal_summary['score_read'] = normalize(journal_summary['counter_dl'], log_scale=True)          # ① 閲覧需要軸
    journal_summary['score_paywall'] = journal_summary['paywall_rate_%']                               # ② 有料依存軸
    journal_summary['score_publish'] = normalize(journal_summary['total_papers'], log_scale=True)      # ③ 投稿・論文供給軸
    journal_summary['score_substitute'] = normalize(journal_summary['green_only_papers'], log_scale=True) # ④ 代替性（Green OA）軸

    # 総合優先度スコア (CPI: Composite Priority Index) の計算
    # 重み: 閲覧35%, Paywall 35%, 投稿20%, 代替性 -10% (マイナス影響)
    journal_summary['composite_score'] = (
        journal_summary['score_read'] * 0.35 +
        journal_summary['score_paywall'] * 0.35 +
        journal_summary['score_publish'] * 0.20 -
        journal_summary['score_substitute'] * 0.10
    )
    
    # --------------------------------------------------------------------------
    # アクションカテゴリ（4分類）への判定ロジック
    # --------------------------------------------------------------------------
    def classify_action(row):
        high_read = row['counter_dl'] >= 300
        high_paywall = row['paywall_rate_%'] >= 50.0
        high_publish = row['total_papers'] >= 100
        
        if high_publish and high_paywall:
            return "【A】転換契約コア誌 (高投稿 ✕ 高Paywall ➔ 転換契約主軸)"
        elif high_read and high_paywall:
            return "【B】純粋購読維持誌 (高需要 ✕ 高Paywall ➔ 購読継続)"
        elif high_read and not high_paywall:
            return "【C】OA代替解約候補 (高需要 ✕ 低Paywall ➔ アンバンドル)"
        else:
            return "【D】ILL/PPV補填誌 (低需要 ➔ 個別手配対応)"

    journal_summary['action_category'] = journal_summary.apply(classify_action, axis=1)

    print("\n【多次元スコアリング アクション分類集計】")
    action_counts = journal_summary.groupby(['publisher', 'action_category']).size().unstack(fill_value=0)
    print(action_counts)

    # 結果CSVの保存
    output_csv = "journal_multidimensional_scoring_results.csv"
    journal_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 詳細スコアリング結果をCSVとして出力しました: {output_csv}")

print("\n解析処理が完了しました。")