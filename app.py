import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 1. ページ基本設定 ＆ セッション管理
# ==============================================================================
st.set_page_config(page_title="9大学連合 スマートコンソーシアム統合シミュレーター v18", layout="wide")

if 'master_db' not in st.session_state:
    st.session_state.master_db = pd.DataFrame()
if 'yop_db' not in st.session_state:
    st.session_state.yop_db = pd.DataFrame()
if 'history_pts' not in st.session_state:
    st.session_state.history_pts = []

st.sidebar.title("🛡️ 9大学コンソーシアム パラメーター")

# ------------------------------------------------------------------------------
# 🔒 プレゼンテーション・機密保護モード
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔒 プレゼン用 機密保護・マスキング設定")
enable_masking = st.sidebar.checkbox("機密保護モード（仮名化＆数値マスク）を有効化", value=False)

if enable_masking:
    mask_tier = st.sidebar.checkbox("Tier分類を隠す (グループ化)", value=True)
    mask_scale = st.sidebar.slider("金額スケーリング（原数秘匿倍率）", 0.5, 2.0, 1.0, 0.1)
else:
    mask_tier = False
    mask_scale = 1.0

st.sidebar.markdown("---")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("📌 現在値を記録"):
    if 'current_cost' in st.session_state and st.session_state.current_cost > 0:
        st.session_state.history_pts.append({
            'cost': float(st.session_state.current_cost), 
            'oa': float(st.session_state.current_oa)
        })
        st.sidebar.success("記録しました")

if col_btn2.button("🧹 履歴リセット"):
    st.session_state.history_pts = []
    st.rerun()

# ==============================================================================
# 2. 安全データ浄化 ＆ マスキング補助関数
# ==============================================================================
def sanitize_list(vals):
    """NaN や Inf を除去して安全な Python float リストを返します"""
    clean_vals = []
    for v in vals:
        try:
            f = float(v)
            if np.isnan(f) or np.isinf(f):
                clean_vals.append(0.0)
            else:
                clean_vals.append(f)
        except Exception:
            clean_vals.append(0.0)
    return clean_vals

def find_pareto_front(x_vals, y_vals, x_mode='min', y_mode='max'):
    x = sanitize_list(x_vals)
    y = sanitize_list(y_vals)
    n = len(x)
    if n == 0:
        return []
    
    x_adj = [-v if x_mode == 'min' else v for v in x]
    y_adj = [-v if y_mode == 'min' else v for v in y]
    
    pareto_indices = []
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            if (x_adj[j] >= x_adj[i] and y_adj[j] >= y_adj[i]) and \
               (x_adj[j] > x_adj[i] or y_adj[j] > y_adj[i]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)
            
    return sorted(pareto_indices, key=lambda idx: x[idx])

def apply_masking_to_df(df_input, is_masked=False, hide_tier=False, scale_factor=1.0):
    """データ構造を壊さずに安全に大学名・Tier・金額を難読化・マスク処理します"""
    df_out = df_input.copy()
    if df_out.empty:
        return df_out

    if is_masked:
        if 'Entity' in df_out.columns:
            unique_entities = sorted(df_out['Entity'].unique())
            anon_map = {u: f"機関_{chr(65+i)}" for i, u in enumerate(unique_entities)}
            df_out['Entity'] = df_out['Entity'].map(anon_map)
        
        if hide_tier and 'Tier' in df_out.columns:
            df_out['Tier'] = '加盟機関'

        money_cols = ['Indiv_Cost', 'Cons_Cost', 'Win_Loss', 'Actual_Cost_Yen', 'Base_Sub_Oku', 'Win_Loss_per_FTE', 'Indiv_Cost_per_FTE', 'Cons_Cost_per_FTE']
        for col in money_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col] * scale_factor

    return df_out

# ==============================================================================
# 3. 9大学純粋実数値データ・ロード関数 (Turnaways自動同期版)
# ==============================================================================
@st.cache_data
def load_exact_real_data_v15(script_dir):
    data_dir = os.path.join(script_dir, "data")

    cost_fte_file = os.path.join(data_dir, "Consortium_9_Universities_Cost_per_FTE_Analysis.csv")
    yop_file = os.path.join(data_dir, "Consortium_9_Universities_Journal_YOP_Detail_Master.csv.gz")
    tsv_el = os.path.join(data_dir, "集計_elsevier.tsv")
    tsv_wl = os.path.join(data_dir, "集計_wiley.tsv")

    tier_mapping = {
        '東京大学': 'Tier1', '京都大学': 'Tier1', '大阪大学': 'Tier1', '九州大学': 'Tier1',
        '筑波大学': 'Tier2', '千葉大学': 'Tier2', '新潟大学': 'Tier2', '岡山大学': 'Tier2',
        '総合研究大学院大学': 'Tier3'
    }

    def get_exact_paper_counts(tsv_path):
        records = {}
        if not os.path.exists(tsv_path): return records
        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            i, current_univ = 0, None
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('====='):
                    i += 1
                    if i < len(lines):
                        cand = lines[i].strip()
                        if "全て" not in cand and cand != "": current_univ = cand
                    i += 1; continue
                parts = line.split('\t')
                if len(parts) > 1 and parts[0] in ['2021', '2022', '2023', '2024', '2025']:
                    y = int(parts[0])
                    try:
                        p_count = float(parts[1]) if parts[1].strip() else 0.0
                        records[(current_univ, y)] = p_count
                    except: pass
                i += 1
        except Exception:
            pass
        return records

    el_papers = get_exact_paper_counts(tsv_el)
    wl_papers = get_exact_paper_counts(tsv_wl)

    df_cost = pd.DataFrame()
    if os.path.exists(cost_fte_file):
        df_cost = pd.read_csv(cost_fte_file).fillna(0)
        df_cost['Tier'] = df_cost['Entity'].map(tier_mapping).fillna('Tier2')
        df_cost['Base_Sub_Oku'] = df_cost['Actual_Cost_Yen'] / 100000000.0
        df_cost['Access'] = df_cost['Total_Unique_Requests']
        
        def match_paper_count(row):
            key = (row['Entity'], int(row['Year']))
            if row['Publisher'] == 'Elsevier' and key in el_papers:
                return el_papers[key]
            elif row['Publisher'] == 'Wiley' and key in wl_papers:
                return wl_papers[key]
            return row['Access'] / 350.0 if row['Access'] > 0 else 300.0

        df_cost['Total_Pubs'] = df_cost.apply(match_paper_count, axis=1)
        df_cost['Tokens'] = df_cost.apply(lambda r: r['Total_Pubs'] * 0.8 if r['Tier'] == 'Tier1' else 0.0, axis=1)

        # 💡【修正点】yop_file から実際の Turnaways 数を自動抽出してマージする
        if os.path.exists(yop_file):
            try:
                df_yop_raw = pd.read_csv(yop_file).fillna(0)
                yop_turn = df_yop_raw.groupby(['Entity', 'Data_Year', 'Publisher'])['Total_Turnaways'].sum().reset_index()
                yop_turn.rename(columns={'Data_Year': 'Year', 'Total_Turnaways': 'YOP_Turnaways'}, inplace=True)
                
                df_cost = pd.merge(df_cost, yop_turn, on=['Entity', 'Year', 'Publisher'], how='left').fillna({'YOP_Turnaways': 0.0})
                
                df_cost['No_License'] = df_cost['YOP_Turnaways'] * 0.85
                df_cost['Limit_Exceeded'] = df_cost['YOP_Turnaways'] * 0.15
            except Exception:
                if 'No_License' not in df_cost.columns: df_cost['No_License'] = 0.0
                if 'Limit_Exceeded' not in df_cost.columns: df_cost['Limit_Exceeded'] = 0.0
        else:
            if 'No_License' not in df_cost.columns: df_cost['No_License'] = 0.0
            if 'Limit_Exceeded' not in df_cost.columns: df_cost['Limit_Exceeded'] = 0.0

        def resolve_contracted_titles(row):
            val = row.get('Contracted_Titles', 0)
            if pd.notna(val) and float(val) > 0:
                return float(val)
            pub = row.get('Publisher', 'Elsevier')
            tier = row.get('Tier', 'Tier2')
            if pub == 'Elsevier':
                return 2200.0 if tier == 'Tier1' else (1400.0 if tier == 'Tier2' else 450.0)
            else:
                return 1400.0 if tier == 'Tier1' else (900.0 if tier == 'Tier2' else 300.0)

        df_cost['Contracted_Titles'] = df_cost.apply(resolve_contracted_titles, axis=1)

    df_yop = pd.DataFrame()
    if os.path.exists(yop_file):
        df_yop = pd.read_csv(yop_file).fillna(0)

    return df_cost, df_yop

def calculate_gini(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0 or np.sum(x) == 0: return 0.0
    n = len(x)
    diff_sum = np.sum(np.abs(x[:, None] - x))
    return float(diff_sum / (2 * n * np.sum(x)))

# ==============================================================================
# 4. シミュレーション計算エンジン
# ==============================================================================
def run_simulation_v15(params, base_df, pub_type, target_year):
    UNIT_APC_INDIV = params['list_apc_price'] / 10000.0
    
    working_df = base_df[(base_df['Publisher'] == pub_type) & (base_df['Year'] == target_year)].copy()

    if working_df.empty and not base_df.empty:
        fallback_year = 2023 if 2023 in base_df['Year'].values else base_df['Year'].max()
        working_df = base_df[(base_df['Publisher'] == pub_type) & (base_df['Year'] == fallback_year)].copy()
        if not working_df.empty:
            working_df['Year'] = target_year

    if working_df.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pd.DataFrame()

    num_cols = ['Access', 'Total_Pubs', 'Base_Sub_Oku', 'FTE_Total', 'Actual_Cost_Yen', 'Tokens', 'ILL_Risk_Potential', 'No_License', 'Limit_Exceeded', 'Contracted_Titles']
    for col in num_cols:
        if col not in working_df.columns:
            working_df[col] = 0.0
        working_df[col] = pd.to_numeric(working_df[col], errors='coerce').fillna(0.0).astype(float)

    green_r = params['green_oa_rate'] / 100.0
    unbundle_r = params['unbundle_rate']
    pca_r = params['pca_cover_rate'] / 100.0

    working_df['Indiv_Cost'] = working_df['Base_Sub_Oku']
    working_df['Indiv_Cost_Yen'] = working_df['Actual_Cost_Yen']
    working_df['Indiv_Cost_per_FTE'] = np.where(working_df['FTE_Total'] > 0, working_df['Indiv_Cost_Yen'] / working_df['FTE_Total'], 0.0)

    working_df['Green_OA_Pubs'] = working_df['Total_Pubs'] * green_r
    working_df['Gold_OA_Pubs'] = np.maximum(0.0, working_df['Total_Pubs'] * (0.65 - green_r))
    working_df['Total_OA_Pubs'] = working_df['Green_OA_Pubs'] + working_df['Gold_OA_Pubs']

    negotiated_apc = UNIT_APC_INDIV * (params['target_apc_price'] / params['list_apc_price'])
    
    total_cons_sub = working_df['Base_Sub_Oku'].sum() * (1.0 - unbundle_r)
    total_cons_apc = working_df['Gold_OA_Pubs'].sum() * negotiated_apc

    working_df['Est_No_License'] = working_df['No_License']
    working_df['Est_Limit_Exceeded'] = working_df['Limit_Exceeded']
    
    total_no_license = float(working_df['Est_No_License'].sum())
    total_limit_exceeded = float(working_df['Est_Limit_Exceeded'].sum())

    base_ill_demand = working_df['ILL_Risk_Potential'].sum()
    if base_ill_demand <= 0:
        base_ill_demand = working_df['Access'].sum() * unbundle_r * params['req_rate']
    else:
        base_ill_demand = base_ill_demand * unbundle_r

    reqs_after_pca = base_ill_demand * (1.0 - pca_r)
    actual_reqs = reqs_after_pca * (1.0 - params['backfile_rate'])
    
    ill_count = actual_reqs * params['ill_cover_rate']
    ppv_count = actual_reqs * (1.0 - params['ill_cover_rate'])

    total_ill_cost = (ill_count * params['smart_ill_unit_cost']) / 100000000.0
    total_ppv_cost = (ppv_count * params['ppv_unit_price']) / 100000000.0
    annual_fund_cost = params['fund_investment'] / 10.0

    total_cons_cost = total_cons_sub + total_cons_apc + total_ill_cost + total_ppv_cost + annual_fund_cost

    w = params['read_weight']
    acc_sum = working_df['Access'].sum()
    pub_sum = working_df['Total_Pubs'].sum()
    acc_share = working_df['Access'] / acc_sum if acc_sum > 0 else 0
    pub_share = working_df['Total_Pubs'] / pub_sum if pub_sum > 0 else 0
    
    working_df['Cons_Cost'] = total_cons_cost * (w * acc_share + (1.0 - w) * pub_share)
    working_df['Cons_Cost_Yen'] = working_df['Cons_Cost'] * 100000000.0
    working_df['Cons_Cost_per_FTE'] = np.where(working_df['FTE_Total'] > 0, working_df['Cons_Cost_Yen'] / working_df['FTE_Total'], 0.0)

    working_df['Win_Loss'] = working_df['Indiv_Cost'] - working_df['Cons_Cost']
    working_df['Win_Loss_per_FTE'] = working_df['Indiv_Cost_per_FTE'] - working_df['Cons_Cost_per_FTE']
    working_df['ROI'] = np.where(working_df['Cons_Cost'] > 0, working_df['Total_OA_Pubs'] / working_df['Cons_Cost'], 0.0)
    
    oa_rate_raw = np.where(working_df['Total_Pubs'] > 0, (working_df['Total_OA_Pubs'] / working_df['Total_Pubs'] * 100.0), 0.0)
    working_df['OA_Rate'] = np.clip(oa_rate_raw, 0.0, 100.0)

    working_df = working_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(total_cons_cost), float(working_df['Total_OA_Pubs'].sum()), float(total_cons_sub), float(total_cons_apc), float(total_ill_cost), float(total_ppv_cost), float(annual_fund_cost), float(ill_count), total_no_license, total_limit_exceeded, working_df

# ==============================================================================
# 5. パレート探査 ＆ モンテカルロ試算エンジン
# ==============================================================================
def compute_actual_pareto_sweep_v15(params_dict, base_df, pub_type, target_year):
    sweep_results = []
    unb_steps = np.linspace(0.0, 0.75, 12)
    pca_steps = np.linspace(0.10, 0.90, 8)
    
    current_p = params_dict.copy()
    
    working_base = base_df[(base_df['Publisher'] == pub_type) & (base_df['Year'] == target_year)]
    if working_base.empty and not base_df.empty:
        fallback_year = 2023 if 2023 in base_df['Year'].values else base_df['Year'].max()
        working_base = base_df[(base_df['Publisher'] == pub_type) & (base_df['Year'] == fallback_year)]

    total_fte = float(working_base['FTE_Total'].sum()) if 'FTE_Total' in working_base.columns else 0.0
    total_pubs = float(working_base['Total_Pubs'].sum()) if 'Total_Pubs' in working_base.columns else 0.0
    baseline_cost = float(working_base['Base_Sub_Oku'].sum()) if 'Base_Sub_Oku' in working_base.columns else 0.0

    for u in unb_steps:
        for p in pca_steps:
            current_p['unbundle_rate'] = float(u)
            current_p['pca_cover_rate'] = float(p * 100.0)
            
            c, oa, _, _, _, _, _, _, no_lic, lim_exc, _ = run_simulation_v15(
                current_p, base_df, pub_type, target_year
            )
            if c > 0:
                savings = baseline_cost - c
                cost_per_fte = (c * 1e8 / total_fte) if total_fte > 0 else 0.0
                savings_per_fte = (savings * 1e8 / total_fte) if total_fte > 0 else 0.0
                oa_rate = min(100.0, (oa / total_pubs * 100.0)) if total_pubs > 0 else 0.0
                roi = (oa / c) if c > 0 else 0.0

                sweep_results.append({
                    'cost': float(c),
                    'oa': float(oa),
                    'savings': float(savings),
                    'cost_per_fte': float(cost_per_fte),
                    'savings_per_fte': float(savings_per_fte),
                    'oa_rate': float(oa_rate),
                    'roi': float(roi),
                    'no_license': float(no_lic),
                    'limit_exceeded': float(lim_exc),
                    'unbundle_rate': float(u),
                    'pca_cover_rate': float(p * 100.0)
                })
                
    df_res = pd.DataFrame(sweep_results)
    if not df_res.empty:
        df_res = df_res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_res

def run_monte_carlo_simulation_v15(base_df, pub_type, target_year, mc_params, n_trials=3000):
    working_df = base_df[(base_df['Publisher'] == pub_type) & (base_df['Year'] == target_year)]
    if working_df.empty and not base_df.empty:
        fallback_year = 2023 if 2023 in base_df['Year'].values else base_df['Year'].max()
        working_df = base_df[(base_df['Publisher'] == pub_type) & (base_df['Year'] == fallback_year)]

    if working_df.empty:
        return pd.DataFrame()

    base_sub = float(working_df['Base_Sub_Oku'].sum())
    total_pubs = float(working_df['Total_Pubs'].sum())
    total_access = float(working_df['Access'].sum())
    total_ill_risk = float(working_df['ILL_Risk_Potential'].sum())

    unb_vec = np.random.uniform(mc_params['unb_min']/100.0, mc_params['unb_max']/100.0, n_trials)
    pca_vec = np.random.uniform(mc_params['pca_min']/100.0, mc_params['pca_max']/100.0, n_trials)
    req_vec = np.random.uniform(mc_params['req_min']/100.0, mc_params['req_max']/100.0, n_trials)
    apc_vec = np.random.uniform(mc_params['apc_min'], mc_params['apc_max'], n_trials) / 10000.0

    green_r = mc_params['green_oa'] / 100.0
    gold_pubs = np.maximum(0.0, total_pubs * (0.65 - green_r))
    total_oa_pubs = total_pubs * 0.65

    sub_cost_vec = base_sub * (1.0 - unb_vec)
    apc_cost_vec = gold_pubs * apc_vec

    if total_ill_risk <= 0:
        base_ill_demand_vec = total_access * unb_vec * req_vec
    else:
        base_ill_demand_vec = total_ill_risk * unb_vec

    actual_reqs_vec = base_ill_demand_vec * (1.0 - pca_vec) * (1.0 - mc_params['bf_rate']/100.0)
    ill_count_vec = actual_reqs_vec * (mc_params['ill_rate']/100.0)
    ppv_count_vec = actual_reqs_vec * (1.0 - mc_params['ill_rate']/100.0)

    ill_cost_vec = (ill_count_vec * mc_params['ill_unit_cost']) / 1e8
    ppv_cost_vec = (ppv_count_vec * 4000) / 1e8
    fund_cost = mc_params['fund_investment'] / 10.0

    total_cost_vec = sub_cost_vec + apc_cost_vec + ill_cost_vec + ppv_cost_vec + fund_cost
    savings_vec = base_sub - total_cost_vec
    roi_vec = np.where(total_cost_vec > 0, total_oa_pubs / total_cost_vec, 0.0)

    no_license_val = float(working_df['No_License'].sum()) if 'No_License' in working_df.columns else 0.0
    limit_exceeded_val = float(working_df['Limit_Exceeded'].sum()) if 'Limit_Exceeded' in working_df.columns else 0.0

    df_mc = pd.DataFrame({
        'trial': np.arange(n_trials),
        'unbundle_rate': unb_vec * 100.0,
        'pca_rate': pca_vec * 100.0,
        'req_rate': req_vec * 100.0,
        'apc_price_man': apc_vec * 10000.0,
        'total_cost': total_cost_vec,
        'savings': savings_vec,
        'total_oa_pubs': np.full(n_trials, total_oa_pubs),
        'roi': roi_vec,
        'no_license': np.full(n_trials, no_license_val),
        'limit_exceeded': np.full(n_trials, limit_exceeded_val)
    })
    return df_mc.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ==============================================================================
# 6. UI設定 & 計算実行
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
real_db, real_yop_db = load_exact_real_data_v15(SCRIPT_DIR)

if not real_db.empty:
    st.sidebar.success("✅ 9大学実数値データをロード完了")
    st.session_state.master_db = real_db
    st.session_state.yop_db = real_yop_db
else:
    st.sidebar.error("❌ processed_data 内に実測CSVが見つかりません。")

p_type = st.sidebar.selectbox("対象出版社", ["Elsevier", "Wiley"])
t_year = st.sidebar.selectbox("分析対象年", [2024, 2023, 2025, 2022, 2021], index=0)

g_oa = st.sidebar.slider("グリーンOA率 (%)", 0, 50, 25)
unb = st.sidebar.slider("購読削減 (Unbundle) 率", 0.0, 1.0, 0.45)
w_read = st.sidebar.slider("按分重み (利用 1.0 ↔ 出版 0.0)", 0.0, 1.0, 0.5)

st.sidebar.divider()
st.sidebar.subheader("🏦 過去契約(PCA) & バックファイル")
pca_rate = st.sidebar.slider("過去契約 (PCA) カバー率 (%)", 0, 100, 55)
fund_inv = st.sidebar.number_input("永久アクセス基金 (億円)", value=10)
bf_rate = st.sidebar.slider("バックファイル一括購入率 (%)", 0, 100, 20)

st.sidebar.divider()
st.sidebar.subheader("⚡ スマートILL / PPV需要")
req_r = st.sidebar.slider("非保持論文リクエスト率 (%)", 1, 30, 5)
ill_r = st.sidebar.slider("スマートILLカバー率 (%)", 0, 100, 85)
ill_u = st.sidebar.number_input("スマートILL単価 (円/件)", value=500)

params = {
    'pub_type': p_type, 'target_year': t_year, 'green_oa_rate': g_oa, 'unbundle_rate': unb, 'read_weight': w_read,
    'pca_cover_rate': pca_rate, 'fund_investment': fund_inv, 'backfile_rate': bf_rate / 100.0,
    'smart_ill_unit_cost': ill_u, 'ill_cover_rate': ill_r / 100.0, 'req_rate': req_r / 100.0,
    'ppv_unit_price': 4000, 'list_apc_price': 45, 'target_apc_price': 30
}

total_cost, total_oa, sub_c, apc_c, ill_c, ppv_c, fund_c, ill_n, total_no_lic, total_lim_exc, df_final = run_simulation_v15(
    params, st.session_state.master_db, p_type, t_year
)

df_final_display = apply_masking_to_df(
    df_final, 
    is_masked=enable_masking, 
    hide_tier=mask_tier, 
    scale_factor=mask_scale
)

st.session_state.current_cost = total_cost
st.session_state.current_oa = total_oa

# ==============================================================================
# 7. メイン画面表示 (全6メニュー構成)
# ==============================================================================
st.title("🏛️ 9大学連合 スマートコンソーシアム統合戦略シミュレーター")

if enable_masking:
    st.warning("🔒 プレゼンテーション・機密保護モードが有効です（大学名仮名化＆スケーリング適用中）。")

if df_final.empty:
    st.error("⚠️ 該当データが存在しません。サイドバーで対象年度・出版社を確認してください。")
else:
    mode = st.sidebar.radio("機能メニュー", [
        "📈 戦略ダッシュボード", 
        "📊 TR_J4 YOP年代別利用分析", 
        "⚖️ 9大学実金額 ＆ FTE平準化評価", 
        "🔄 TAトークン循環(Sankey)", 
        "🎲 モンテカルロ多目的最適化", 
        "💾 実データ一覧・出力"
    ])

    # --------------------------------------------------------------------------
    # メニュー 1: 📈 戦略ダッシュボード
    # --------------------------------------------------------------------------
    if mode == "📈 戦略ダッシュボード":
        st.header(f"🚀 {p_type} ({t_year}年) 財務・交渉戦略ダッシュボード")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("コンソーシアム総コスト", f"{total_cost * mask_scale:.2f} 億円", delta=f"現状比 {-(df_final['Win_Loss'].sum() * mask_scale):.2f} 億円")
        m2.metric("総OA論文創出数", f"{total_oa:,.0f} 本")
        m3.metric("No_License (未契約拒否)", f"{total_no_lic:,.0f} 件")
        m4.metric("Limit_Exceeded (同時超過)", f"{total_lim_exc:,.0f} 件")
        m5.metric("ILL+PPV補填コスト", f"{((ill_c + ppv_c) * mask_scale):.2f} 億円")

        st.divider()
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.subheader("🎯 パレート・フロンティア（多角的多基準分析 ＆ アクセス拒否指標同調表示）")
            
            metric_mode = st.selectbox(
                "📊 分析に使用する評価指標の組み合わせ（軸）を選択してください:",
                [
                    "【基本軸】 総コスト (億円) ✕ OA創出本数 (本)",
                    "【削減軸】 コスト削減額 (億円) ✕ OA創出本数 (本)",
                    "【FTE軸】 FTE一人当たり削減額 (円/人) ✕ OA率 (%)",
                    "【効率軸】 総コスト (億円) ✕ 投資効率 ROI (本/億円)"
                ]
            )

            if "【削減軸】" in metric_mode:
                x_col, x_label, x_mode = 'savings', '削減額 (億円)', 'max'
                y_col, y_label, y_mode = 'oa', 'OA論文創出数 (本)', 'max'
            elif "【FTE軸】" in metric_mode:
                x_col, x_label, x_mode = 'savings_per_fte', 'FTE一人当たり削減額 (円/人)', 'max'
                y_col, y_label, y_mode = 'oa_rate', 'OA率 (%)', 'max'
            elif "【効率軸】" in metric_mode:
                x_col, x_label, x_mode = 'cost', '総コスト (億円)', 'min'
                y_col, y_label, y_mode = 'roi', '投資効率 ROI (本/億円)', 'max'
            else:
                x_col, x_label, x_mode = 'cost', '総コスト (億円)', 'min'
                y_col, y_label, y_mode = 'oa', 'OA論文創出数 (本)', 'max'

            sweep_df = compute_actual_pareto_sweep_v15(params, st.session_state.master_db, p_type, t_year)
            
            baseline_cost = float(df_final['Indiv_Cost'].sum())
            curr_savings = baseline_cost - total_cost
            total_fte = float(df_final['FTE_Total'].sum())
            total_pubs = float(df_final['Total_Pubs'].sum())
            curr_savings_per_fte = (curr_savings * 1e8 / total_fte) if total_fte > 0 else 0.0
            curr_oa_rate = min(100.0, (total_oa / total_pubs * 100.0)) if total_pubs > 0 else 0.0
            curr_roi = (total_oa / total_cost) if total_cost > 0 else 0.0

            curr_point = {
                'cost': float(total_cost * mask_scale),
                'oa': float(total_oa),
                'savings': float(curr_savings * mask_scale),
                'savings_per_fte': float(curr_savings_per_fte * mask_scale),
                'oa_rate': float(curr_oa_rate),
                'roi': float(curr_roi)
            }

            try:
                fig = go.Figure()
                
                if not sweep_df.empty and x_col in sweep_df.columns and y_col in sweep_df.columns:
                    p_indices = find_pareto_front(
                        sweep_df[x_col].values, sweep_df[y_col].values, 
                        x_mode=x_mode, y_mode=y_mode
                    )
                    pareto_df = sweep_df.iloc[p_indices].copy().reset_index(drop=True)

                    x_vals_plot = (sweep_df[x_col] * mask_scale) if 'cost' in x_col or 'savings' in x_col else sweep_df[x_col]
                    y_vals_plot = (sweep_df[y_col] * mask_scale) if 'cost' in y_col or 'savings' in y_col else sweep_df[y_col]

                    fig.add_trace(go.Scatter(
                        x=sanitize_list(x_vals_plot.tolist()),
                        y=sanitize_list(y_vals_plot.tolist()),
                        mode='markers', name='試算シナリオ空間',
                        marker=dict(color='lightgray', size=5, opacity=0.4)
                    ))
                
                    if not pareto_df.empty:
                        px_plot = (pareto_df[x_col] * mask_scale) if 'cost' in x_col or 'savings' in x_col else pareto_df[x_col]
                        py_plot = (pareto_df[y_col] * mask_scale) if 'cost' in y_col or 'savings' in y_col else pareto_df[y_col]

                        fig.add_trace(go.Scatter(
                            x=sanitize_list(px_plot.tolist()),
                            y=sanitize_list(py_plot.tolist()),
                            mode='lines+markers', name='選択指標のパレート最適限界',
                            line=dict(color='#d62728', width=3),
                            marker=dict(color='#d62728', size=8, symbol='circle')
                        ))
                
                fig.add_trace(go.Scatter(
                    x=sanitize_list([curr_point[x_col]]),
                    y=sanitize_list([curr_point[y_col]]),
                    mode='markers', name='現在の設定シナリオ',
                    marker=dict(color='#1f77b4', size=16, symbol='star')
                ))
                
                fig.update_layout(
                    xaxis=dict(title=x_label),
                    yaxis=dict(title=y_label),
                    height=420,
                    legend=dict(orientation="h", y=1.15),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as ex_fig:
                st.error(f"⚠️ パレート図描画中に例外が発生しました: {ex_fig}")

        with col_r:
            st.subheader("💰 支出構造内訳 ＆ 拒否件数内訳")
            pie_vals = sanitize_list([sub_c * mask_scale, apc_c * mask_scale, ill_c * mask_scale, ppv_c * mask_scale, fund_c * mask_scale])
            if sum(pie_vals) > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['購読費 (Read)', 'APC包括費 (Publish)', 'スマートILL', 'PPV従量', '基金投資'],
                    values=pie_vals, hole=.4,
                    marker_colors=['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
                )])
                fig_pie.update_layout(height=260, legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            fig_turn = go.Figure()
            fig_turn.add_trace(go.Bar(name='No_License (未契約拒否)', x=['全校合計'], y=[total_no_lic], marker_color='#f39c12'))
            fig_turn.add_trace(go.Bar(name='Limit_Exceeded (同時超過)', x=['全校合計'], y=[total_lim_exc], marker_color='#e67e22'))
            fig_turn.update_layout(barmode='stack', title="COUNTERアクセス拒否内訳", height=180, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
            st.plotly_chart(fig_turn, use_container_width=True)

    # --------------------------------------------------------------------------
    # メニュー 2: 📊 TR_J4 YOP年代別利用分析 (契約タイトル数対比統合版)
    # --------------------------------------------------------------------------
    elif mode == "📊 TR_J4 YOP年代別利用分析":
        st.header("📊 事前解析：TR_J4に基づく出版年代別(YOP)アクセス構造 ＆ 契約タイトル対比分析")
        st.markdown("""
        2020年以前（過去アーカイブ）のアクセス比率や、年代別の Turnaway（アクセス拒否）発生割合を評価します。  
        **契約タイトル数（購読誌数）と Turnaway の関係** を並べて評価することで、現在の契約パッケージと実際の研究需要のミスマッチを特定できます。
        """)

        if not st.session_state.yop_db.empty:
            yop_sub = st.session_state.yop_db[
                (st.session_state.yop_db['Publisher'] == p_type) & 
                (st.session_state.yop_db['Data_Year'] == t_year)
            ].fillna(0).copy()
            
            if not yop_sub.empty:
                eras = ['2020s', '2010s', '2000s', '1990s_or_before']
                turn_cols = [f'Turnaways_{era}' for era in eras]
                
                for col in turn_cols:
                    if col not in yop_sub.columns:
                        yop_sub[col] = 0.0

                for idx, row in yop_sub.iterrows():
                    tot_turn = row.get('Total_Turnaways', 0.0)
                    t_2010s_sum = row['Turnaways_2010s'] + row['Turnaways_2000s'] + row['Turnaways_1990s_or_before']
                    
                    if tot_turn > 0 and t_2010s_sum == 0:
                        tot_req = row['Total_Requests']
                        if tot_req > 0:
                            yop_sub.at[idx, 'Turnaways_2020s'] = tot_turn * (row['Requests_2020s'] / tot_req)
                            yop_sub.at[idx, 'Turnaways_2010s'] = tot_turn * (row['Requests_2010s'] / tot_req)
                            yop_sub.at[idx, 'Turnaways_2000s'] = tot_turn * (row['Requests_2000s'] / tot_req)
                            yop_sub.at[idx, 'Turnaways_1990s_or_before'] = tot_turn * (row['Requests_1990s_or_before'] / tot_req)

                agg_dict = {f'Requests_{era}': 'sum' for era in eras}
                agg_dict.update({f'Turnaways_{era}': 'sum' for era in eras})
                
                yop_summary = yop_sub.groupby('Entity').agg(agg_dict).reset_index()

                yop_summary = apply_masking_to_df(yop_summary, is_masked=enable_masking, hide_tier=mask_tier)

                master_sub = df_final_display[['Entity', 'Contracted_Titles']].drop_duplicates(subset=['Entity'])
                yop_summary = pd.merge(yop_summary, master_sub, on='Entity', how='left').fillna(0)

                yop_summary['Total_Requests'] = sum(yop_summary[f'Requests_{era}'] for era in eras)
                yop_summary['Total_Turnaways'] = sum(yop_summary[f'Turnaways_{era}'] for era in eras)

                for era in eras:
                    req_col = f'Requests_{era}'
                    turn_col = f'Turnaways_{era}'
                    rate_col = f'Turnaway_Rate_{era}'
                    
                    total_attempt = yop_summary[req_col] + yop_summary[turn_col]
                    yop_summary[rate_col] = np.where(total_attempt > 0, (yop_summary[turn_col] / total_attempt) * 100.0, 0.0)

                def format_title_label(r):
                    titles = int(r['Contracted_Titles'])
                    if titles <= 0:
                        titles = 2200 if p_type == 'Elsevier' else 1400
                    return f"{r['Entity']}<br>({titles:,}誌)"

                yop_summary['Entity_With_Titles'] = yop_summary.apply(format_title_label, axis=1)

                tab_yop1, tab_yop2 = st.tabs(["📈 機関別 年代構造 ＆ Turnaway実数 (契約タイトル数付き)", "🎯 年代別 Turnaway 実発生率 (%)"])

                with tab_yop1:
                    fig_yop = go.Figure()
                    
                    fig_yop.add_trace(go.Bar(name='2020年代 (アクセス)', x=yop_summary['Entity_With_Titles'], y=sanitize_list(yop_summary['Requests_2020s'].tolist()), marker_color='#2c3e50'))
                    fig_yop.add_trace(go.Bar(name='2010年代 (アクセス)', x=yop_summary['Entity_With_Titles'], y=sanitize_list(yop_summary['Requests_2010s'].tolist()), marker_color='#34495e'))
                    fig_yop.add_trace(go.Bar(name='2000年代 (アクセス)', x=yop_summary['Entity_With_Titles'], y=sanitize_list(yop_summary['Requests_2000s'].tolist()), marker_color='#7f8c8d'))
                    fig_yop.add_trace(go.Bar(name='1990年代以前 (アクセス)', x=yop_summary['Entity_With_Titles'], y=sanitize_list(yop_summary['Requests_1990s_or_before'].tolist()), marker_color='#bdc3c7'))

                    fig_yop.add_trace(go.Bar(name='Turnaway 2020年代', x=yop_summary['Entity_With_Titles'], y=sanitize_list(yop_summary['Turnaways_2020s'].tolist()), marker_color='#e74c3c', opacity=0.85))
                    fig_yop.add_trace(go.Bar(name='Turnaway 2010年代以前', x=yop_summary['Entity_With_Titles'], y=sanitize_list((yop_summary['Turnaways_2010s']+yop_summary['Turnaways_2000s']+yop_summary['Turnaways_1990s_or_before']).tolist()), marker_color='#e67e22', opacity=0.85))

                    fig_yop.update_layout(
                        barmode='stack', 
                        title=f"機関別 契約タイトル数 ✕ YOP（出版年）アクセス・Turnaway発生数 ({p_type} {t_year}年)", 
                        xaxis_title="機関名（下段：現在の契約タイトル数）",
                        height=540, 
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_yop, use_container_width=True)

                with tab_yop2:
                    era_labels = {'2020s': '2020年代 (最新誌)', '2010s': '2010年代', '2000s': '2000年代', '1990s_or_before': '1990年代以前'}
                    
                    fig_rates = go.Figure()
                    for era, label_name in era_labels.items():
                        fig_rates.add_trace(go.Bar(
                            name=label_name,
                            x=yop_summary['Entity_With_Titles'],
                            y=sanitize_list(yop_summary[f'Turnaway_Rate_{era}'].tolist()),
                            text=[f"{v:.1f}%" for v in yop_summary[f'Turnaway_Rate_{era}']],
                            textposition='auto'
                        ))

                    max_rate = yop_summary[[f'Turnaway_Rate_{e}' for e in era_labels.keys()]].max().max()
                    fig_rates.update_layout(
                        barmode='group',
                        title=f"機関別・年代別 Turnaway 実発生率 (%) 比較 ({p_type} {t_year}年)",
                        xaxis_title="機関名（下段：現在の契約タイトル数）",
                        yaxis=dict(title="Turnaway 発生率 (%)", range=[0, max(25, max_rate * 1.25)]),
                        height=500,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_rates, use_container_width=True)

                st.subheader("📋 機関別 契約タイトル数 ✕ YOP アクセス ＆ Turnaway 詳細対比一覧")
                yop_disp = yop_summary.rename(columns={
                    'Entity': '大学名',
                    'Contracted_Titles': '契約タイトル数(誌)',
                    'Requests_2020s': '2020s DL',
                    'Turnaways_2020s': '2020s 拒否',
                    'Turnaway_Rate_2020s': '2020s 拒否率(%)',
                    'Requests_2010s': '2010s DL',
                    'Turnaways_2010s': '2010s 拒否',
                    'Turnaway_Rate_2010s': '2010s 拒否率(%)',
                    'Total_Requests': '総DL数',
                    'Total_Turnaways': '総拒否数'
                })

                disp_cols = ['大学名', '契約タイトル数(誌)', '総DL数', '総拒否数', '2020s DL', '2020s 拒否', '2020s 拒否率(%)', '2010s DL', '2010s 拒否', '2010s 拒否率(%)']
                st.dataframe(yop_disp[disp_cols].style.format({
                    '契約タイトル数(誌)': '{:,.0f}',
                    '総DL数': '{:,.0f}', '総拒否数': '{:,.0f}',
                    '2020s DL': '{:,.0f}', '2020s 拒否': '{:,.0f}', '2020s 拒否率(%)': '{:.1f}%',
                    '2010s DL': '{:,.0f}', '2010s 拒否': '{:,.0f}', '2010s 拒否率(%)': '{:.1f}%'
                }), use_container_width=True)

            else:
                st.warning(f"{t_year}年のYOP詳細データが存在しません。")

    # --------------------------------------------------------------------------
    # メニュー 3: ⚖️ 9大学実金額 ＆ FTE平準化評価
    # --------------------------------------------------------------------------
    elif mode == "⚖️ 9大学実金額 ＆ FTE平準化評価":
        st.header("⚖️ 9大学実契約金額 ＆ FTE（研究者数）平準化評価")
        
        gini = calculate_gini(df_final_display['ROI'])
        c_g1, c_g2 = st.columns(2)
        c_g1.metric("投資効率 (ROI) ジニ係数", f"{gini:.3f}", delta="0に近いほど加盟大学間で公平")
        c_g2.info("※FTE（研究者人口）で割ることで、大学の規模に依存しない実質的なコスト削減インパクト（円/人）を評価できます。")

        cl1, cl2 = st.columns(2)
        with cl1:
            fig_wl = px.bar(df_final_display, x='Entity', y=['Indiv_Cost', 'Cons_Cost'], barmode='group',
                            title="現状の実契約額 vs 法人化後コスト (金額軸/校)",
                            labels={'value': '金額', 'variable': '区分'},
                            color_discrete_sequence=['#d62728', '#1f77b4'])
            st.plotly_chart(fig_wl, use_container_width=True)

        with cl2:
            fig_fte = px.bar(df_final_display, x='Entity', y='Win_Loss_per_FTE', color='Tier',
                             title="FTE一人当たりの年間削減額 (円 / 人)",
                             color_discrete_map={'Tier1': '#1f77b4', 'Tier2': '#2ca02c', 'Tier3': '#ff7f0e', '加盟機関': '#1f77b4'})
            fig_fte.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_fte, use_container_width=True)

        st.subheader("📋 機関別データ詳細一覧（契約タイトル数・COUNTER拒否内訳含む）")
        f_dict = {
            'Actual_Cost_Yen': "{:,.0f}", 'Indiv_Cost': "{:.3f}", 'Cons_Cost': "{:.3f}", 
            'Win_Loss': "{:.3f}", 'Indiv_Cost_per_FTE': "{:,.0f}", 'Cons_Cost_per_FTE': "{:,.0f}", 
            'Win_Loss_per_FTE': "{:,.0f}", 'Est_No_License': "{:,.0f}", 'Est_Limit_Exceeded': "{:,.0f}",
            'Contracted_Titles': "{:,.0f}"
        }
        show_cols = ['Entity', 'Tier', 'FTE_Total', 'Contracted_Titles', 'Total_Pubs', 'Est_No_License', 'Est_Limit_Exceeded', 'Actual_Cost_Yen', 'Indiv_Cost', 'Cons_Cost', 'Win_Loss', 'Indiv_Cost_per_FTE', 'Cons_Cost_per_FTE', 'Win_Loss_per_FTE']
        st.dataframe(df_final_display[show_cols].style.format(f_dict), use_container_width=True)

    # --------------------------------------------------------------------------
    # メニュー 4: 🔄 TAトークン循環(Sankey)
    # --------------------------------------------------------------------------
    elif mode == "🔄 TAトークン循環(Sankey)":
        st.header("🔄 転換契約 (TA) トークン・プールと動的融通フロー")
        st.markdown("大規模校（Tier 1）で使い切れなかった出版枠（トークン）を共通プールに回収し、投稿超過が発生する大学へ流動させる動的最適化モデルです。")

        t1_tokens = float(df_final[df_final['Tier'] == 'Tier1']['Tokens'].sum())
        t1_used = float(df_final[df_final['Tier'] == 'Tier1']['Gold_OA_Pubs'].sum())
        t1_excess = max(0.0, t1_tokens - t1_used)
        
        t2_req = float(df_final[df_final['Tier'] == 'Tier2']['Gold_OA_Pubs'].sum())
        t3_req = float(df_final[df_final['Tier'] == 'Tier3']['Gold_OA_Pubs'].sum())
        total_req = max(1.0, t2_req + t3_req)

        flow_t2 = float(t1_excess * (t2_req / total_req))
        flow_t3 = float(t1_excess * (t3_req / total_req))

        fig_s = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20, thickness=25,
                label=["Tier1 (大規模研究校) 余剰枠", "動的配分プール", "Tier2 (中規模校) 充当", "Tier3 (専門機関) 充当"] if mask_tier else ["Tier1 (東大・京大・阪大・九大) 余剰枠", "9大学連合動的プール", "Tier2 (筑波・千葉・新潟・岡山) 充当", "Tier3 (専門機関) 充当"],
                color=["#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e"]
            ),
            link=dict(
                source=[0, 1, 1],
                target=[1, 2, 3],
                value=[max(0.1, t1_excess), max(0.05, flow_t2), max(0.05, flow_t3)],
                color=["rgba(31,119,180,0.3)", "rgba(44,160,44,0.3)", "rgba(255,127,14,0.3)"]
            )
        )])
        fig_s.update_layout(height=450, font=dict(size=13))
        st.plotly_chart(fig_s, use_container_width=True)

    # --------------------------------------------------------------------------
    # メニュー 5: 🎲 モンテカルロ多目的最適化
    # --------------------------------------------------------------------------
    elif mode == "🎲 モンテカルロ多目的最適化":
        st.header("🎲 モンテカルロ確率的・多目的最適化 ＆ リスク解析")
        st.markdown("交渉における不確実性（購読削減率の妥結幅、リクエスト率のブレ、APC価格など）を確率分布として設定し、**不確実性下でのリスク・パレート最適解・感度分析**を行います。")

        st.divider()
        mc_c1, mc_c2 = st.columns(2)

        with mc_c1:
            st.subheader("⚙️ 不確実性パラメータの変動範囲 (一様分布)")
            n_trials = st.select_slider("試行サンプル数 (N)", options=[1000, 3000, 5000, 10000], value=3000)
            unb_range = st.slider("購読削減率 (Unbundle) 変動範囲 (%)", 0, 80, (25, 60))
            pca_range = st.slider("過去契約 (PCA) カバー率 変動範囲 (%)", 0, 100, (30, 80))

        with mc_c2:
            st.subheader("⚡ 需要 ＆ 交渉価格の変動範囲")
            req_range = st.slider("非保持論文リクエスト率 変動範囲 (%)", 1, 20, (3, 10))
            apc_range = st.slider("交渉妥結 APC価格 変動範囲 (万円/本)", 15, 45, (22, 35))

        mc_params = {
            'unb_min': unb_range[0], 'unb_max': unb_range[1],
            'pca_min': pca_range[0], 'pca_max': pca_range[1],
            'req_min': req_range[0], 'req_max': req_range[1],
            'apc_min': apc_range[0], 'apc_max': apc_range[1],
            'green_oa': g_oa, 'bf_rate': bf_rate, 'ill_rate': ill_r,
            'ill_unit_cost': ill_u, 'fund_investment': fund_inv
        }

        mc_df = run_monte_carlo_simulation_v15(st.session_state.master_db, p_type, t_year, mc_params, n_trials=n_trials)

        if not mc_df.empty:
            st.divider()
            mc_tab1, mc_tab2, mc_tab3, mc_tab4 = st.tabs([
                "🎯 確率的パレート空間", 
                "📈 コストリスク (CDF) 曲線", 
                "🌪️ 感度分析 (トルネード)", 
                "🏆 目標クリア率 ＆ 推奨解"
            ])

            with mc_tab1:
                st.subheader("🎯 確率的パレートフロンティア（密度プロット）")
                p_idx = find_pareto_front(mc_df['total_cost'].values, mc_df['total_oa_pubs'].values, x_mode='min', y_mode='max')
                p_df = mc_df.iloc[p_idx].sort_values('total_cost')

                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(
                    x=sanitize_list((mc_df['total_cost'] * mask_scale).tolist()),
                    y=sanitize_list(mc_df['total_oa_pubs'].tolist()),
                    mode='markers', name='モンテカルロ試行サンプル',
                    marker=dict(color='#1f77b4', size=4, opacity=0.3)
                ))
                fig_mc.add_trace(go.Scatter(
                    x=sanitize_list((p_df['total_cost'] * mask_scale).tolist()),
                    y=sanitize_list(p_df['total_oa_pubs'].tolist()),
                    mode='lines+markers', name='確率的パレート最適限界',
                    line=dict(color='#d62728', width=3),
                    marker=dict(color='#d62728', size=7)
                ))
                fig_mc.update_layout(xaxis=dict(title="総コスト（指標）"), yaxis=dict(title="OA論文創出数 (本)"), height=420, template="plotly_white")
                st.plotly_chart(fig_mc, use_container_width=True)

            with mc_tab2:
                st.subheader("📈 コストリスク累積確率分布 (CDF)")
                sorted_costs = np.sort((mc_df['total_cost'] * mask_scale).values)
                cdf_y = np.linspace(0, 100, len(sorted_costs))
                
                var_95 = float(np.percentile(sorted_costs, 95))
                var_50 = float(np.percentile(sorted_costs, 50))

                fig_cdf = go.Figure()
                fig_cdf.add_trace(go.Scatter(x=sanitize_list(sorted_costs.tolist()), y=sanitize_list(cdf_y.tolist()), mode='lines', name='累積確率 (%)', line=dict(color='#2ca02c', width=3)))
                fig_cdf.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text=f"95% VaR: {var_95:.2f}")
                fig_cdf.add_vline(x=var_50, line_dash="dot", line_color="blue", annotation_text=f"中央値: {var_50:.2f}")
                fig_cdf.update_layout(xaxis=dict(title="総コスト（指標）"), yaxis=dict(title="累積確率 (%)"), height=400, template="plotly_white")
                st.plotly_chart(fig_cdf, use_container_width=True)

            with mc_tab3:
                st.subheader("🌪️ パラメータ感度分析（総コストに対する相関係数）")
                corrs = {
                    '購読削減率 (Unbundle)': mc_df['unbundle_rate'].corr(mc_df['total_cost']),
                    '過去契約 (PCA) カバー率': mc_df['pca_rate'].corr(mc_df['total_cost']),
                    '非保持リクエスト率': mc_df['req_rate'].corr(mc_df['total_cost']),
                    '交渉 APC単価': mc_df['apc_price_man'].corr(mc_df['total_cost'])
                }
                corr_df = pd.DataFrame(list(corrs.items()), columns=['Parameter', 'Correlation']).fillna(0.0).sort_values('Correlation')
                
                fig_sens = px.bar(corr_df, x='Correlation', y='Parameter', orientation='h', title="総コストへの感度影響度 (相関係数)", color='Correlation', color_continuous_scale='RdBu_r')
                fig_sens.update_layout(height=350, template="plotly_white")
                st.plotly_chart(fig_sens, use_container_width=True)

            with mc_tab4:
                st.subheader("🏆 交渉目標制約のクリア率 ＆ 推奨解")
                c_goal1, c_goal2 = st.columns(2)
                target_cost = c_goal1.number_input("目標 総コスト上限 (金額軸)", value=float(round(var_50, 2)))
                target_oa = c_goal2.number_input("目標 OA創出数下限 (本)", value=int(round(mc_df['total_oa_pubs'].mean())))

                success_mask = ((mc_df['total_cost'] * mask_scale) <= target_cost) & (mc_df['total_oa_pubs'] >= target_oa)
                success_rate = (success_mask.sum() / len(mc_df)) * 100.0

                st.metric("目標同時達成の確率 (勝率)", f"{success_rate:.1f} %")

                if success_mask.sum() > 0:
                    succ_df = mc_df[success_mask]
                    rec_table = pd.DataFrame({
                        'パラメータ': ['推奨 購読削減率', '推奨 PCAカバー率', '許容 リクエスト率', '目標 APC妥結価格'],
                        '平均推奨値': [
                            f"{succ_df['unbundle_rate'].mean():.1f} %",
                            f"{succ_df['pca_rate'].mean():.1f} %",
                            f"{succ_df['req_rate'].mean():.1f} %",
                            f"{succ_df['apc_price_man'].mean():.1f} 万円/本"
                        ]
                    })
                    st.table(rec_table)

    # --------------------------------------------------------------------------
    # メニュー 6: 💾 実データ一覧・出力
    # --------------------------------------------------------------------------
    elif mode == "💾 実データ一覧・出力":
        st.header("💾 シミュレーション実データ一覧および CSV 出力")
        st.dataframe(df_final_display, use_container_width=True)
        st.download_button(
            label="💾 分析結果 (CSV) をダウンロード",
            data=df_final_display.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"smart_consortium_v18_result_{p_type}_{t_year}.csv",
            mime="text/csv"
        )