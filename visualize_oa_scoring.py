import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. ページ基本設定
# ==============================================================================
st.set_page_config(
    page_title="雑誌別OAスコアリング ＆ 契約優先度検証ツール",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 出版社OAデータ ✕ 利用実績 雑誌別スコアリング検証")
st.markdown("""
本ツールは、InCites/Web of Science由来の**雑誌別OA比率データ**とCOUNTER等の**利用実績**をクロス分析し、  
パッケージ解体（Unbundle）時に**「どの雑誌を個別購読として残すべきか」「どれをILL/PPV対応に回せるか」**をスコアリング評価します。
""")

# ==============================================================================
# 2. パス自動探索付き データロード関数
# ==============================================================================
def find_file(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(current_dir, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(current_dir, "data", filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(current_dir, "..", filename)
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

@st.cache_data
def load_and_merge_oa_data():
    files = {
        'Elsevier': 'Elsevierまとめ_1980-2025.xlsx',
        'Wiley': 'Wileyまとめ_1980-2025.xlsx'
    }
    
    dfs = []
    missing_files = []
    
    for pub_name, filename in files.items():
        filepath = find_file(filename)
        if filepath and os.path.exists(filepath):
            try:
                sheet_name = f"{pub_name}_2020-2025"
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                df['Publisher'] = pub_name
                
                df['Journal_Title'] = df['Name'].astype(str).str.strip()
                df['ISSN_Clean'] = df['ISSN'].astype(str).str.replace('-', '').str.strip()
                df['eISSN_Clean'] = df['eISSN'].astype(str).str.replace('-', '').str.strip()
                
                df['WoS_Docs'] = pd.to_numeric(df['Web of Science Documents'], errors='coerce').fillna(0)
                df['Paywall_Docs'] = pd.to_numeric(df['出版社OAを除いた数(A)'], errors='coerce').fillna(0)
                
                df['Paywall_Rate_%'] = np.where(
                    df['WoS_Docs'] > 0,
                    (df['Paywall_Docs'] / df['WoS_Docs']) * 100.0,
                    0.0
                )
                df['OA_Rate_%'] = 100.0 - df['Paywall_Rate_%']
                
                dfs.append(df)
            except Exception as e:
                st.error(f"{pub_name} のロード中にエラーが発生しました: {e}")
        else:
            missing_files.append(filename)
                
    if missing_files:
        st.warning(f"⚠️ 以下のファイルが見つかりません: {', '.join(missing_files)}")
        
    if not dfs:
        return pd.DataFrame()
        
    df_merged = pd.concat(dfs, ignore_index=True)
    return df_merged

df_oa = load_and_merge_oa_data()

if df_oa.empty:
    st.error("⚠️ Excelファイルが発見できませんでした。")
    st.stop()

# ==============================================================================
# 3. サイドバー・パラメータ設定
# ==============================================================================
st.sidebar.header("⚙️ スコアリング閾値設定")

selected_pub = st.sidebar.selectbox("分析対象出版社", ["全体 (Elsevier + Wiley)", "Elsevier", "Wiley"])

if selected_pub != "全体 (Elsevier + Wiley)":
    df_filtered = df_oa[df_oa['Publisher'] == selected_pub].copy()
else:
    df_filtered = df_oa.copy()

st.sidebar.subheader("🎯 象限の閾値（基準値）")
paywall_threshold = st.sidebar.slider(
    "Paywall（非OA）率の境界線 (%)", 
    min_value=10, max_value=90, value=50, step=5,
    help="この値よりPaywall率が高い雑誌は、有料購読しないと読めない割合が高い論文群です。"
)

# 利用実績データのダミー（視認性の良い0〜1000件の範囲に調整）
np.random.seed(42)
df_filtered['Est_Requests'] = np.random.randint(10, 800, size=len(df_filtered))

requests_threshold = st.sidebar.number_input(
    "年間リクエスト（DL）数の境界線 (件/年)", 
    min_value=10, max_value=1000, value=300, step=10,
    help="この利用件数を超える雑誌は『高需要誌』と判定されます。"
)

# ==============================================================================
# 4. 象限（クアドラント）判定ロジック
# ==============================================================================
def classify_quadrant(row):
    high_req = row['Est_Requests'] >= requests_threshold
    high_paywall = row['Paywall_Rate_%'] >= paywall_threshold
    
    if high_req and high_paywall:
        return "Priority 1: 核心購読維持誌 (高需要 ✕ 高Paywall)"
    elif high_req and not high_paywall:
        return "Priority 2: OA代替解約候補 (高需要 ✕ 低Paywall)"
    elif not high_req and high_paywall:
        return "Priority 3: ILL/PPV補填誌 (低需要 ✕ 高Paywall)"
    else:
        return "Priority 4: 低優先誌 (低需要 ✕ 低Paywall)"

df_filtered['Quadrant'] = df_filtered.apply(classify_quadrant, axis=1)

# ==============================================================================
# 5. メイン画面描画
# ==============================================================================

# サマリーメトリクス
m1, m2, m3, m4 = st.columns(4)
p1_count = len(df_filtered[df_filtered['Quadrant'].str.startswith("Priority 1")])
p2_count = len(df_filtered[df_filtered['Quadrant'].str.startswith("Priority 2")])
p3_count = len(df_filtered[df_filtered['Quadrant'].str.startswith("Priority 3")])
p4_count = len(df_filtered[df_filtered['Quadrant'].str.startswith("Priority 4")])

m1.metric("🔴 P1: 核心購読維持誌", f"{p1_count:,} 誌", help="解約すると打撃が大きい雑誌")
m2.metric("🟡 P2: OA代替解約候補", f"{p2_count:,} 誌", help="利用はあるがOAで代替可能")
m3.metric("🟠 P3: ILL/PPV補填誌", f"{p3_count:,} 誌", help="ILLやPPVで対応した方が安価")
m4.metric("⚪ P4: 低優先誌", f"{p4_count:,} 誌", help="契約・補填ともに不要")

st.divider()

col_chart, col_stat = st.columns([2, 1])

with col_chart:
    st.subheader("📊 雑誌別 OA比率 ✕ 利用実績 4象限マトリクス")
    
    # 💡【修正点】log_y=True を削除し、通常の直線形Y軸に変更
    fig = px.scatter(
        df_filtered,
        x="Paywall_Rate_%",
        y="Est_Requests",
        color="Quadrant",
        hover_name="Journal_Title",
        # size="WoS_Docs",
        # size_max=50,
        color_discrete_map={
            "Priority 1: 核心購読維持誌 (高需要 ✕ 高Paywall)": "#d62728",
            "Priority 2: OA代替解約候補 (高需要 ✕ 低Paywall)": "#bcbd22",
            "Priority 3: ILL/PPV補填誌 (低需要 ✕ 高Paywall)": "#ff7f0e",
            "Priority 4: 低優先誌 (低需要 ✕ 低Paywall)": "#7f7f7f"
        },
        labels={
            "Paywall_Rate_%": "Paywall（非OA）率 (%) ➔ 右にいくほど購読が必要",
            "Est_Requests": "年間想定リクエスト（DL）数 ➔ 上にいくほど需要が高い",
            "Quadrant": "契約分類判定"
        }
    )
    
    # 十字の点線（境界線）を追加
    fig.add_vline(x=paywall_threshold, line_dash="dash", line_color="black", annotation_text=f"Paywall閾値: {paywall_threshold}%")
    fig.add_hline(y=requests_threshold, line_dash="dash", line_color="black", annotation_text=f"DL閾値: {requests_threshold}件")
    
    # X軸・Y軸の表示範囲をきれいに調整
    max_y = max(df_filtered['Est_Requests'].max() * 1.05, requests_threshold * 1.2)
    fig.update_layout(
        xaxis=dict(range=[0, 105]),
        yaxis=dict(range=[0, max_y]),
        height=520, 
        template="plotly_white", 
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_stat:
    st.subheader("💡 戦略的示唆 ＆ 構成比")
    
    quad_counts = df_filtered['Quadrant'].value_counts()
    fig_pie = px.pie(
        names=quad_counts.index,
        values=quad_counts.values,
        hole=0.4,
        color_discrete_sequence=['#d62728', '#bcbd22', '#ff7f0e', '#7f7f7f']
    )
    fig_pie.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)
    
    p2_ratio = (p2_count / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
    st.info(f"""
    **分析インサイト**:  
    全 {len(df_filtered):,} 誌中、**{p2_ratio:.1f}% ({p2_count:,} 誌)** の雑誌は利用数が多いものの **OA率が高いため、個別購読契約を解除しても学内への影響が小さい（OA代替可能）** 候補です。
    """)

# ==============================================================================
# 6. 詳細データテーブル
# ==============================================================================
st.divider()
st.subheader("📋 雑誌別スコアリング・評価詳細テーブル")

selected_quad = st.multiselect(
    "表示する象限（カテゴリー）でフィルタリング:",
    options=df_filtered['Quadrant'].unique().tolist(),
    default=df_filtered['Quadrant'].unique().tolist()
)

df_table = df_filtered[df_filtered['Quadrant'].isin(selected_quad)].sort_values(by="Est_Requests", ascending=False)

disp_cols = [
    'Journal_Title', 'Publisher', 'Quadrant', 'Est_Requests', 
    'Paywall_Rate_%', 'OA_Rate_%', 'WoS_Docs', 'Paywall_Docs', 'ISSN_Clean'
]

st.dataframe(
    df_table[disp_cols].style.format({
        'Est_Requests': '{:,.0f} 件',
        'Paywall_Rate_%': '{:.1f}%',
        'OA_Rate_%': '{:.1f}%',
        'WoS_Docs': '{:,.0f} 本',
        'Paywall_Docs': '{:,.0f} 本'
    }),
    use_container_width=True
)

st.download_button(
    label="💾 雑誌別スコアリング結果 (CSV) をダウンロード",
    data=df_table[disp_cols].to_csv(index=False).encode('utf-8-sig'),
    file_name=f"journal_oa_scoring_result_{selected_pub}.csv",
    mime="text/csv"
)