import os
import math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ==========================================
# 1. 『良問の風』『名問の森』マスター設定 (旧帝大到達 = 100,000pt)
# ==========================================
GOAL_POINTS = 100000
DATA_FILE = "butsuri_progress_data.csv"

# 構成: (参考書名, 分野, 章名, 開始番号, 終了番号, 標準問題数, 章目標ポイント[総合ルート時])
# 良問の風 (全154問: 小計 40,000 pt) + 名問の森 (全140問: 小計 60,000 pt) = 合計 100,000 pt
PHYSICS_CHAPTERS = [
    # --- 良問の風（全154問：基礎〜標準・地方国公立/MARCH〜旧帝大基礎 = 40,000 pt） ---
    ("良問の風", "力学", "良問 第1章 運動と力・剛体", 1, 18, 12, 4800),
    ("良問の風", "力学", "良問 第2章 エネルギーと運動量", 19, 34, 11, 4200),
    ("良問の風", "力学", "良問 第3章 円運動・単振動・万有引力", 35, 56, 14, 6000),
    ("良問の風", "熱力学", "良問 第4章 熱と気体の法則", 57, 73, 12, 4500),
    ("良問の風", "波動", "良問 第5章 波の性質・音波・光波", 74, 99, 18, 6500),
    ("良問の風", "電磁気", "良問 第6章 電場とコンデンサー・直流", 100, 121, 15, 5500),
    ("良問の風", "電磁気", "良問 第7章 電流と磁場・電磁誘導・交流", 122, 141, 13, 5500),
    ("良問の風", "原子", "良問 第8章 原子物理・核反応（巻末含）", 142, 154, 9, 3000),

    # --- 名問の森（全140問：応用〜難関・旧帝大合格完成 = 60,000 pt） ---
    ("名問の森", "力学", "名問 第1章 力学I（運動・エネルギー・運動量）", 1, 22, 12, 9500),
    ("名問の森", "力学", "名問 第2章 力学II（慣性力・円運動・単振動・万有引力）", 23, 45, 12, 10500),
    ("名問の森", "熱力学", "名問 第3章 熱力学（気体分子運動論・状態変化）", 46, 61, 9, 6500),
    ("名問の森", "波動", "名問 第4章 波動I（波の性質・定常波・ドップラー効果）", 62, 75, 8, 6000),
    ("名問の森", "波動", "名問 第5章 波動II（光の屈折・干渉・回折格子）", 76, 87, 7, 5000),
    ("名問の森", "電磁気", "名問 第6章 電磁気I（電場・コンデンサー・直流回路）", 88, 106, 10, 8000),
    ("名問の森", "電磁気", "名問 第7章 電磁気II（ローレンツ力・電磁誘導・交流）", 107, 127, 11, 9500),
    ("名問の森", "原子", "名問 第8章 原子（光電効果・ボーア模型・原子核）", 128, 140, 7, 5000),
]

STATUS_MULTIPLIER = {
    "未着手": 0.0,
    "解けなかった": 0.1,
    "一部解けた": 0.5,
    "解けた": 1.0,
}

STAGES = [
    ("基礎固め", "良問・基礎固め I", 0, 12000),
    ("基礎固め", "良問・基礎固め II", 12000, 26000),
    ("基礎固め", "良問マスター（標準完成）", 26000, 40000),
    ("得点力養成", "名問・応用演習 I", 40000, 55000),
    ("得点力養成", "名問・応用演習 II", 55000, 70000),
    ("実戦力完成", "名問・実戦力完成 I", 70000, 85000),
    ("実戦力完成", "名問・実戦力完成 II", 85000, 95000),
    ("旧帝大突破", "旧帝大合格圏", 95000, 100000),
]

CAUSE_TAGS = [
    "公式・定義の理解不足",
    "束縛条件・運動方程式の立式ミス",
    "保存則（エネルギー・運動量）の選択迷い",
    "正負の符号・向き・座標軸ミス",
    "近似計算・三角関数・微積の計算ミス",
    "グラフ描図・論述のポイント抜け",
    "完全理解・物理の本質が詰まった良問"
]


def build_master_dataframe() -> pd.DataFrame:
    rows = []
    uid = 1
    for book, field, chapter, start_no, end_no, std_count, chap_goal in PHYSICS_CHAPTERS:
        total_q = end_no - start_no + 1
        hard_count = total_q - std_count
        weight_sum = std_count * 1.0 + hard_count * 1.4
        base_unit = chap_goal / weight_sum

        allocated = 0
        for idx, q_no in enumerate(range(start_no, end_no + 1)):
            if book == "良問の風":
                if idx < std_count:
                    q_type = "良問・基本"
                    default_lvl = 4 if idx < (std_count // 2) else 5
                    weight = 1.0
                else:
                    q_type = "良問・＊頻出"
                    default_lvl = 6 if idx < total_q - 2 else 7
                    weight = 1.4
            else:  # 名問の森
                if idx < std_count:
                    q_type = "名問・標準"
                    default_lvl = 7 if idx < (std_count // 2) else 8
                    weight = 1.0
                else:
                    q_type = "名問・＊難関"
                    default_lvl = 9 if idx < total_q - 2 else 10
                    weight = 1.4

            if q_no == end_no:
                max_pt = chap_goal - allocated
            else:
                max_pt = int(round(base_unit * weight))
                allocated += max_pt

            rows.append({
                "ID": uid,
                "参考書": book,
                "問題番号": q_no,
                "表示番号": f"{book[:2]} No.{q_no:03d}",
                "分野": field,
                "章": chapter,
                "区分": q_type,
                "レベル": default_lvl,
                "満点pt": max_pt,
                "解答状況": "未着手",
                "獲得pt": 0,
                "周回数": 0,
                "最終学習日": "-",
                "原因タグ": "",
                "つまずきメモ": "",
                "解法のコツ": ""
            })
            uid += 1
    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        df_loaded = pd.read_csv(DATA_FILE)
        for col in ["原因タグ", "つまずきメモ", "解法のコツ"]:
            if col not in df_loaded.columns:
                df_loaded[col] = ""
            else:
                df_loaded[col] = df_loaded[col].fillna("").astype(str)
        return df_loaded
    df_init = build_master_dataframe()
    df_init.to_csv(DATA_FILE, index=False)
    return df_init


def save_data(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False)


def evaluate_physics_image(img: Image.Image, book: str, q_type: str, has_star_mark: bool) -> int:
    gray = np.array(img.convert("L"))
    contrast = float(np.std(gray))
    edges = (
        np.abs(np.diff(gray.astype(float), axis=0)).mean()
        + np.abs(np.diff(gray.astype(float), axis=1)).mean()
    )
    visual_complexity = (edges * 0.32) + (contrast * 0.04)
    base_offset = 0 if book == "良問の風" else 3
    if "難関" in q_type or "頻出" in q_type:
        base_offset += 1
    star_bonus = 1 if has_star_mark else 0
    return int(np.clip(round(visual_complexity + base_offset + star_bonus), 1, 10))


def get_stage_info(points: int):
    for idx, (cat, name, low, high) in enumerate(STAGES):
        if points < high or idx == len(STAGES) - 1:
            next_name = STAGES[min(idx + 1, len(STAGES) - 1)][1]
            rem = max(0, high - points)
            span = max(1, high - low)
            stage_pct = min(100.0, max(0.0, (points - low) / span * 100.0))
            return cat, name, next_name, rem, stage_pct
    return "基礎固め", "良問・基礎固め I", "良問・基礎固め II", 12000, 0.0


def render_polar_chart_svg(chap_df: pd.DataFrame, target_ratio: float = 0.72) -> str:
    size = 360
    cx, cy = size / 2, size / 2
    max_r = 132
    n = len(chap_df)

    color_map = {
        "力学": ["#39D353", "#56D364", "#2DA44E", "#7CE38B", "#4AC26B"],
        "熱力学": ["#FF7B72", "#FFA198", "#F85149"],
        "波動": ["#549EF7", "#79C0FF", "#388BFD"],
        "電磁気": ["#F7DF54", "#F5CD38", "#FFE875", "#E3B341"],
        "原子": ["#D2A8FF", "#BC8CFF", "#A371F7"]
    }

    svg = [f'<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:380px;display:block;margin:0 auto;">']
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r}" fill="#FFFFFF" stroke="#000000" stroke-width="1.5"/>')

    for frac in [0.2, 0.4, 0.6, 0.8]:
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r * frac:.1f}" fill="none" stroke="#E2E0D8" stroke-width="1"/>')

    field_counters = {"力学": 0, "熱力学": 0, "波動": 0, "電磁気": 0, "原子": 0}
    for i, (_, row) in enumerate(chap_df.iterrows()):
        start_deg = -90 + (i * 360 / n)
        end_deg = -90 + ((i + 1) * 360 / n)
        a1 = math.radians(start_deg)
        a2 = math.radians(end_deg)

        gx = cx + max_r * math.cos(a1)
        gy = cy + max_r * math.sin(a1)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{gx:.1f}" y2="{gy:.1f}" stroke="#E2E0D8" stroke-width="1"/>')

        pct = min(100.0, max(0.0, float(row["進捗率(%)"])))
        r_val = max_r * (max(pct, 4.0) / 100.0) if pct > 0 else 0

        field = row["分野"]
        c_list = color_map.get(field, ["#549EF7"])
        fill_color = c_list[field_counters.get(field, 0) % len(c_list)]
        field_counters[field] = field_counters.get(field, 0) + 1

        if r_val > 0:
            x1 = cx + r_val * math.cos(a1)
            y1 = cy + r_val * math.sin(a1)
            x2 = cx + r_val * math.cos(a2)
            y2 = cy + r_val * math.sin(a2)
            path_d = f"M {cx},{cy} L {x1:.1f},{y1:.1f} A {r_val:.1f},{r_val:.1f} 0 0,1 {x2:.1f},{y2:.1f} Z"
            svg.append(f'<path d="{path_d}" fill="{fill_color}" fill-opacity="0.88" stroke="#FFFFFF" stroke-width="0.6"/>')

        mid_rad = math.radians((start_deg + end_deg) / 2)
        lx = cx + (max_r + 16) * math.cos(mid_rad)
        ly = cy + (max_r + 16) * math.sin(mid_rad) + 4
        svg.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="800" fill="#000000" font-family="sans-serif">{i + 1}</text>'
        )

    target_r = max_r * target_ratio
    svg.append(
        f'<circle cx="{cx}" cy="{cy}" r="{target_r:.1f}" fill="none" '
        f'stroke="#C93B3B" stroke-width="2" stroke-dasharray="5,4"/>'
    )
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r}" fill="none" stroke="#000000" stroke-width="1.8"/>')
    svg.append("</svg>")
    return "".join(svg)


# ==========================================
# 2. ページ設定 ＆ カスタムCSS（テキスト黒字統一）
# ==========================================
st.set_page_config(page_title="SIRUS - 良問の風・名問の森", page_icon="⚡", layout="centered")

st.markdown("""
<style>
.stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #000000 !important;
    font-family: "Hiragino Kaku Gothic ProN", "Noto Sans JP", sans-serif;
}
.stApp {
    background-color: #EEECE2;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 560px !important;
}
.hl-yellow {
    background: linear-gradient(transparent 55%, #F7DF54 55%);
    font-weight: 800;
    padding: 0 3px;
    color: #000000 !important;
}
.brand-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 10px;
    border-bottom: 1px solid #DCD9CA;
    margin-bottom: 12px;
}
.brand-logo {
    font-size: 24px;
    font-weight: 900;
    letter-spacing: 1.5px;
    color: #000000 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.icon-pill {
    border: 1.5px solid #000000;
    border-radius: 999px;
    padding: 4px 14px;
    background: #F5F4EC;
    font-size: 16px;
    display: flex;
    gap: 12px;
    color: #000000 !important;
}
.univ-header {
    font-size: 20px;
    font-weight: 800;
    color: #000000 !important;
    margin: 6px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.stage-bar {
    display: flex;
    gap: 4px;
    margin-bottom: 14px;
}
.stage-step {
    flex: 1;
    text-align: center;
    padding: 7px 4px;
    font-size: 12px;
    font-weight: 700;
    background: #DFDDD0;
    color: #000000 !important;
    clip-path: polygon(0% 0%, 90% 0%, 100% 50%, 90% 100%, 0% 100%, 10% 50%);
}
.stage-step:first-child {
    clip-path: polygon(0% 0%, 90% 0%, 100% 50%, 90% 100%, 0% 100%);
    border-radius: 4px 0 0 4px;
}
.stage-step.active {
    background: #000000;
    color: #FFFFFF !important;
}
.pill-progress-outer {
    width: 100%;
    height: 30px;
    background: #FFFFFF;
    border: 1.8px solid #000000;
    border-radius: 999px;
    padding: 3px;
    box-sizing: border-box;
    margin: 10px 0 14px 0;
}
.pill-progress-inner {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #E8F54A 0%, #38C958 100%);
    transition: width 0.4s ease;
}
.dotted-box {
    background-color: #F8F7F2;
    background-image: radial-gradient(#D5D2C4 1.2px, transparent 1.2px);
    background-size: 12px 12px;
    border-top: 1.5px solid #000000;
    border-bottom: 1.5px solid #000000;
    padding: 16px 14px;
    margin: 18px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stack-wrapper {
    position: relative;
    width: 175px;
    height: 150px;
    margin: 10px auto;
}
.stack-card {
    position: absolute;
    width: 145px;
    height: 130px;
    background: #FFFFFF;
    border: 1.5px solid #000000;
    border-radius: 4px;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.12);
    padding: 8px;
    font-size: 11px;
    color: #000000 !important;
}
.stack-tab {
    position: absolute;
    top: -18px;
    left: 4px;
    background: #FFFFFF;
    border: 1.5px solid #000000;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 1px 8px;
    font-size: 10px;
    font-weight: 700;
    color: #000000 !important;
}
.date-badge {
    position: absolute;
    right: -18px;
    background: #65A2F8;
    color: #000000 !important;
    border: 1.2px solid #000000;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 800;
}
.memo-card {
    background: #FFFFFF;
    border: 1.8px solid #000000;
    box-shadow: 3px 4px 0px #000000;
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 14px;
    position: relative;
    color: #000000 !important;
}
.memo-tag-chip {
    display: inline-block;
    background: #F7DF54;
    border: 1.2px solid #000000;
    border-radius: 999px;
    padding: 1px 9px;
    font-size: 11px;
    font-weight: 800;
    margin-right: 5px;
    margin-bottom: 4px;
    color: #000000 !important;
}
.memo-box-trouble {
    background: #FFF5F5;
    border-left: 4px solid #E05252;
    padding: 8px 10px;
    margin-top: 8px;
    font-size: 13px;
    border-radius: 0 4px 4px 0;
    color: #000000 !important;
}
.memo-box-tip {
    background: #F2FBF4;
    border-left: 4px solid #38C958;
    padding: 8px 10px;
    margin-top: 6px;
    font-size: 13px;
    border-radius: 0 4px 4px 0;
    color: #000000 !important;
}
div[data-testid="stTabs"] button {
    font-weight: 800 !important;
    font-size: 13px !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. データ集計と状態管理
# ==========================================
df = load_data()

if "selected_book" not in st.session_state:
    st.session_state.selected_book = "良問の風"
if "selected_q_no" not in st.session_state:
    st.session_state.selected_q_no = 1
if "target_univ" not in st.session_state:
    st.session_state.target_univ = "九州大学 理系(非医学部)"

# 上部ブランドバー
st.markdown("""
<div class="brand-bar">
    <div class="brand-logo">⚡ SIRUS PHYSICS</div>
    <div class="icon-pill"><span>↻</span><span>🔔</span><span>⚙️</span><span>⏱️</span></div>
</div>
""", unsafe_allow_html=True)

# 対象参考書モードの切り替え
book_mode = st.selectbox(
    "📚 表示・対策モードを選択",
    [
        "🔥 良問の風 ＋ 名問の森 総合ルート (全294問 / 100,000pt)",
        "🍃 良問の風 単独モード (全154問 / 100,000pt換算)",
        "🌲 名問の森 単独モード (全140問 / 100,000pt換算)"
    ]
)

if "良問の風 単独" in book_mode:
    df_view = df[df["参考書"] == "良問の風"].copy()
    scale_factor = GOAL_POINTS / df_view["満点pt"].sum()
elif "名問の森 単独" in book_mode:
    df_view = df[df["参考書"] == "名問の森"].copy()
    scale_factor = GOAL_POINTS / df_view["満点pt"].sum()
else:
    df_view = df.copy()
    scale_factor = 1.0

total_earned = int(round(df_view["獲得pt"].sum() * scale_factor))
total_pct = min(100.0, round((total_earned / GOAL_POINTS) * 100, 2))
solved_count = int((df_view["解答状況"] == "解けた").sum())
answered_count = int((df_view["解答状況"] != "未着手").sum())
study_days = max(1, df_view[df_view["最終学習日"] != "-"]["最終学習日"].str[:10].nunique())
total_difficulty = int(df_view[df_view["解答状況"] != "未着手"]["レベル"].sum())
memo_count = int(((df_view["つまずきメモ"].str.strip() != "") | (df_view["解法のコツ"].str.strip() != "")).sum())

cat_stage, cur_stage_name, next_stage_name, rem_pt, stage_pct = get_stage_info(total_earned)

tab_home, tab_train, tab_review, tab_mypage = st.tabs([
    "🏠 ホーム",
    "🏋️ トレーニング",
    "📄 解き直し・メモ",
    "👤 マイページ"
])

# ==========================================
# タブ1：🏠 ホーム
# ==========================================
with tab_home:
    st.markdown(f'<div class="univ-header">🏛️ {st.session_state.target_univ}</div>', unsafe_allow_html=True)

    step_names = ["基礎固め", "得点力養成", "実戦力完成", "🔒 ???"]
    step_html = '<div class="stage-bar">'
    for s_name in step_names:
        is_act = "active" if (cat_stage == s_name or (cat_stage == "旧帝大突破" and s_name == "🔒 ???")) else ""
        label = "旧帝大突破" if (cat_stage == "旧帝大突破" and s_name == "🔒 ???") else s_name
        step_html += f'<div class="stage-step {is_act}">{label}</div>'
    step_html += "</div>"
    st.markdown(step_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:12px; color:#000000; font-weight:700; margin-bottom:2px;">到達ステージ（総合進度 {total_pct:.1f}% ｜ 完答 {solved_count}/{len(df_view)}問）</div>
    <div style="display:flex; justify-content:space-between; align-items:baseline;">
        <div style="font-size:26px; font-weight:900; letter-spacing:0.5px; color:#000000;">{cur_stage_name}</div>
        <div style="font-size:28px; font-weight:900; color:#000000;">{total_earned:,} <span style="font-size:16px;">pt</span></div>
    </div>
    <div class="pill-progress-outer">
        <div class="pill-progress-inner" style="width: {max(6.0, stage_pct):.1f}%;"></div>
    </div>
    <div style="font-size:14px; font-weight:600; line-height:1.8; margin-bottom:14px; color:#000000;">
        次のステージ {next_stage_name} まで <span class="hl-yellow">あと {rem_pt:,} pt</span>
    </div>
    <hr style="border:none; border-top:1.5px solid #DCD9CA; margin:16px 0;">
    """, unsafe_allow_html=True)

    # 🎯 分野ごとの現在地（極座標グラフ）
    chap_summary = df_view.groupby(["参考書", "分野", "章"], sort=False).agg(
        問題数=("問題番号", "count"),
        完答=("解答状況", lambda s: (s == "解けた").sum()),
        獲得pt=("獲得pt", "sum"),
        目標pt=("満点pt", "sum")
    ).reset_index()
    chap_summary["進捗率(%)"] = (chap_summary["獲得pt"] / chap_summary["目標pt"] * 100).round(1)

    svg_chart = render_polar_chart_svg(chap_summary, target_ratio=0.72)
    st.markdown(f"""
    <div style="font-size:20px; font-weight:900; display:flex; align-items:center; gap:8px; margin-bottom:8px; color:#000000;">
        🎯 分野・章ごとの現在地
    </div>
    <div style="display:flex; justify-content:center; gap:24px; font-size:12px; font-weight:700; color:#000000; margin-bottom:4px;">
        <span>- - 今月の目標 (72%)</span>
        <span>— 合格ライン (100%: 10万pt)</span>
    </div>
    <div style="display:flex; justify-content:center; gap:12px; font-size:11px; font-weight:700; color:#000000; margin-bottom:6px;">
        <span>🟢力学</span>
        <span>🔴熱力学</span>
        <span>🔵波動</span>
        <span>🟡電磁気</span>
        <span>🟣原子</span>
    </div>
    {svg_chart}
    """, unsafe_allow_html=True)

    selected_chap_idx = st.selectbox(
        f"📌 扇形の番号（1〜{len(chap_summary)}）から章の詳細を確認",
        options=list(range(1, len(chap_summary) + 1)),
        format_func=lambda x: (
            f"{x}. {chap_summary.iloc[x-1]['章']} "
            f"（進捗: {chap_summary.iloc[x-1]['進捗率(%)']}%｜完答 {chap_summary.iloc[x-1]['完答']}/{chap_summary.iloc[x-1]['問題数']}問）"
        )
    )

    review_df = df_view[df_view["解答状況"].isin(["一部解けた", "解けなかった"])]
    memo_df = df_view[(df_view["つまずきメモ"].str.strip() != "") | (df_view["解法のコツ"].str.strip() != "")]
    recent_date = pd.Timestamp.now().strftime("%m/%d")

    if not memo_df.empty:
        latest_memo_row = memo_df.iloc[-1]
        preview_txt = latest_memo_row["解法のコツ"] if latest_memo_row["解法のコツ"] else latest_memo_row["つまずきメモ"]
        preview_txt = (preview_txt[:28] + "…") if len(preview_txt) > 28 else preview_txt
        stack_preview = f"<strong>{latest_memo_row['表示番号']} メモ:</strong><br>{preview_txt}"
    else:
        stack_preview = "「🏋️ トレーニング」で書いた解法のコツや弱点メモがここに表示されます。"

    st.markdown(f"""
    <div class="dotted-box">
        <div style="flex:1;">
            <div style="font-size:11px; font-weight:800; color:#000000;">総回答 / メモ数</div>
            <div style="font-size:24px; font-weight:900; margin-bottom:8px; color:#000000;">
                ✏️ <span class="hl-yellow">{answered_count}</span>
                <span style="font-size:13px; margin-left:4px;">(📝{memo_count})</span>
            </div>
            <div style="font-size:11px; font-weight:800; color:#000000;">累計日数</div>
            <div style="font-size:22px; font-weight:900; margin-bottom:8px; color:#000000;">📅 <span class="hl-yellow">{study_days}</span></div>
            <div style="font-size:11px; font-weight:800; color:#000000;">難易度合計</div>
            <div style="font-size:22px; font-weight:900; color:#000000;">📊 <span class="hl-yellow">{total_difficulty}</span></div>
        </div>
        <div class="stack-wrapper">
            <div class="stack-card" style="top:0px; left:24px; background:#F2F0E8;">
                <div class="stack-tab">見直し</div>
                <div class="date-badge" style="top:14px;">10/1</div>
            </div>
            <div class="stack-card" style="top:10px; left:12px; background:#F8F7F2;">
                <div class="stack-tab">見直し</div>
            </div>
            <div class="stack-card" style="top:20px; left:0px; background:#FFFFFF;">
                <div class="stack-tab">見直し</div>
                <div class="date-badge" style="top:36px;">{recent_date}</div>
                <div style="margin-top:8px; font-weight:800; color:#000000;">要復習: {len(review_df)}問</div>
                <div style="font-size:10px; color:#000000; margin-top:4px; line-height:1.35;">
                    {stack_preview}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# タブ2：🏋️ トレーニング
# ==========================================
with tab_train:
    st.subheader("🏋️ 『良問の風』『名問の森』問題記録 ＆ 苦手克服メモ")

    col_b, col_n = st.columns([1, 1])
    with col_b:
        book_options = ["良問の風", "名問の森"]
        sel_book = st.selectbox(
            "参考書を選択",
            book_options,
            index=book_options.index(st.session_state.selected_book)
        )
        st.session_state.selected_book = sel_book

    max_q_num = 154 if sel_book == "良問の風" else 140
    default_q_num = min(int(st.session_state.selected_q_no), max_q_num)

    with col_n:
        q_number = st.number_input(
            f"問題番号を選択 (No.1 〜 No.{max_q_num})",
            min_value=1,
            max_value=max_q_num,
            value=default_q_num,
            step=1
        )
        st.session_state.selected_q_no = q_number

    q_row = df[(df["参考書"] == sel_book) & (df["問題番号"] == q_number)].iloc[0]

    st.markdown(f"""
    <div style="background:#FFFFFF; border:1.8px solid #000000; box-shadow:3px 4px 0px #000000; padding:14px; margin-bottom:16px; color:#000000;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:12px; font-weight:800; color:#000000;">【{q_row['参考書']}】 {q_row['分野']} ｜ {q_row['章']}</span>
            <span style="background:#7CE38B; border:1.5px solid #000000; border-radius:999px; padding:2px 10px; font-size:12px; font-weight:900; color:#000000;">
                基準満点 {q_row['満点pt']} pt
            </span>
        </div>
        <div style="font-size:28px; font-weight:900; margin:4px 0; color:#000000;">
            {q_row['参考書']} No.{q_number:02d} <span style="font-size:16px;">（{q_row['区分']}）</span>
        </div>
        <div style="font-size:13px; font-weight:700; color:#000000;">
            現在の状態: <span class="hl-yellow">{q_row['解答状況']}</span>
            （獲得済: {q_row['獲得pt']} pt ／ 周回数: {q_row['周回数']}周 ／ 最終学習日: {q_row['最終学習日']}）
        </div>
    </div>
    """, unsafe_allow_html=True)

    status_choice = st.radio(
        "今回の解答状況を選択",
        ["解けた", "一部解けた", "解けなかった"],
        horizontal=True
    )
    has_star = st.checkbox("誌面に「＊（重要・頻出）」マークまたは「論述・描図」がある問題")

    st.markdown("#### 📝 物理・解法のコツ ＆ つまずきメモ")
    existing_tags = [t for t in str(q_row["原因タグ"]).split(",") if t in CAUSE_TAGS]
    selected_tags = st.multiselect(
        "🏷️ つまずき原因・特徴タグ（複数選択可）",
        options=CAUSE_TAGS,
        default=existing_tags
    )

    trouble_memo = st.text_area(
        "❓ 分からなかった箇所・ミスした原因",
        value=str(q_row["つまずきメモ"]) if str(q_row["つまずきメモ"]) != "nan" else "",
        placeholder="例：(3)の動く台の上での小物体の運動で、慣性力の向きと束縛条件の加速度成分を間違えた。",
        height=80
    )

    tip_memo = st.text_area(
        "💡 解法のコツ・次回の着眼点",
        value=str(q_row["解法のコツ"]) if str(q_row["解法のコツ"]) != "nan" else "",
        placeholder="例：外力の水平成分が0なら『水平方向の運動量保存則』＋『力学的エネルギー保存則』の連立が鉄板！",
        height=80
    )

    with st.expander("📸 問題画像のAI難易度判定（Lv.1〜10）を使う"):
        img_mode = st.radio("画像の入力方法", ["ファイル/写真選択", "スマホカメラで直接撮影"], horizontal=True)
        if img_mode == "スマホカメラで直接撮影":
            uploaded_img = st.camera_input("問題を撮影してください")
        else:
            uploaded_img = st.file_uploader("問題画像を選択", type=["png", "jpg", "jpeg"])

        current_lvl = int(q_row["レベル"])
        if uploaded_img is not None:
            img = Image.open(uploaded_img)
            if st.button("🔍 画像から難易度レベル(1〜10)を自動判定", use_container_width=True):
                current_lvl = evaluate_physics_image(img, sel_book, q_row["区分"], has_star)
                st.success(f"AI判定完了：{sel_book} No.{q_number} の難易度は **Lv.{current_lvl}** です！")

    final_lvl = st.slider("問題レベル (1〜10)", 1, 10, value=int(q_row["レベル"]))
    level_factor = 0.8 + (final_lvl / int(q_row["レベル"])) * 0.2
    new_earned_pt = int(round(q_row["満点pt"] * min(level_factor, 1.0) * STATUS_MULTIPLIER[status_choice]))

    if st.button(f"✅ {sel_book} No.{q_number} の学習結果とメモを保存（獲得: {new_earned_pt} pt）", type="primary", use_container_width=True):
        idx = df.index[(df["参考書"] == sel_book) & (df["問題番号"] == q_number)][0]
        df.at[idx, "レベル"] = final_lvl
        df.at[idx, "解答状況"] = status_choice
        df.at[idx, "獲得pt"] = new_earned_pt
        df.at[idx, "周回数"] = int(q_row["周回数"]) + 1
        df.at[idx, "最終学習日"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        df.at[idx, "原因タグ"] = ",".join(selected_tags)
        df.at[idx, "つまずきメモ"] = trouble_memo.strip()
        df.at[idx, "解法のコツ"] = tip_memo.strip()
        save_data(df)
        st.toast(f"{sel_book} No.{q_number} を保存しました！")
        st.rerun()

# ==========================================
# タブ3：📄 解き直し・メモ
# ==========================================
with tab_review:
    st.subheader("🗂️ 苦手対策・物理法則メモノート")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_mode = st.selectbox(
            "表示フィルター",
            ["📝 メモがある問題のみ", "⚠️ 要復習（一部解けた・解けなかった）", "すべて表示"]
        )
    with col_f2:
        tag_filter = st.selectbox("原因タグで絞り込み", ["すべて"] + CAUSE_TAGS)

    keyword = st.text_input("🔍 キーワード検索（例：単振動、ドップラー効果、電磁誘導、熱効率など）")

    filtered_df = df_view.copy()
    if filter_mode == "📝 メモがある問題のみ":
        filtered_df = filtered_df[
            (filtered_df["つまずきメモ"].str.strip() != "")
            | (filtered_df["解法のコツ"].str.strip() != "")
            | (filtered_df["原因タグ"].str.strip() != "")
        ]
    elif filter_mode == "⚠️ 要復習（一部解けた・解けなかった）":
        filtered_df = filtered_df[filtered_df["解答状況"].isin(["一部解けた", "解けなかった"])]

    if tag_filter != "すべて":
        filtered_df = filtered_df[filtered_df["原因タグ"].str.contains(tag_filter, na=False, regex=False)]

    if keyword.strip():
        kw = keyword.strip()
        filtered_df = filtered_df[
            filtered_df["つまずきメモ"].str.contains(kw, case=False, na=False)
            | filtered_df["解法のコツ"].str.contains(kw, case=False, na=False)
            | filtered_df["章"].str.contains(kw, case=False, na=False)
        ]

    if filtered_df.empty:
        st.info("条件に一致するメモ・復習問題はまだありません。")
    else:
        st.caption(f"該当件数: **{len(filtered_df)} 問**")
        for _, m_row in filtered_df.iterrows():
            q_no = int(m_row["問題番号"])
            b_name = m_row["参考書"]
            tags_list = [t for t in str(m_row["原因タグ"]).split(",") if t]
            tags_html = "".join([f'<span class="memo-tag-chip">🏷️ {t}</span>' for t in tags_list])

            trouble_html = (
                f'<div class="memo-box-trouble"><strong>❓ つまずき原因:</strong><br>{m_row["つまずきメモ"]}</div>'
                if str(m_row["つまずきメモ"]).strip() else ""
            )
            tip_html = (
                f'<div class="memo-box-tip"><strong>💡 解法のコツ:</strong><br>{m_row["解法のコツ"]}</div>'
                if str(m_row["解法のコツ"]).strip() else ""
            )

            st.markdown(f"""
            <div class="memo-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-size:11px; font-weight:800; color:#000000;">【{b_name}】 {m_row['分野']} ｜ {m_row['章']}</span>
                    <span style="font-size:11px; font-weight:800; background:#EEECE2; border:1px solid #000000; padding:1px 8px; border-radius:999px; color:#000000;">
                        {m_row['解答状況']}（{m_row['周回数']}周）
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <div style="font-size:22px; font-weight:900; color:#000000;">{b_name} No.{q_no:02d} <span style="font-size:13px;">（{m_row['区分']} / Lv.{m_row['レベル']}）</span></div>
                    <div style="font-size:11px; color:#000000;">最終学習: {m_row['最終学習日']}</div>
                </div>
                <div style="margin-top:4px;">{tags_html}</div>
                {trouble_html}
                {tip_html}
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"✏️ {b_name} No.{q_no:02d} をトレーニングで開く", key=f"edit_memo_{m_row['ID']}"):
                st.session_state.selected_book = b_name
                st.session_state.selected_q_no = q_no
                st.toast(f"{b_name} No.{q_no} をセットしました！")

    st.divider()
    with st.expander("📚 全16章（良問8章＋名問8章）進捗サマリー ＆ 全問題一覧を開く"):
        st.dataframe(chap_summary, use_container_width=True, hide_index=True)
        filter_chap = st.selectbox("章で絞り込み", ["すべて"] + [c[2] for c in PHYSICS_CHAPTERS])
        view_list_df = df if filter_chap == "すべて" else df[df["章"] == filter_chap]
        st.dataframe(view_list_df, use_container_width=True, hide_index=True)

# ==========================================
# タブ4：👤 マイページ
# ==========================================
with tab_mypage:
    st.subheader("👤 志望校設定 ＆ 分野別グラフ・データ管理")
    new_univ = st.text_input("目標大学・学部名", value=st.session_state.target_univ)
    if new_univ != st.session_state.target_univ:
        st.session_state.target_univ = new_univ
        st.rerun()

    st.markdown("#### 📊 物理5分野別 進捗率 (%)")
    field_grp = df_view.groupby("分野", sort=False).agg({"獲得pt": "sum", "満点pt": "sum"}).reset_index()
    field_grp["進捗率(%)"] = (field_grp["獲得pt"] / field_grp["満点pt"] * 100).round(2)
    st.bar_chart(field_grp.set_index("分野")["進捗率(%)"])

    st.markdown("#### 📚 参考書別 進捗率 (%)")
    book_grp = df.groupby("参考書", sort=False).agg({"獲得pt": "sum", "満点pt": "sum"}).reset_index()
    book_grp["進捗率(%)"] = (book_grp["獲得pt"] / book_grp["満点pt"] * 100).round(2)
    st.bar_chart(book_grp.set_index("参考書")["進捗率(%)"])

    st.divider()
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 進捗・メモデータのバックアップ (CSV)",
        data=csv_bytes,
        file_name="butsuri_progress_backup.csv",
        mime="text/csv",
        use_container_width=True
    )
    restore_file = st.file_uploader("📤 バックアップから復元", type=["csv"])
    if restore_file is not None:
        if st.button("データを復元する", use_container_width=True):
            df_restored = pd.read_csv(restore_file)
            save_data(df_restored)
            st.success("復元しました！")
            st.rerun()