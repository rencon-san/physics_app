import os
import math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ==========================================
# 1. 『良問の風 三訂版』『名問の森』マスター設定 (旧帝大到達 = 100,000pt)
# ==========================================
GOAL_POINTS = 100000
DATA_FILE = "butsuri_progress_v3.csv"

# 構成: (参考書名, 分野, 章名, 開始番号, 終了番号, 標準問題数, 章目標ポイント[総合ルート時])
# 良問の風 三訂版 (全154問: 小計 40,000 pt)
#   - 力学: 1〜53 (53問)
#   - 熱: 54〜70 (17問)
#   - 波動: 71〜99 (29問)
#   - 電磁気: 100〜142 (43問)
#   - 原子: 143〜154 (12問)
# 名問の森 (全140問: 小計 60,000 pt) = 合計 100,000 pt
PHYSICS_CHAPTERS = [
    # --- 良問の風 三訂版（全154問：小計 40,000 pt） ---
    ("良問の風", "力学", "良問 力学① 運動と力・剛体 (No.1〜18)", 1, 18, 12, 4600),
    ("良問の風", "力学", "良問 力学② エネルギーと運動量 (No.19〜35)", 19, 35, 12, 4400),
    ("良問の風", "力学", "良問 力学③ 円運動・単振動・万有引力 (No.36〜53)", 36, 53, 12, 5000),
    ("良問の風", "熱", "良問 熱 熱と気体の法則 (No.54〜70)", 54, 70, 12, 4500),
    ("良問の風", "波動", "良問 波動 波の性質・音波・光波 (No.71〜99)", 71, 99, 20, 7500),
    ("良問の風", "電磁気", "良問 電磁気① 電場・コンデンサー・直流 (No.100〜121)", 100, 121, 15, 5500),
    ("良問の風", "電磁気", "良問 電磁気② 電流と磁場・電磁誘導・交流 (No.122〜142)", 122, 142, 14, 5500),
    ("良問の風", "原子", "良問 原子 原子物理・核反応 (No.143〜154)", 143, 154, 8, 3000),

    # --- 名問の森（全140問：小計 60,000 pt） ---
    ("名問の森", "力学", "名問 第1章 力学I（運動・エネルギー・運動量）", 1, 22, 12, 9500),
    ("名問の森", "力学", "名問 第2章 力学II（慣性力・円運動・単振動・万有引力）", 23, 45, 12, 10500),
    ("名問の森", "熱", "名問 第3章 熱（気体分子運動論・状態変化）", 46, 61, 9, 6500),
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
                    q_type = "良問・標準"
                    default_lvl = 4 if idx < (std_count // 2) else 5
                    weight = 1.0
                else:
                    q_type = "良問・＊頻出/応用"
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
        "熱": ["#FF7B72", "#FFA198", "#F85149"],
        "波動": ["#549EF7", "#79C0FF", "#388BFD"],
        "電磁気": ["#F7DF54", "#F5CD38", "#FFE875", "#E3B341"],
        "原子": ["#D2A8FF", "#BC8CFF", "#A371F7"]
    }

    svg = [f'<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:380px;display:block;margin:0 auto;">']
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r}" fill="#FFFFFF" stroke="#000000" stroke-width="1.5"/>')

    for frac in [0.2, 0.4, 0.6, 0.8]:
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r * frac:.1f}" fill="none" stroke="#E2E0D8" stroke-width="1"/>')

    field_counters = {"力学": 0, "熱": 0, "波動": 0, "電磁気": 0, "原子": 0}
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
st.set_page_config(page_title="SIRUS - 良問の風(三訂版)・名問の森", page_icon="⚡", layout="centered")

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
        "🔥 良問の風(三訂版) ＋ 名問の森 総合ルート (全294問 / 100,000pt)",
        "🍃 良問の風(三訂版) 単独モード (全154問 / 100,000pt換算)",
        "🌲 名問の森 単独モード (全140問 / 100,000pt換算)"
    ]
)

if "良問の風(三訂版) 単独" in book_mode:
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
        <span>🔴熱</span>
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
            <div style="font-size:11px; font-weight:800; color:#000000;"