# -*- coding: utf-8 -*-
# app.py
# streamlit run app.py

import pandas as pd
import streamlit as st
import string

import spofes_engine as eng

# 先に1回だけ
st.set_page_config(page_title="スポフェス自動編成", layout="wide")
st.title("スポフェス自動編成（リーグ分け＋時程表＋CSV出力）")

COLOR_OPTIONS = ["", "赤", "青", "黄"]  # 空欄=未選択

def make_default_class_df():
    rows = []
    for letter in list(string.ascii_uppercase[:10]):  # 1年 A-J
        rows.append({"学年": 1, "クラス": letter, "色": ""})
    for letter in list(string.ascii_uppercase[:9]):   # 2年 A-I
        rows.append({"学年": 2, "クラス": letter, "色": ""})
    for letter in list(string.ascii_uppercase[:9]):   # 3年 A-I
        rows.append({"学年": 3, "クラス": letter, "色": ""})
    return pd.DataFrame(rows)

def build_classes_from_df(df: pd.DataFrame):
    classes = []
    for _, r in df.iterrows():
        g = int(r["学年"])
        c = str(r["クラス"]).strip().upper()
        color = str(r["色"]).strip()
        if not c:
            continue
        if color not in ["赤", "青", "黄"]:
            continue
        classes.append([f"{g}{c}", g, color])
    return classes

def validate_colors(df: pd.DataFrame):
    bad = []
    for _, r in df.iterrows():
        g = int(r["学年"])
        c = str(r["クラス"]).strip().upper()
        color = str(r["色"]).strip()
        if color not in ["赤", "青", "黄"]:
            bad.append(f"{g}{c}")
    if bad:
        st.error(
            "❌ 色が未選択のクラスがあります（赤・青・黄から1つ選んでください）\n\n"
            + "・" + "\n・".join(bad)
        )
        st.stop()

def normalize_config(cfg: dict):
    classes = cfg.get("classes", [])
    events = cfg.get("events", {})
    params = cfg.get("params", {})

    classes_t = [tuple(x) for x in classes]

    events_n = {}
    all_class_ids = [c[0] for c in classes]
    for name, info in events.items():
        ii = dict(info)
        parts = ii.get("participants", [])
        if not parts:
            parts = all_class_ids
        ii["participants"] = set(parts)
        events_n[name] = ii

    return classes_t, events_n, params

DEFAULT_CONFIG = {
    "events": {
        "リレー(男子)": {
            "participants": [],
            "gender": "M",
            "min_teams": 3,
            "parallel": 2,
            "tournament_max_teams": 8,
            "consolation_parallel": 1
        }
    },
    "params": {
        "start_time": "09:00",
        "match_min": 5,
        "change_min": 3,
        "lookahead": 20,
        "league_attempts": 30,
        "seed": 42,
        "tournament_start_time": "13:00",
        "enforce_tournament_start": True,
        "min_games": 3
    }
}

# ----- editorの安定同期 -----
def _sync_editor_to_df():
    st.session_state.class_df = st.session_state.class_editor_df

def _reset_df():
    st.session_state.class_df = make_default_class_df()
    st.session_state.class_editor_df = st.session_state.class_df.copy()

# 初期化
if "class_df" not in st.session_state:
    st.session_state.class_df = make_default_class_df()
if "class_editor_df" not in st.session_state:
    st.session_state.class_editor_df = st.session_state.class_df.copy()

# ----------------------------
# クラス設定
# ----------------------------
st.header("① クラス設定")

st.button("デフォルトに戻す", on_click=_reset_df)

st.data_editor(
    st.session_state.class_df,
    key="class_editor_df",
    on_change=_sync_editor_to_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "学年": st.column_config.SelectboxColumn("学年", options=[1, 2, 3], required=True),
        "クラス": st.column_config.TextColumn("クラス", required=True),
        "色": st.column_config.SelectboxColumn("色", options=COLOR_OPTIONS, required=True),
    },
)

# ----------------------------
# 実行
# ----------------------------
st.subheader("実行")
run = st.button("スケジュール生成", type="primary")

if run:
    validate_colors(st.session_state.class_df)
    classes_ui = build_classes_from_df(st.session_state.class_df)

    cfg = {
        **DEFAULT_CONFIG,
        "classes": classes_ui,
        "per_event_parallel": 1,
    }
    classes, events, params = normalize_config(cfg)

    with st.spinner("生成中..."):
        final_timetable, info = eng.try_build_parallel_timetable_with_retries_v2(
            events=events,
            classes=classes,
            start_time=params.get("start_time", "09:00"),
            match_min=int(params.get("match_min", 5)),
            change_min=int(params.get("change_min", 3)),
            lookahead=int(params.get("lookahead", 20)),
            league_attempts=int(params.get("league_attempts", 30)),
            seed=int(params.get("seed", 42)),
        )

    if not info.get("success"):
        st.error("生成に失敗しました")
        st.code(info.get("last_error", "unknown error"))
        st.stop()

    st.success("生成成功！")
    st.session_state.final_timetable = final_timetable
    st.session_state.info = info
    st.session_state.classes = classes
    st.session_state.events = events
    st.session_state.params = params

# ----------------------------
# 結果表示
# ----------------------------
if "final_timetable" in st.session_state and "info" in st.session_state:
    final_timetable = st.session_state.final_timetable
    info = st.session_state.info
    classes = st.session_state.classes
    events = st.session_state.events
    params = st.session_state.params

    league_seed = info.get("league_seed", params.get("seed", 42))
    all_event_results = [
        eng.make_event_schedule(event_name, event_info, classes, league_seed=league_seed)
        for event_name, event_info in events.items()
    ]

    st.subheader("リーグ分け")
    league_rows = []
    for ev in sorted(all_event_results, key=lambda x: x["event"]):
        for L in sorted(ev["leagues"].keys()):
            for team in ev["leagues"][L]:
                league_rows.append({"event": ev["event"], "league": L, "team": team})
    df_leagues = pd.DataFrame(league_rows)
    st.dataframe(df_leagues, use_container_width=True, hide_index=True)

    st.subheader("時程表")
    tt_rows = []
    for slot_no, slot in enumerate(final_timetable, start=1):
        if not slot:
            continue
        for g in slot:
            a, b = g.get("display_teams", g.get("teams", (None, None)))
            tt_rows.append({
                "slot_no": slot_no,
                "start": g.get("start", ""),
                "end": g.get("end", ""),
                "event": g.get("event", ""),
                "name": g.get("name", ""),
                "team_a": "" if a is None else str(a),
                "team_b": "" if b is None else str(b),
                "referee": g.get("referee", ""),
                "phase": g.get("phase", ""),
                "gender": g.get("gender", "")
            })
    df_tt = pd.DataFrame(tt_rows)
    st.dataframe(df_tt, use_container_width=True, hide_index=True)

    st.subheader("CSVダウンロード")

    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("leagues.csv をダウンロード", data=to_csv_bytes(df_leagues),
                           file_name="leagues.csv", mime="text/csv")
    with col2:
        st.download_button("timetable.csv をダウンロード", data=to_csv_bytes(df_tt),
                           file_name="timetable.csv", mime="text/csv")

    st.caption(f"成功情報: {info}")
