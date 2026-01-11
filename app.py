import requests
import pandas as pd
import streamlit as st

from scheduler import (
    try_build_parallel_timetable_with_retries_v2,
    export_leagues_and_timetable_dfs,
)

# ----------------------------
# Utility
# ----------------------------
def fetch_payload(gas_url: str, year: str | None):
    params = {}
    if year:
        params["year"] = year.strip()

    r = requests.get(gas_url, params=params, timeout=30)
    text = r.text
    r.raise_for_status()

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(
            "GASの返り値がJSONではありません。\n"
            f"status={r.status_code}\n"
            f"先頭200文字:\n{text[:200]}"
        )

    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Unknown GAS error"))

    return data["payload"]


def payload_to_inputs(payload: dict):
    # classes
    classes = [tuple(c) for c in payload["classes"]]

    # events
    events = {}
    for name, info in payload["events"].items():
        events[name] = {
            "participants": set(info["participants"]),
            "gender": info["gender"],
            "min_teams": info["min_teams"],
            "parallel": info["parallel"],
            "tournament_max_teams": info["tournament_max_teams"],
            "consolation_parallel": info["consolation_parallel"],
        }

    params = payload["params"]
    return classes, events, params


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="スポフェス時程表ジェネレーター", layout="wide")
st.title("スポフェス 時程表ジェネレーター")

with st.sidebar:
    st.header("接続設定")

    gas_url = st.secrets.get("GAS_WEBAPP_URL", "")
    gas_url = st.text_input(
        "GAS WebApp URL（/execまで）",
        value=gas_url,
        placeholder="https://script.google.com/macros/s/XXXX/exec",
    )

    year = st.text_input("年度（空欄＝最新）", value="")

    run = st.button("取得して生成", type="primary", use_container_width=True)

if not gas_url:
    st.info("左のサイドバーで GAS WebApp URL を入力してください。")
    st.stop()

if run:
    try:
        with st.status("GASからデータ取得中…"):
            payload = fetch_payload(gas_url, year)

        classes, events, params = payload_to_inputs(payload)

        st.success("データ取得完了")

        c1, c2 = st.columns(2)
        c1.metric("クラス数", len(classes))
        c2.metric("種目数", len(events))

        with st.status("時程表を生成中…", expanded=True):
            timetable, info = try_build_parallel_timetable_with_retries_v2(
                events,
                classes,
                start_time=params["start_time"],
                match_min=params["match_min"],
                change_min=params["change_min"],
                lookahead=params["lookahead"],
                league_attempts=params["league_attempts"],
                seed=params["seed"],
            )

            if not info["success"]:
                st.error("生成に失敗しました")
                st.code(info["last_error"])
                st.stop()

            leagues_df, timetable_df = export_leagues_and_timetable_dfs(
                events, classes, timetable, info
            )

        st.success("✅ 生成成功")

        tab1, tab2, tab3 = st.tabs(["時程表", "リーグ", "ダウンロード"])

        with tab1:
            st.dataframe(timetable_df, use_container_width=True, height=520)

        with tab2:
            st.dataframe(leagues_df, use_container_width=True, height=520)

        with tab3:
            st.download_button(
                "timetable.csv をダウンロード",
                df_to_csv_bytes(timetable_df),
                file_name="timetable.csv",
                use_container_width=True,
            )
            st.download_button(
                "leagues.csv をダウンロード",
                df_to_csv_bytes(leagues_df),
                file_name="leagues.csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"エラー: {e}")
        st.caption("GASのURLが間違っている / 未デプロイ / 再デプロイ忘れの可能性があります。")
else:
    st.caption("左でURLを設定し「取得して生成」を押してください。")
