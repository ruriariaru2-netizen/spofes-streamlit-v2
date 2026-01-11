import json
import pandas as pd
import requests
import streamlit as st

# あなたのロジック（scheduler.py）から使う
from scheduler import try_build_parallel_timetable_with_retries_v2, export_leagues_and_timetable_dfs


def fetch_payload(gas_url: str, year: str | None):
    params = {}
    if year and year.strip():
        params["year"] = year.strip()

    r = requests.get(gas_url, params=params, timeout=30)
    # JSONでない時の原因特定用に、まずテキストを持っておく
    text = r.text
    r.raise_for_status()

    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(
            "GASの返り値がJSONではありません。\n"
            f"status={r.status_code}, content-type={r.headers.get('content-type')}\n"
            f"先頭200文字:\n{text[:200]}"
        ) from e

    if not isinstance(data, dict) or "ok" not in data:
        raise RuntimeError(f"GAS返り値が想定形式ではありません: {str(data)[:200]}")

    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Unknown error from GAS"))

    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("payloadが不正です")

    return payload


def gender_to_X(g: str) -> str:
    g = (g or "").strip().upper()
    if g in ("M", "男子"):
        return "M"
    if g in ("F", "女子"):
        return "F"
    # GAS側は "MIX" を返す想定
    return "X"


def payload_to_inputs(payload: dict):
    """
    GAS payload -> (classes, events, params)
    scheduler.py が期待する形に変換する
    """
    # classes: GASは [["1A",1,"赤"], ...]
    classes = [tuple(x) for x in payload["classes"]]

    # events: GASは participants が list なので set にする
    events = {}
    for name, info in payload["events"].items():
        events[name] = {
            "participants": set(info.get("participants", [])),
            "gender": gender_to_X(info.get("gender", "X")),
            "min_teams": int(info.get("min_teams", 3)),
            "parallel": int(info.get("parallel", 2)),
            "tournament_max_teams": int(info.get("tournament_max_teams", 8)),
            "consolation_parallel": int(info.get("consolation_parallel", 1)),
        }

    params = payload.get("params", {})
    return classes, events, params


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="スポフェス時程表ジェネレーター", layout="wide")
st.title("スポフェス 時程表ジェネレーター（半自動）")

with st.sidebar:
    st.header("接続")
    gas_url = st.secrets.get("GAS_WEBAPP_URL", "")
    gas_url = st.text_input(
        "GAS_WEBAPP_URL（/execまで）",
        value=gas_url,
        placeholder="https://script.google.com/macros/s/.../exec",
    )

    st.header("入力")
    year = st.text_input("年度（空なら最新）", value="2026")

    st.header("生成パラメータ（基本はpayload優先）")
    override_match = st.checkbox("試合時間/入替を上書きする", value=False)
    match_min = st.number_input("試合時間（分）", min_value=1, max_value=60, value=5)
    change_min = st.number_input("入れ替え（分）", min_value=0, max_value=30, value=3)
    lookahead = st.number_input("lookahead", min_value=5, max_value=300, value=20)
    league_attempts = st.number_input("league_attempts", min_value=1, max_value=200, value=30)
    seed = st.number_input("seed", min_value=0, max_value=999999, value=42)

    run = st.button("取得して生成", type="primary", use_container_width=True)

if not gas_url:
    st.info("サイドバーで GAS_WEBAPP_URL を設定してください。")
    st.stop()

if run:
    try:
        with st.status("GASからpayload取得中…", expanded=False):
            payload = fetch_payload(gas_url, year)

        classes, events, params = payload_to_inputs(payload)

        # payload params を優先、ただしUIで上書きできる
        start_time = params.get("start_time", "09:00")
        tournament_start_time = params.get("tournament_start_time", "13:00")
        enforce_tournament_start = bool(params.get("enforce_tournament_start", True))

        use_match_min = int(params.get("match_min", 5))
        use_change_min = int(params.get("change_min", 3))
        if override_match:
            use_match_min = int(match_min)
            use_change_min = int(change_min)

        use_lookahead = int(params.get("lookahead", lookahead))
        use_attempts = int(params.get("league_attempts", league_attempts))
        use_seed = int(params.get("seed", seed))

        st.success("payload取得OK")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("クラス数", len(classes))
        c2.metric("種目数", len(events))
        c3.metric("試合時間", f"{use_match_min}分")
        c4.metric("入替", f"{use_change_min}分")

        with st.status("スケジュール生成中…", expanded=True):
            final_timetable, info = try_build_parallel_timetable_with_retries_v2(
                events, classes,
                start_time=start_time,
                match_min=use_match_min,
                change_min=use_change_min,
                lookahead=use_lookahead,
                league_attempts=use_attempts,
                seed=use_seed
            )

            if not info.get("success"):
                st.error("生成失敗")
                st.code(info.get("last_error", "unknown error"))
                st.stop()

            leagues_df, timetable_df = export_leagues_and_timetable_dfs(
                events, classes, final_timetable, info
            )

        st.success("✅ 生成成功！")

        # 表示
        tab1, tab2, tab3, tab4 = st.tabs(["時程表", "リーグ", "ダウンロード", "デバッグ情報"])

        with tab1:
            st.subheader("時程表")
            st.dataframe(timetable_df, use_container_width=True, height=520)

        with tab2:
            st.subheader("リーグ")
            st.dataframe(leagues_df, use_container_width=True, height=520)

        with tab3:
            st.subheader("CSVダウンロード")
            st.download_button(
                "timetable.csv をダウンロード",
                df_to_csv_bytes(timetable_df),
                file_name="timetable.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "leagues.csv をダウンロード",
                df_to_csv_bytes(leagues_df),
                file_name="leagues.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with tab4:
            st.subheader("info（再現用）")
            st.json(info)
            st.subheader("params（payload）")
            st.json(params)

    except Exception as e:
        st.error(f"エラー: {e}")
        st.caption("※ GASのURL/公開設定/再デプロイ漏れのときは、JSONじゃないHTMLが返ってこのエラーになりやすいです。")
else:
    st.caption("左の「取得して生成」を押すと、GASからpayloadを取って時程表を生成します。")
