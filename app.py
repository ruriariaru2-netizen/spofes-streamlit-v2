import json
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import streamlit as st


# ----------------------------
# 設定
# ----------------------------
JST = timezone(timedelta(hours=9))


def fetch_payload(gas_url: str, year: str | None):
    """GAS Webアプリから payload を取得（{ok, payload/error} 形式）"""
    params = {}
    if year and year.strip():
        params["year"] = year.strip()

    r = requests.get(gas_url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, dict) or "ok" not in data:
        raise ValueError("GASの返り値が想定形式ではありません（okが無い）")

    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Unknown error from GAS"))

    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload が不正です")

    return payload


def payload_to_dataframes(payload: dict):
    """最低限の表にして返す（あなたの本処理に差し替え予定）"""
    # classes: [["1A",1,"赤"], ...]
    classes_df = pd.DataFrame(payload["classes"], columns=["class_id", "grade", "color"])

    # events: { "リレー": {participants:[], gender:"M", parallel:...}, ... }
    rows = []
    for name, info in payload["events"].items():
        rows.append(
            {
                "event_name": name,
                "gender": info.get("gender", ""),
                "parallel": info.get("parallel", ""),
                "participants": ",".join(info.get("participants", [])),
                "min_teams": info.get("min_teams", ""),
                "tournament_max_teams": info.get("tournament_max_teams", ""),
                "consolation_parallel": info.get("consolation_parallel", ""),
            }
        )
    events_df = pd.DataFrame(rows)

    params_df = pd.DataFrame([payload.get("params", {})])

    return classes_df, events_df, params_df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Excelでも文字化けしにくいUTF-8(BOM付き)"""
    return df.to_csv(index=False).encode("utf-8-sig")


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="体育祭ジェネレーター", layout="wide")
st.title("体育祭 時程表ジェネレーター（半自動）")

with st.sidebar:
    st.header("接続設定")
    st.caption("GAS WebアプリURL（/exec で終わる）を設定してね")

    # Streamlit Cloudなら secrets 推奨
    gas_url = st.secrets.get("GAS_WEBAPP_URL", "")
    gas_url = st.text_input("GAS_WEBAPP_URL", value=gas_url, placeholder="https://script.google.com/macros/s/.../exec")

    st.divider()
    st.header("入力")
    year = st.text_input("年度（空なら最新）", value="2026")
    run = st.button("取得して生成", type="primary", use_container_width=True)

if not gas_url:
    st.info("左のサイドバーで GAS_WEBAPP_URL を設定してください。")
    st.stop()

if run:
    try:
        with st.status("GASからpayloadを取得中...", expanded=False):
            payload = fetch_payload(gas_url, year)

        st.success("payload取得成功！")

        # 情報表示
        col1, col2, col3 = st.columns(3)
        col1.metric("クラス数", len(payload.get("classes", [])))
        col2.metric("種目数", len(payload.get("events", {})))
        col3.metric("年度", payload.get("tournamentId", year.strip() if year else "最新"))

        # ここで本来は「時程表生成」を呼ぶ
        # schedule_df, leagues_df = build_schedule(payload)

        classes_df, events_df, params_df = payload_to_dataframes(payload)

        tabs = st.tabs(["クラス", "種目", "設定(params)", "生payload(JSON)"])

        with tabs[0]:
            st.subheader("クラス一覧")
            st.dataframe(classes_df, use_container_width=True)
            st.download_button(
                "classes.csv をダウンロード",
                df_to_csv_bytes(classes_df),
                file_name="classes.csv",
                mime="text/csv",
            )

        with tabs[1]:
            st.subheader("種目一覧")
            st.dataframe(events_df, use_container_width=True)
            st.download_button(
                "events.csv をダウンロード",
                df_to_csv_bytes(events_df),
                file_name="events.csv",
                mime="text/csv",
            )

        with tabs[2]:
            st.subheader("設定(params)")
            st.dataframe(params_df, use_container_width=True)
            st.download_button(
                "params.csv をダウンロード",
                df_to_csv_bytes(params_df),
                file_name="params.csv",
                mime="text/csv",
            )

        with tabs[3]:
            st.subheader("payload(JSON)")
            st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

    except requests.HTTPError as e:
        st.error(f"HTTPエラー: {e}")
    except Exception as e:
        st.error(f"エラー: {e}")

else:
    st.caption("左の「取得して生成」を押すと、GASから最新（または指定年度）のデータを取得します。")
