import json
import pandas as pd
import requests
import streamlit as st

st.title("体育祭 時程表ジェネレーター（半自動）")

GAS_WEBAPP_URL = st.secrets.get("GAS_WEBAPP_URL", "")  # 後でSecretsに入れる
if not GAS_WEBAPP_URL:
    st.error("secrets に GAS_WEBAPP_URL を設定してください")
    st.stop()

year = st.text_input("年度（空なら最新）", value="2026")

if st.button("payload取得 → CSV作成"):
    # 1) payloadを取得
    params = {}
    if year.strip():
        params["year"] = year.strip()

    r = requests.get(GAS_WEBAPP_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    st.success("payload 取得OK")
    st.write("クラス数:", len(payload["classes"]))
    st.write("種目数:", len(payload["events"]))

    # 2) ここであなたのPythonロジックに渡して時程表を作る
    # schedule_df, leagues_df = build_schedule(payload)

    # 例：とりあえず classes をCSVにする
    classes_df = pd.DataFrame(payload["classes"], columns=["class_id", "grade", "color"])
    st.dataframe(classes_df, use_container_width=True)

    st.download_button(
        "classes.csv をダウンロード",
        classes_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="classes.csv",
        mime="text/csv",
    )
