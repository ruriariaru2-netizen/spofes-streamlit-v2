import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="スポフェス 時程表", layout="wide")

# =========================
# 設定：GASのURLを入れる
# =========================

GAS_URL = st.secrets.get("GAS_WEBAPP_URL", "")


# =========================
# データ取得（キャッシュはONだが ttl無し＝勝手に再取得しない）
# =========================
@st.cache_data
def fetch_payload(gas_url: str, year: str) -> dict:
    if not gas_url:
        raise RuntimeError("GAS_URL が未設定です（st.secrets または直書きで設定してね）")

    # 例: .../exec?year=2026
    r = requests.get(gas_url, params={"year": year}, timeout=30)
    r.raise_for_status()

    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "GASから ok:false が返りました"))

    return data["payload"]


from scheduler import try_build_parallel_timetable_with_retries_v2, export_leagues_and_timetable_dfs

def build_schedule_locally(payload: dict):
    tt, info = try_build_parallel_timetable_with_retries_v2(
        payload["events"],
        payload["classes"],
        **payload["params"]
    )
    leagues_df, timetable_df = export_leagues_and_timetable_dfs(
        payload["events"],
        payload["classes"],
        tt,
        info
    )
    return leagues_df, timetable_df



# =========================
# UI：年度選択 + 半自動更新
# =========================
st.title("スポフェス：リーグ分け & 時程表")

year = st.text_input("年度（例: 2026 / DUMMY など）", value="DUMMY").strip()

col1, col2 = st.columns([1, 3])
with col1:
    manual_refresh = st.button("🔄 最新データを取得（手動）", use_container_width=True)
with col2:
    st.caption("半自動モード：最初の1回だけ自動取得。以降はこのボタンを押した時だけ更新します。")

# セッションに「このyearを読み込んだか」を覚えさせる
key_loaded = f"loaded_{year}"

# 初回だけ自動取得、ボタン押下時は強制再取得
should_fetch = False
if key_loaded not in st.session_state:
    should_fetch = True
elif manual_refresh:
    should_fetch = True

if should_fetch:
    # 手動更新の時だけキャッシュをクリアして取り直す
    if manual_refresh:
        st.cache_data.clear()

    with st.spinner("GASからデータ取得中..."):
        try:
            payload = fetch_payload(GAS_URL, year)
            st.session_state[key_loaded] = True
            st.session_state[f"payload_{year}"] = payload
            st.success("取得しました")
        except Exception as e:
            st.error(f"データ取得に失敗: {e}")

# payload が無ければ終了（ここで勝手に更新されない）
payload = st.session_state.get(f"payload_{year}")
if not payload:
    st.info("まだデータがありません。上のボタンで取得してください。")
    st.stop()


# =========================
# 表示（ここでスケジュール生成）
# =========================
st.subheader(f"取得した年度: {payload.get('tournamentId', year)}")

with st.expander("payload（確認用）", expanded=False):
    st.json(payload)

st.divider()

st.subheader("リーグ分け / 時程表の生成")
run_build = st.button("📌 このデータで時程表を生成", type="primary")

if run_build:
    with st.spinner("時程表を生成中..."):
        try:
            leagues_df, timetable_df = build_schedule_locally(payload)

            if timetable_df.empty:
                st.error("時程表データが空です。入力データ（種目/参加クラス/同時進行数など）を確認してね。")
                st.stop()

            st.success("生成できました！")

            st.subheader("リーグ分け")
            st.dataframe(leagues_df, use_container_width=True)

            st.subheader("時程表")
            st.dataframe(timetable_df, use_container_width=True)

            # ダウンロード
            st.download_button(
                "⬇ leagues.csv をダウンロード",
                data=leagues_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"leagues_{year}.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇ timetable.csv をダウンロード",
                data=timetable_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"timetable_{year}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"生成に失敗: {e}")
else:
    st.info("上の「このデータで時程表を生成」を押すと生成します（半自動で勝手に再生成しません）。")
