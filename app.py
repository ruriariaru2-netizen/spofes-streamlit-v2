# app.py
import os
import requests
import pandas as pd
import streamlit as st

AUTO_REFRESH_SECONDS = 15

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False


def get_gas_url() -> str:
    if "GAS_WEBAPP_URL" in st.secrets:
        return str(st.secrets["GAS_WEBAPP_URL"])
    env = os.environ.get("GAS_WEBAPP_URL", "").strip()
    if env:
        return env
    raise RuntimeError("GAS_WEBAPP_URL が未設定です（secrets か環境変数に設定してください）")


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def fetch_year_list(gas_url: str) -> list:
    """
    GASに ?mode=years を投げて、年度一覧を取る想定。
    戻り値例: {"ok": true, "years": [2024, 2025, 2026]}
    """
    r = requests.get(gas_url, params={"mode": "years"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok", False):
        raise RuntimeError(data.get("error", "ok=false"))
    years = data.get("years", [])
    # 文字列でも数値でもOKに
    years = [str(y) for y in years]
    if not years:
        raise RuntimeError("years が空です（GAS側で years を返してください）")
    return years


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def fetch_payload(gas_url: str, year: str) -> dict:
    """
    GASに ?year=YYYY を投げて、その年度の payload を取る想定。
    戻り値例:
    {
      "ok": true,
      "payload": {"leagues":[...], "timetable":[...]},
      "updated_at": "..."
    }
    """
    r = requests.get(gas_url, params={"year": year}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok", False):
        raise RuntimeError(data.get("error", "ok=false"))
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("payload が dict ではありません")
    return data


def to_df_safe(obj, columns=None) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame(columns=columns or [])
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame([obj])
    return pd.DataFrame(columns=columns or [])


# =========================
# UI
# =========================
st.set_page_config(page_title="スポフェス 時程表", layout="wide")
st.title("スポフェス：リーグ分け & 時程表（年度選択・自動更新）")

if _HAS_AUTOREFRESH:
    st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="autorefresh")
else:
    st.warning("自動更新を有効にするには `streamlit-autorefresh` を入れてください。")

gas_url = get_gas_url()

# 年度一覧を取得
try:
    years = fetch_year_list(gas_url)
except Exception as e:
    st.error(f"年度一覧の取得に失敗：{e}")
    st.stop()

# 年度選択（サイドバー）
st.sidebar.header("表示設定")
default_year = years[-1]  # 最新をデフォルト
year = st.sidebar.selectbox("年度", options=years, index=years.index(default_year))

# 年度ごとのデータ取得
try:
    data = fetch_payload(gas_url, year)
except Exception as e:
    st.error(f"データ取得に失敗（year={year}）：{e}")
    st.stop()

payload = data.get("payload", {})
updated_at = data.get("updated_at", "")

leagues_raw = payload.get("leagues") or payload.get("leagues_df") or []
timetable_raw = payload.get("timetable") or payload.get("timetable_df") or []

leagues_df = to_df_safe(leagues_raw, columns=["event", "league", "team"])
timetable_df = to_df_safe(
    timetable_raw,
    columns=["slot_no", "start", "end", "event", "name", "team_a", "team_b", "referee", "phase", "gender"]
)

if "slot_no" in timetable_df.columns:
    timetable_df["slot_no"] = pd.to_numeric(timetable_df["slot_no"], errors="coerce")
    timetable_df = timetable_df.sort_values(["slot_no", "event", "name"], na_position="last")

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    st.caption("選択年度")
    st.write(year)
with c2:
    st.caption("最終更新（GAS側）")
    st.write(updated_at if updated_at else "（未提供）")
with c3:
    st.caption("データ取得元")
    st.code(gas_url, language="text")

tabs = st.tabs(["時程表", "リーグ分け", "ダウンロード"])

with tabs[0]:
    st.subheader("時程表")
    if timetable_df.empty:
        st.info("時程表データが空です。")
    else:
        st.dataframe(timetable_df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("リーグ分け")
    if leagues_df.empty:
        st.info("リーグ分けデータが空です。")
    else:
        st.dataframe(leagues_df.sort_values(["event", "league", "team"]), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("CSV ダウンロード")
    if not leagues_df.empty:
        st.download_button(
            "leagues.csv をダウンロード",
            data=leagues_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"leagues_{year}.csv",
            mime="text/csv",
        )
    if not timetable_df.empty:
        st.download_button(
            "timetable.csv をダウンロード",
            data=timetable_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"timetable_{year}.csv",
            mime="text/csv",
        )
