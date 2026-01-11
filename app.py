# app.py
# Streamlit app: Google Sheets/GAS から最新のリーグ分け + 時程表を表示（入力なし・自動更新）

import os
import json
import requests
import pandas as pd
import streamlit as st

# --- 15秒ごとに自動リラン（あれば使う） ---
AUTO_REFRESH_SECONDS = 15
try:
    # pip install streamlit-autorefresh が入っていれば使える
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False


# =========================
# 設定（Secrets or 環境変数）
# =========================
def get_gas_url() -> str:
    """
    Streamlit Cloud なら .streamlit/secrets.toml で
      GAS_WEBAPP_URL="https://script.google.com/macros/s/...../exec"
    を設定するのが推奨。
    ローカルなら環境変数でもOK。
    """
    if "GAS_WEBAPP_URL" in st.secrets:
        return str(st.secrets["GAS_WEBAPP_URL"])
    env = os.environ.get("GAS_WEBAPP_URL", "").strip()
    if env:
        return env
    raise RuntimeError(
        "GAS_WEBAPP_URL が未設定です。\n"
        "Streamlit Cloud: secrets.toml に GAS_WEBAPP_URL を設定\n"
        "ローカル: 環境変数 GAS_WEBAPP_URL を設定\n"
    )


# =========================
# データ取得（15秒キャッシュ）
# =========================
@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def fetch_payload(gas_url: str) -> dict:
    """
    GAS Webアプリが返す JSON を取得。
    想定フォーマット例：
    {
      "ok": true,
      "payload": {
        "leagues": [...],
        "timetable": [...]
      },
      "updated_at": "2026-01-11 19:10:00"
    }
    """
    r = requests.get(gas_url, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, dict):
        raise RuntimeError("GASの戻り値が dict ではありません。")

    if not data.get("ok", False):
        raise RuntimeError(data.get("error", "GAS returned ok=false"))

    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("payload が dict ではありません。")

    return data


def to_df_safe(obj, columns=None) -> pd.DataFrame:
    """
    list[dict] / dict / None を DataFrame にする保険
    """
    if obj is None:
        return pd.DataFrame(columns=columns or [])
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # dict of list みたいな形ならそのままDF化、1レコードなら1行に
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame([obj])
    return pd.DataFrame(columns=columns or [])


# =========================
# UI
# =========================
st.set_page_config(page_title="スポフェス 時程表", layout="wide")

st.title("スポフェス：リーグ分け & 時程表（自動更新）")

# 自動更新（15秒）
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="autorefresh")
else:
    st.warning(
        "自動更新を有効にするには `streamlit-autorefresh` を入れてください。\n"
        "（入っていない場合でも、ブラウザ更新で最新になります）"
    )

# 取得
try:
    gas_url = get_gas_url()
    data = fetch_payload(gas_url)
except Exception as e:
    st.error(f"データ取得に失敗しました：{e}")
    st.stop()

payload = data.get("payload", {})
updated_at = data.get("updated_at", "")

# payload から leagues / timetable を取り出す（キー名違いにも少し耐性）
leagues_raw = payload.get("leagues") or payload.get("leagues_df") or payload.get("league") or []
timetable_raw = payload.get("timetable") or payload.get("timetable_df") or payload.get("schedule") or []

leagues_df = to_df_safe(leagues_raw, columns=["event", "league", "team"])
timetable_df = to_df_safe(
    timetable_raw,
    columns=["slot_no", "start", "end", "event", "name", "team_a", "team_b", "referee", "phase", "gender"]
)

# 見た目を整える
if "slot_no" in timetable_df.columns:
    # 数値としてソートできるように
    timetable_df["slot_no"] = pd.to_numeric(timetable_df["slot_no"], errors="coerce")
    timetable_df = timetable_df.sort_values(["slot_no", "event", "name"], na_position="last")

# 上部情報
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    st.caption("更新頻度")
    st.write(f"{AUTO_REFRESH_SECONDS} 秒ごと（開いている画面が自動で更新）")
with c2:
    st.caption("最終更新（GAS側）")
    st.write(updated_at if updated_at else "（未提供）")
with c3:
    st.caption("データ取得元（GAS Webアプリ）")
    st.code(gas_url, language="text")

tabs = st.tabs(["時程表", "リーグ分け", "ダウンロード"])

# ========= 時程表 =========
with tabs[0]:
    st.subheader("時程表")

    if timetable_df.empty:
        st.info("時程表データが空です。GASの返却 payload を確認してください。")
    else:
        # フィルタ（入力ではなく“選択”だけ・任意）
        # ※「入力欄は不要」と言っていたので、最低限のUIに留めてます
        events = sorted([x for x in timetable_df.get("event", pd.Series(dtype=str)).dropna().unique()])
        phases = sorted([x for x in timetable_df.get("phase", pd.Series(dtype=str)).dropna().unique()])

        colA, colB = st.columns([2, 2])
        with colA:
            ev_sel = st.multiselect("表示する種目（空=全部）", options=events, default=[])
        with colB:
            ph_sel = st.multiselect("表示する区分（空=全部）", options=phases, default=[])

        view = timetable_df.copy()
        if ev_sel:
            view = view[view["event"].isin(ev_sel)]
        if ph_sel and "phase" in view.columns:
            view = view[view["phase"].isin(ph_sel)]

        st.dataframe(view, use_container_width=True, hide_index=True)

# ========= リーグ分け =========
with tabs[1]:
    st.subheader("リーグ分け")

    if leagues_df.empty:
        st.info("リーグ分けデータが空です。GASの返却 payload を確認してください。")
    else:
        st.dataframe(leagues_df.sort_values(["event", "league", "team"]), use_container_width=True, hide_index=True)

# ========= ダウンロード =========
with tabs[2]:
    st.subheader("CSV ダウンロード")

    if not leagues_df.empty:
        st.download_button(
            label="leagues.csv をダウンロード",
            data=leagues_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="leagues.csv",
            mime="text/csv",
        )
    else:
        st.info("leagues.csv は空です。")

    if not timetable_df.empty:
        st.download_button(
            label="timetable.csv をダウンロード",
            data=timetable_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="timetable.csv",
            mime="text/csv",
        )
    else:
        st.info("timetable.csv は空です。")
