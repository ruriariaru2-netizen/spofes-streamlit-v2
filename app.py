import requests
import pandas as pd
import streamlit as st

from scheduler import (
    RobustTimetableBuilder,
    TimeConfig,
    ScheduleConfig,
    TimetableExporter,
)

st.set_page_config(page_title="スポフェス 時程表", layout="wide")

# =========================
# 設定：GASのURL
# =========================
GAS_URL = st.secrets.get("GAS_WEBAPP_URL", "")

# =========================
# データ取得（キャッシュON / ttl無し）
# =========================
@st.cache_data
def fetch_payload(gas_url: str, year: str) -> dict:
    if not gas_url:
        raise RuntimeError("GAS_URL が未設定です（st.secrets または直書きで設定してね）")

    r = requests.get(gas_url, params={"year": year}, timeout=30)
    r.raise_for_status()

    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "GASから ok:false が返りました"))

    return data["payload"]


# =========================
# payload → scheduler入力に整形
# =========================
def _normalize_classes(classes_raw):
    """
    classes は [(code, name), ...] 想定。
    payload が [{"code":..,"name":..}] / [["1A","1年A"],...] などでも吸収する。
    """
    if not classes_raw:
        return []

    out = []
    for c in classes_raw:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append((str(c[0]), str(c[1])))
        elif isinstance(c, dict):
            code = c.get("code") or c.get("class") or c.get("id") or c.get("name")
            name = c.get("name") or c.get("label") or str(code)
            out.append((str(code), str(name)))
        else:
            # 最低限 code だけでも持つ
            out.append((str(c), str(c)))
    return out


def _normalize_events(events_raw):
    """
    events は {event_name: {...}} 想定。
    payload が [{"event":.., ...}, ...] の形でも吸収する。
    """
    if not events_raw:
        return {}

    if isinstance(events_raw, dict):
        return events_raw

    if isinstance(events_raw, list):
        out = {}
        for e in events_raw:
            if isinstance(e, dict):
                name = e.get("event") or e.get("event_name") or e.get("name")
                if not name:
                    continue
                out[str(name)] = e
        return out

    return {}


def _build_configs_from_params(params: dict) -> tuple[TimeConfig, ScheduleConfig, int]:
    """
    payload["params"] のキーが多少違っても動くように寄せる。
    ついでに seed も拾えるなら拾う。
    """
    params = params or {}

    # ---- seed（任意）----
    seed = params.get("seed", None)
    if seed is None:
        seed = params.get("base_seed", None)
    if seed is None:
        seed = 0
    try:
        seed = int(seed)
    except Exception:
        seed = 0

    # ---- TimeConfig ----
    time_config = TimeConfig(
        start_time=str(params.get("start_time", params.get("tournament_start_time", "09:00"))),
        match_min=int(params.get("match_min", params.get("matchMinutes", 10))),
        change_min=int(params.get("change_min", params.get("changeMinutes", 3))),
        tournament_start_time=str(params.get("tournament_start_time", "13:00")),
        enforce_tournament_start=bool(params.get("enforce_tournament_start", True)),
    )

    # ---- ScheduleConfig ----
    schedule_config = ScheduleConfig(
        lookahead=int(params.get("lookahead", 80)),
        topn_k1=int(params.get("topn_k1", 20)),
        pair_trials_k2=int(params.get("pair_trials_k2", 200)),
        repair_iters=int(params.get("repair_iters", 80)),
        repair_redraws=int(params.get("repair_redraws", 30)),
        enable_cooldown=bool(params.get("enable_cooldown", True)),
        enable_repair=bool(params.get("enable_repair", True)),
        league_attempts=int(params.get("league_attempts", 30)),
        min_games=int(params.get("min_games", 3)),
    )

    return time_config, schedule_config, seed


def build_schedule_locally(payload: dict):
    events = _normalize_events(payload.get("events", {}))
    classes = _normalize_classes(payload.get("classes", []))
    params = payload.get("params", {})

    time_config, schedule_config, seed = _build_configs_from_params(params)

    builder = RobustTimetableBuilder(
        events=events,
        classes=classes,
        time_config=time_config,
        schedule_config=schedule_config,
    )

    timetable, info = builder.build_with_retries(seed=seed)

    if not info.get("success"):
        raise RuntimeError(info.get("last_error", "スケジューリングに失敗しました"))

    all_event_results = info.get("all_event_results", [])
    leagues_df, timetable_df = TimetableExporter.to_dataframes(timetable, all_event_results)
    return leagues_df, timetable_df, info


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

key_loaded = f"loaded_{year}"

should_fetch = False
if key_loaded not in st.session_state:
    should_fetch = True
elif manual_refresh:
    should_fetch = True

if should_fetch:
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

payload = st.session_state.get(f"payload_{year}")
if not payload:
    st.info("まだデータがありません。上のボタンで取得してください。")
    st.stop()

# =========================
# 表示
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
            leagues_df, timetable_df, info = build_schedule_locally(payload)

            if timetable_df.empty:
                st.error("時程表データが空です。入力データ（種目/参加クラス/同時進行数など）を確認してね。")
                st.stop()

            st.success("生成できました！")

            # 生成ログ（軽く）
            with st.expander("生成情報（info）", expanded=False):
                st.json({k: v for k, v in info.items() if k != "all_event_results"})

            st.subheader("リーグ分け")
            st.dataframe(leagues_df, use_container_width=True)

            st.subheader("時程表")
            st.dataframe(timetable_df, use_container_width=True)

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
