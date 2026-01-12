    # -*- coding: utf-8 -*-
"""
スポーツ大会スケジューリングシステム（リファクタリング版）
責務分離・クラス化により保守性を向上
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque
import random
import string
import re
import csv
import pandas as pd
import copy
import logging

# ============================================================
# Config（設定の外部化）
# ============================================================

@dataclass
class TimeConfig:
    """時刻関連の設定"""
    start_time: str = "09:00"
    match_min: int = 10
    change_min: int = 3
    tournament_start_time: str = "13:00"
    enforce_tournament_start: bool = True


@dataclass
class ScheduleConfig:
    """スケジューリング関連の設定"""
    lookahead: int = 80
    topn_k1: int = 20
    pair_trials_k2: int = 200
    repair_iters: int = 80
    repair_redraws: int = 30
    enable_cooldown: bool = True
    enable_repair: bool = True
    league_attempts: int = 30
    min_games: int = 3
    # ★追加：repair の方式
    # "slot"  : いまの方式（各slotごとに衝突修理）
    # "global": まず全体を並べてから、同一event内swapで修理（あなたがやりたい方式）
    repair_mode: str = "global"
    global_repair_iters: int = 3000
    global_repair_max_tries_per_conflict: int = 200
    allow_insert_empty_slot: bool = True
    initial_shuffle: bool = True     
    initial_shuffle_within_phase: bool = True  
    global_repair_restarts: int = 10

# ============================================================
# Time Management（時刻管理）
# ============================================================

class TimeManager:
    """時刻の操作と計算"""

    @staticmethod
    def add_minutes(hhmm: str, minutes: int) -> str:
        """時刻に分を足す"""
        h, m = map(int, hhmm.split(":"))
        total = h * 60 + m + minutes
        return f"{total // 60:02d}:{total % 60:02d}"

    @staticmethod
    def minutes_between(t1: str, t2: str) -> int:
        """t2 - t1 を分で返す（t2が後なら正）"""
        h1, m1 = map(int, t1.split(":"))
        h2, m2 = map(int, t2.split(":"))
        return (h2 * 60 + m2) - (h1 * 60 + m1)

    @staticmethod
    def end_time_after_slots(
        start_time: str, n_slots: int, match_min: int, change_min: int
    ) -> str:
        """n_slots 個の枠を消化したときの次の開始時刻"""
        t = start_time
        for _ in range(n_slots):
            t = TimeManager.add_minutes(t, match_min + change_min)
        return t

    @staticmethod
    def num_empty_slots_to_reach(
        start_time: str, target_time: str, match_min: int, change_min: int
    ) -> int:
        """start_time から target_time まで、何枠必要か"""
        slot_len = match_min + change_min
        diff = TimeManager.minutes_between(start_time, target_time)
        if diff <= 0:
            return 0
        return (diff + slot_len - 1) // slot_len

    @staticmethod
    def add_times_to_timetable(
        raw_slots: List[List[dict]], 
        start_time: str, 
        match_min: int, 
        change_min: int
    ) -> List[List[dict]]:
        """スロットに開始・終了時刻を付与"""
        t = start_time
        out = []
        for slot in raw_slots:
            slot_out = []
            start = t
            end = TimeManager.add_minutes(start, match_min)
            for g in slot:
                gg = dict(g)
                gg["start"] = start
                gg["end"] = end
                slot_out.append(gg)
            out.append(slot_out)
            t = TimeManager.add_minutes(end, change_min)
        return out


# ============================================================
# Resource Management（リソース管理）
# ============================================================

class ResourceType:
    """リソースの種類を定義"""
    COLLISION = "collision"      # 同時刻衝突チェック
    COOLDOWN = "cooldown"        # 連続出場禁止
    CONSOLATION = "consolation"  # 敗者戦（種目分離）


class ResourceManager:
    """試合が使用するリソースを一元管理"""

    @staticmethod
    def team_gender_resources(team: Optional[str], gender: str) -> Set[str]:
        """クラス×性別のリソース集合"""
        if team is None:
            return set()
        if gender == "M":
            return {f"{team}_M"}
        if gender == "F":
            return {f"{team}_F"}
        # 混合(X)は両方
        return {f"{team}_M", f"{team}_F"}

    @staticmethod
    def get_collision_resources(game: dict) -> Set[str]:
        """
        同時刻衝突チェック用リソース
        
        - 予選・本選：クラス×性別
        - 敗者戦：種目|チーム|CONSOL（他eventと分離）
        """
        phase = game.get("phase", "")
        gender = game.get("gender", "X")
        event = game.get("event", "")
        
        used = set()
        
        if phase == "consolation":
            # 敗者戦：同一種目内での同順位ラベルの重複を防ぐ
            for t in game.get("teams", (None, None)):
                if t is not None:
                    used.add(f"{event}|{t}|CONSOL")
            return used

        # 予選・本選：男女/混合の衝突を防ぐ
        for t in game.get("teams", (None, None)):
            if t is not None:
                used |= ResourceManager.team_gender_resources(t, gender)

        return used

    @staticmethod
    def get_cooldown_resources(game: dict) -> Set[str]:
        """連続出場禁止用リソース（予選のみ）"""
        if game.get("phase") != "prelim":
            return set()
        gender = game.get("gender", "X")
        used = set()
        for t in game.get("teams", (None, None)):
            if t is not None:
                used |= ResourceManager.team_gender_resources(t, gender)
        return used

    @staticmethod
    def get_teams_only(game: dict) -> Set[str]:
        """チーム名のみを抽出（ログ用）"""
        a, b = game.get("teams", (None, None))
        return {t for t in (a, b) if t is not None}


# ============================================================
# League Management（リーグ管理）
# ============================================================

class LeagueManager:
    """リーグ分けと予選試合の生成"""

    @staticmethod
    def split_into_leagues(
        participants: List[str], min_teams: int, seed: int = 0
    ) -> Dict[str, List[str]]:
        """参加チームを均等なリーグに分割"""
        rng = random.Random(seed)
        parts = list(participants)
        rng.shuffle(parts)

        N = len(parts)
        k = max(2, min_teams)

        if N < k:
            return {"A": parts}

        L = N // k
        r = N % k
        league_sizes = [k + 1] * r + [k] * (L - r)

        leagues = {}
        idx = 0
        for i, size in enumerate(league_sizes):
            name = string.ascii_uppercase[i]
            leagues[name] = parts[idx : idx + size]
            idx += size

        return leagues

    @staticmethod
    def make_all_pairs(teams: List[str]) -> List[Tuple[str, str]]:
        """総当たり戦のペアを生成"""
        pairs = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                pairs.append((teams[i], teams[j]))
        return pairs

    @staticmethod
    def make_league_games(
        event_name: str, gender: str, leagues: Dict[str, List[str]]
    ) -> List[dict]:
        """リーグの総当たり試合を生成"""
        games = []
        for L in sorted(leagues.keys()):
            teams = leagues[L]
            pairs = LeagueManager.make_all_pairs(teams)
            for idx, (a, b) in enumerate(pairs, 1):
                games.append({
                    "event": event_name,
                    "gender": gender,
                    "name": f"{L}-予選{idx}",
                    "teams": (a, b),
                    "display_teams": (f"{L}{a}", f"{L}{b}"),
                    "league": L,
                    "phase": "prelim",
                })
        return games


# ============================================================
# Tournament Management（トーナメント管理）
# ============================================================

class TournamentManager:
    """本選トーナメントと敗者戦の生成"""

    @staticmethod
    def build_advancers(
        leagues: Dict[str, List[str]], max_teams: int = 8
    ) -> Tuple[List[str], Dict[int, List[str]]]:
        """
        本選進出チームと敗者戦対象者を決定
        
        Returns:
            (advancers, losers_by_rank)
            - advancers: 本選進出順位ラベル（強い順）
            - losers_by_rank: {順位: [順位ラベル]}
        """
        league_names = sorted(leagues.keys())
        max_rank = max(len(ts) for ts in leagues.values())

        advancers = []
        losers_by_rank = defaultdict(list)

        for rank in range(1, max_rank + 1):
            for L in league_names:
                if len(leagues[L]) >= rank:
                    label = f"{L}{rank}位"
                    if len(advancers) < max_teams:
                        advancers.append(label)
                    else:
                        losers_by_rank[rank].append(label)

        return advancers, dict(losers_by_rank)

    @staticmethod
    def _next_pow2(n: int) -> int:
        """n以上の最小2冪を返す"""
        p = 1
        while p < n:
            p *= 2
        return p

    @staticmethod
    def _round_name(n: int) -> str:
        """回戦数からラウンド名を生成"""
        if n == 2:
            return "決勝"
        if n == 4:
            return "準決勝"
        if n == 8:
            return "準々決勝"
        return f"{n}回戦"

    @staticmethod
    def make_seeded_pairs(advancers: List[str], bracket_size: int):
        """シード方式でペアを生成（1番vs下位方式）"""
        seeds = list(advancers) + [None] * max(0, bracket_size - len(advancers))
        pairs = []
        for i in range(bracket_size // 2):
            a = seeds[i]
            b = seeds[bracket_size - 1 - i]
            pairs.append((a, b))
        return pairs

    @staticmethod
    def make_tournament_slots(
        event_name: str, gender: str, advancers: List[str]
    ) -> List[List[dict]]:
        """本選トーナメント時程を生成（ラベル埋め）"""
        n = len(advancers)
        if n < 2:
            return []

        bracket = TournamentManager._next_pow2(n)
        first_pairs = TournamentManager.make_seeded_pairs(advancers, bracket)

        raw = []
        cur = bracket
        prev_match_names = None

        while cur >= 2:
            rname = TournamentManager._round_name(cur)
            m = cur // 2
            slot = []
            match_names = []

            for i in range(1, m + 1):
                mname = f"{rname}{i}" if m > 1 else rname
                match_names.append(mname)

                if prev_match_names is None:
                    a, b = first_pairs[i - 1]
                else:
                    src1 = prev_match_names[2 * (i - 1)]
                    src2 = prev_match_names[2 * (i - 1) + 1]
                    a = f"{src1} 勝ち"
                    b = f"{src2} 勝ち"

                slot.append({
                    "event": event_name,
                    "gender": gender,
                    "name": mname,
                    "teams": (a, b),
                    "phase": "tournament",
                })

            raw.append(slot)
            prev_match_names = match_names
            cur //= 2

        return raw

    @staticmethod
    def _extract_league(label: str) -> str:
        """順位ラベルからリーグ名を抽出"""
        m = re.match(r"^([A-Z]+)", str(label))
        return m.group(1) if m else str(label)

    @staticmethod

    # ✅ 修正
    @staticmethod
    def _pick_best_triple(pool: List[str]) -> Optional[Tuple[str, str, str]]:
        """3チーム選択：リーグ重複が最小になる組を選ぶ"""
        if len(pool) < 3:
            return None
        
        for i in range(min(len(pool) - 2, 20)):
            for j in range(i + 1, min(len(pool) - 1, i + 20)):
                for k in range(j + 1, min(len(pool), j + 20)):
                    tri = [pool[i], pool[j], pool[k]]
                    leagues = [TournamentManager._extract_league(x) for x in tri]
                    if len(set(leagues)) == 3:
                        return tuple(tri)
        
        # 異リーグが見つからない場合、任意の3つを返す
        if len(pool) >= 3:
            return tuple(pool[:3])
        return None

    @staticmethod
    def make_consolation_games(
        event_name: str, gender: str, losers_by_rank: Dict[int, List[str]]
    ) -> List[dict]:
        """敗者戦（下位のみ）を生成"""
        games = []
        idx = 1

        def pick_partner(i_label, pool):
            """異リーグパートナーを優先"""
            li = TournamentManager._extract_league(i_label)
            for j, cand in enumerate(pool):
                if TournamentManager._extract_league(cand) != li:
                    return j
            return 0

        for rank in sorted(losers_by_rank.keys()):
            pool = list(losers_by_rank[rank])
            if len(pool) < 2:
                continue

            # 奇数の場合、3チーム総当たり用を確保
            triple = None
            if len(pool) % 2 == 1 and len(pool) >= 3:
                triple = TournamentManager._pick_best_triple(pool)
                triple_set = set(triple)
                pool = [x for x in pool if x not in triple_set]

            # 残り（偶数）をペアリング
            while len(pool) >= 2:
                a = pool.pop(0)
                j = pick_partner(a, pool)
                b = pool.pop(j)

                games.append({
                    "event": event_name,
                    "gender": gender,
                    "name": f"敗者戦{idx}",
                    "teams": (a, b),
                    "phase": "consolation",
                })
                idx += 1

            # 3チーム総当たり
            if triple is not None:
                x, y, z = triple
                games.extend(TournamentManager._make_rr3_games(
                    event_name, gender, x, y, z, idx
                ))
                idx += 3

        return games

    @staticmethod
    def _make_rr3_games(
        event_name: str, gender: str, x: str, y: str, z: str, idx: int
    ) -> List[dict]:
        """3チーム総当たりの3試合を生成"""
        return [
            {
                "event": event_name,
                "gender": gender,
                "name": f"敗者戦{idx}:総当1",
                "teams": (x, y),
                "referee_hint": z,
                "is_rr3": True,
                "rr3_teams": (x, y, z),
                "phase": "consolation",
            },
            {
                "event": event_name,
                "gender": gender,
                "name": f"敗者戦{idx+1}:総当2",
                "teams": (x, z),
                "referee_hint": y,
                "is_rr3": True,
                "rr3_teams": (x, y, z),
                "phase": "consolation",
            },
            {
                "event": event_name,
                "gender": gender,
                "name": f"敗者戦{idx+2}:総当3",
                "teams": (y, z),
                "referee_hint": x,
                "is_rr3": True,
                "rr3_teams": (x, y, z),
                "phase": "consolation",
            },
        ]


# ============================================================
# Game Selection（試合選択）
# ============================================================

class GameSelector:
    """キューから衝突なく試合を選択"""

    def __init__(self, rng: random.Random, config: ScheduleConfig):
        self.rng = rng
        self.config = config

    def pick_k_games(
        self,
        queue: List[dict],
        k: int,
        forbidden: Set[str],
        lookahead: int,
    ) -> Optional[Tuple[List[int], List[dict], Set[str]]]:
        """
        キューから k個の非衝突試合を選択
        
        Returns:
            (picked_indices, picked_games, used_resources) or None
        """
        if k <= 0:
            return [], [], set()
        if not queue:
            return None

        # ★重要：予選が残っているのに敗者戦が混ざらないよう、
        # 先頭の phase と同じ phase の試合だけを候補にする
        head_phase = queue[0].get("phase")

        limit = min(lookahead, len(queue))
        candidates = []
        for i in range(limit):
            g = queue[i]
            if g.get("phase") != head_phase:
                continue
            used = ResourceManager.get_collision_resources(g)
            if used & forbidden:
                continue
            candidates.append((i, g, used))

        if len(candidates) < k:
            return None

        if k == 1:
            return self._pick_k1(candidates)
        if k == 2:
            return self._pick_k2(candidates)
        return self._pick_kn(candidates, k)

    def _pick_k1(self, candidates) -> Tuple[List[int], List[dict], Set[str]]:
        """k==1の場合：先頭topn範囲から選ぶ"""
        candidates.sort(key=lambda x: x[0])
        pool = candidates[: max(1, min(self.config.topn_k1, len(candidates)))]
        i, g, used = pool[self.rng.randrange(len(pool))]
        return [i], [g], set(used)

    def _pick_k2(self, candidates) -> Optional[Tuple[List[int], List[dict], Set[str]]]:
        """k==2の場合：ランダム試行 → 決定論探索へフォールバック"""
        candidates.sort(key=lambda x: x[0])
        pool = candidates[: max(2, min(self.config.topn_k1 * 2, len(candidates)))]

        best_score = None
        best = None

        # ランダム試行
        n = len(pool)
        for _ in range(self.config.pair_trials_k2):
            a = self.rng.randrange(n)
            b = self.rng.randrange(n - 1)
            if b >= a:
                b += 1
            i1, g1, u1 = pool[a]
            i2, g2, u2 = pool[b]
            if u1 & u2:
                continue
            union = set(u1 | u2)
            score = (max(i1, i2), len(union))

            if best_score is None or score < best_score:
                best_score = score
                best = (sorted([i1, i2]), [g1, g2], union)

        # フォールバック：決定論探索
        if best is None:
            for a in range(len(pool)):
                i1, g1, u1 = pool[a]
                for b in range(a + 1, len(pool)):
                    i2, g2, u2 = pool[b]
                    if u1 & u2:
                        continue
                    union = set(u1 | u2)
                    score = (max(i1, i2), len(union))
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (sorted([i1, i2]), [g1, g2], union)

        return best

    def _pick_kn(
        self, candidates, k: int
    ) -> Optional[Tuple[List[int], List[dict], Set[str]]]:
        """k>2の場合：バックトラック探索"""
        cand_copy = list(candidates)
        self.rng.shuffle(cand_copy)

        chosen = []
        used_total = set()

        def bt(start, left):
            nonlocal used_total
            if left == 0:
                return True
            for ci in range(start, len(cand_copy)):
                i, g, used = cand_copy[ci]
                if used & used_total:
                    continue
                chosen.append((i, g, used))
                prev = used_total
                used_total = used_total | used
                if bt(ci + 1, left - 1):
                    return True
                used_total = prev
                chosen.pop()
            return False

        if not bt(0, k):
            return None

        idxs = [x[0] for x in chosen]
        games = [x[1] for x in chosen]
        union = set().union(*(x[2] for x in chosen))
        return idxs, games, union


# ============================================================
# Schedule Building（スケジュール組立）
# ============================================================

class ScheduleError(Exception):
    """スケジューリング失敗時の例外"""

    # ✅ 修正
    def __init__(
        self,
        slot_no: int,
        active_events: List[str],
        detailed_failures: Dict[str, str],
        history: Optional[List[str]] = None,
    ):
        self.slot_no = slot_no
        self.active_events = active_events
        self.detailed_failures = detailed_failures
        self.history = history or []
    def __str__(self):
        lines = [f"時程 {self.slot_no} でスケジューリング失敗"]
        lines.append(f"  アクティブ種目: {', '.join(self.active_events)}")
        for ev, reason in self.detailed_failures.items():
            lines.append(f"  {ev}: {reason}")
        return "\n".join(lines)


class ScheduleBuilder:
    """試合キューからスケジュールを組み立てる"""

    def __init__(
        self,
        league_queues: Dict[str, List[dict]],
        per_event_parallel: Dict[str, int],
        per_event_parallel_consolation: Optional[Dict[str, int]],
        config: ScheduleConfig,
        seed: int = 0,
    ):
        self.league_q = {e: list(q) for e, q in league_queues.items()}
        self.per_event_parallel = per_event_parallel
        self.per_event_parallel_consolation = per_event_parallel_consolation or {}
        self.config = config
        self.rng = random.Random(seed)
        self.selector = GameSelector(self.rng, config)
        self.result: List[List[dict]] = []
        self.cooldown_forbidden: Set[str] = set()

    
        
    # ✅ 修正（ScheduleBuilder クラスのメソッドとして）
    def build(self) -> List[List[dict]]:
        """スケジュールを組み立てる"""
        logger.info(f"スケジューリング開始: {len(self.league_q)} 種目")
        try:
            if self.config.enable_repair and self.config.repair_mode == "global":
                tt = self._build_global_then_repair()
                logger.info(f"グローバルモード成功: {len(tt)} 時程")
                return tt
            else:
                tt = self._build_slot_by_slot()
                logger.info(f"スロットバイスロット成功: {len(tt)} 時程")
                return tt
        except ScheduleError as e:
            logger.error(f"スケジューリング失敗: {e}")
            raise
    


    def _build_slot_by_slot(self) -> List[List[dict]]:
        slot_no = 0
        self.result = []
        while self._has_remaining_games():
            slot_no += 1
            slot = self._build_slot(slot_no)
            self.result.append(slot)
            self._update_cooldown(slot)
        return self.result


    def _build_global_then_repair(self) -> List[List[dict]]:
        self.cooldown_forbidden = set()  # 明示的に初期化
        original_q = copy.deepcopy(self.league_q)
        best = None
        best_conflicts = 10**9

        for _ in range(self.config.global_repair_restarts):
            # ★毎回元に戻してから作る
            self.league_q = copy.deepcopy(original_q)
    
            initial = self._build_initial_timetable_ignore_conflicts()
            repairer = GlobalTimetableRepairer(rng=self.rng, config=self.config)
            repaired = repairer.repair(initial)
    
            remaining = len(repairer._find_conflicts(repaired))
            if remaining == 0:
                return repaired
    
            if remaining < best_conflicts:
                best_conflicts = remaining
                best = copy.deepcopy(repaired)
    
        if best is None:
            raise ScheduleError(0, [], {"GLOBAL": "初期配置/修理に失敗"})
        return best


    def _has_remaining_games(self) -> bool:
        return any(len(self.league_q[e]) > 0 for e in self.league_q.keys())

    def _build_initial_timetable_ignore_conflicts(self) -> List[List[dict]]:
        """初期時程を配置（衝突は無視）"""
        per_event_slots: Dict[str, List[List[dict]]] = {}
    
        for ev, q in self.league_q.items():
            qq = list(q)
    
            # ★初期シャッフル（phaseを混ぜない）
            if self.config.initial_shuffle and self.config.initial_shuffle_within_phase:
                prelim = [g for g in qq if g.get("phase") == "prelim"]
                cons   = [g for g in qq if g.get("phase") == "consolation"]
                self.rng.shuffle(prelim)
                self.rng.shuffle(cons)
                qq = prelim + cons
            elif self.config.initial_shuffle:
                self.rng.shuffle(qq)
    
            slots: List[List[dict]] = []
            i = 0
            while i < len(qq):
                phase = qq[i].get("phase")
                k = self._get_parallel(ev, phase == "consolation")
    
                chunk: List[dict] = []
                while i < len(qq) and len(chunk) < k and qq[i].get("phase") == phase:
                    chunk.append(qq[i])
                    i += 1
    
                if chunk:
                    slots.append(chunk)
    
            per_event_slots[ev] = slots
    
        # 時程番号で合体（event順もランダム化すると成功率↑）
        timetable: List[List[dict]] = []
        t = 0
        events = list(per_event_slots.keys())
        self.rng.shuffle(events)
    
        while any(t < len(per_event_slots[ev]) for ev in events):
            slot: List[dict] = []
            for ev in events:
                if t < len(per_event_slots[ev]):
                    slot.extend(per_event_slots[ev][t])
            timetable.append(slot)
            t += 1
        return timetable


    def _build_slot(self, slot_no: int) -> List[dict]:
        """1時程を組み立てる"""
        events = [e for e in self.league_q.keys() if len(self.league_q[e]) > 0]
        if not events:
            return []

        self.rng.shuffle(events)

        needs = {}
        for ev in events:
            q = self.league_q[ev]
            is_consolation = q[0].get("phase") == "consolation"
            pe = self._get_parallel(ev, is_consolation)
            needs[ev] = min(pe, len(q))

        if self.config.enable_repair:
            chosen = self._build_with_repair(events, needs)
        else:
            chosen = self._build_with_dfs(events, needs)
        # ★追加：どうしても k=2 が組めないなら k=1 に落として続行
        if chosen is None:
            relaxed_needs = dict(needs)
            changed = False
            for ev in events:
                if relaxed_needs[ev] >= 2:
                    relaxed_needs[ev] = 1
                    changed = True
            if changed:
                if self.config.enable_repair:
                    chosen = self._build_with_repair(events, relaxed_needs)
                else:
                    chosen = self._build_with_dfs(events, relaxed_needs)
                needs = relaxed_needs  # 後段の削除に合わせる

        if chosen is None:
            failures = {ev: f"{needs[ev]}試合必要" for ev in events}
            raise ScheduleError(slot_no, events, failures)

        slot_games: List[dict] = []
        for ev in events:
            idxs, games, _ = chosen[ev]
            for i in sorted(idxs, reverse=True):
                del self.league_q[ev][i]
            slot_games.extend(games)

        return slot_games

    def _get_parallel(self, event: str, is_consolation: bool) -> int:
        if is_consolation:
            return self.per_event_parallel_consolation.get(event, 1)
        return self.per_event_parallel.get(event, 2)

    def _build_with_repair(
        self, events: List[str], needs: Dict[str, int]
    ) -> Optional[Dict[str, Tuple]]:
        base_forbidden = self.cooldown_forbidden if self.config.enable_cooldown else set()

        chosen = {}
        for ev in events:
            pick = self.selector.pick_k_games(
                self.league_q[ev],
                needs[ev],
                base_forbidden,
                self.config.lookahead,
            )
            if pick is None:
                return None
            chosen[ev] = pick

        for _ in range(self.config.repair_iters):
            if not self._has_conflicts(chosen):
                return chosen

            has_conflict, bad_events = self._get_conflict_events(chosen)
            if not has_conflict:
                return chosen

            for ev in list(bad_events)[: self.config.repair_redraws]:
                other_used = set().union(*(chosen[e2][2] for e2 in chosen if e2 != ev))
                pick = self.selector.pick_k_games(
                    self.league_q[ev],
                    needs[ev],
                    base_forbidden | other_used,
                    self.config.lookahead,
                )
                if pick is not None:
                    chosen[ev] = pick
                    break

        return chosen

    def _build_with_dfs(
        self, events: List[str], needs: Dict[str, int]
    ) -> Optional[Dict[str, Tuple]]:
        base_forbidden = self.cooldown_forbidden if self.config.enable_cooldown else set()
        chosen = {}
        used_global = set()

        def dfs(idx: int) -> bool:
            nonlocal used_global
            if idx == len(events):
                return True

            ev = events[idx]
            k = needs[ev]
            forbidden = used_global | base_forbidden

            pick = self.selector.pick_k_games(
                self.league_q[ev], k, forbidden, self.config.lookahead
            )
            if pick is None:
                return False

            tries = [pick]
            if k == 1:
                for i in range(min(self.config.topn_k1, len(self.league_q[ev]))):
                    p = self.selector.pick_k_games(self.league_q[ev], k, forbidden, i + 1)
                    if p:
                        tries.append(p)

            for real_idxs, games, used in tries:
                chosen[ev] = (real_idxs, games, used)
                prev = used_global
                used_global |= used
                if dfs(idx + 1):
                    return True
                used_global = prev
                chosen.pop(ev, None)

            return False

        return chosen if dfs(0) else None

    def _has_conflicts(self, chosen: Dict[str, Tuple]) -> bool:
        has_conflict, _ = self._get_conflict_events(chosen)
        return has_conflict

    def _get_conflict_events(self, chosen: Dict[str, Tuple]) -> Tuple[bool, Set[str]]:
        used_owner = {}
        conflict_events = set()
        for ev, (_, _, used) in chosen.items():
            for r in used:
                if r in used_owner and used_owner[r] != ev:
                    conflict_events.add(ev)
                    conflict_events.add(used_owner[r])
                else:
                    used_owner[r] = ev
        return len(conflict_events) > 0, conflict_events

    def _update_cooldown(self, slot_games: List[dict]):
        if not self.config.enable_cooldown:
            self.cooldown_forbidden = set()
            return
        new_cd = set()
        for g in slot_games:
            new_cd |= ResourceManager.get_cooldown_resources(g)
        self.cooldown_forbidden = new_cd


class GlobalTimetableRepairer:
    """
    全体を先に並べて → 衝突があったら「同一event内」swap/moveで直す
    """

    def __init__(self, rng: random.Random, config: ScheduleConfig):
        self.rng = rng
        self.config = config

    def repair(self, timetable: List[List[dict]]) -> List[List[dict]]:
        tt = timetable  # 破壊的に直す

        def ensure_empty_slot():
            if not self.config.allow_insert_empty_slot:
                return None
            tt.append([])
            return len(tt) - 1

        for _ in range(self.config.global_repair_iters):
            conflicts = self._find_conflicts(tt)
            if not conflicts:
                return tt

            slot_i, game_i = conflicts[self.rng.randrange(len(conflicts))]
            if not (0 <= slot_i < len(tt)) or not (0 <= game_i < len(tt[slot_i])):
                continue

            g = tt[slot_i][game_i]
            ev = g.get("event")
            if not ev:
                continue

            candidate_slots = self._slots_containing_event(tt, ev)
            if slot_i in candidate_slots:
                candidate_slots.remove(slot_i)

            if not candidate_slots:
                empty_idx = ensure_empty_slot()
                if empty_idx is not None:
                    candidate_slots = [empty_idx]

            self.rng.shuffle(candidate_slots)

            fixed = False
            tries = 0

            for slot_j in candidate_slots:
                if tries >= self.config.global_repair_max_tries_per_conflict:
                    break
                tries += 1

                # move to empty slot
                if len(tt[slot_j]) == 0:
                    if self._move_is_ok(tt, slot_i, game_i, slot_j):
                        self._move_game(tt, slot_i, game_i, slot_j)
                        fixed = True
                        break
                    continue

                # swap within same event
                idxs2 = [k for k, h in enumerate(tt[slot_j]) if h.get("event") == ev]
                if not idxs2:
                    continue
                game_j = idxs2[self.rng.randrange(len(idxs2))]

                if self._swap_is_ok(tt, slot_i, game_i, slot_j, game_j):
                    self._swap_games(tt, slot_i, game_i, slot_j, game_j)
                    fixed = True
                    break

            if not fixed:
                continue

        return tt

    def _find_conflicts(self, tt: List[List[dict]]) -> List[Tuple[int, int]]:
        bad: List[Tuple[int, int]] = []
        prev_cd: Set[str] = set()

        for si, slot in enumerate(tt):
            # collision
            used = set()
            for gi, g in enumerate(slot):
                rset = ResourceManager.get_collision_resources(g)
                if used & rset:
                    bad.append((si, gi))
                used |= rset

            # cooldown
            if self.config.enable_cooldown:
                cur_cd = set()
                for gi, g in enumerate(slot):
                    cd = ResourceManager.get_cooldown_resources(g)
                    if cd & prev_cd:
                        bad.append((si, gi))
                    cur_cd |= cd
                prev_cd = cur_cd

        return bad

    def _slots_containing_event(self, tt: List[List[dict]], ev: str) -> List[int]:
        return [i for i, slot in enumerate(tt) if any(g.get("event") == ev for g in slot)]

    def _swap_is_ok(self, tt, si, gi, sj, gj) -> bool:
        g1 = tt[si][gi]
        g2 = tt[sj][gj]
        tt[si][gi], tt[sj][gj] = g2, g1
        ok = self._local_ok(tt, {si, sj, si - 1, si + 1, sj - 1, sj + 1})
        tt[si][gi], tt[sj][gj] = g1, g2
        return ok

    def _move_is_ok(self, tt, si, gi, sj_empty) -> bool:
        g1 = tt[si][gi]
        tt[sj_empty].append(g1)
        del tt[si][gi]
        ok = self._local_ok(tt, {si, sj_empty, si - 1, si + 1, sj_empty - 1, sj_empty + 1})
        tt[si].insert(gi, g1)
        tt[sj_empty].pop()
        return ok

    def _swap_games(self, tt, si, gi, sj, gj):
        tt[si][gi], tt[sj][gj] = tt[sj][gj], tt[si][gi]

    def _move_game(self, tt, si, gi, sj_empty):
        g1 = tt[si][gi]
        del tt[si][gi]
        tt[sj_empty].append(g1)

    def _local_ok(self, tt: List[List[dict]], slot_indices: Set[int]) -> bool:
        idxs = [i for i in slot_indices if 0 <= i < len(tt)]
        idxs = sorted(set(idxs))

        # collision
        for si in idxs:
            used = set()
            for g in tt[si]:
                rset = ResourceManager.get_collision_resources(g)
                if used & rset:
                    return False
                used |= rset

        # cooldown adjacency only
        if self.config.enable_cooldown:
            for si in idxs:
                if si == 0:
                    continue
                prev_cd = set()
                for g in tt[si - 1]:
                    prev_cd |= ResourceManager.get_cooldown_resources(g)
                for g in tt[si]:
                    if ResourceManager.get_cooldown_resources(g) & prev_cd:
                        return False

        return True

# ============================================================
# Referee Assignment（審判割当）
# ============================================================

class RefereeAssigner:
    """審判を割り当てる"""

    def __init__(self, all_classes: List[str], seed: int = 0):
        self.all_classes = all_classes
        self.rng = random.Random(seed)
        self.ref_count = defaultdict(int)

    def assign(self, raw_slots: List[List[dict]]) -> List[List[dict]]:
        """
        全時程の審判を割り当てる
        
        規則：
        - referee_hint が設定されている場合は絶対固定（上書きしない）
        - それ以外は前時程参加チーム → 将来の予選参加チームの順で選ぶ
        """
        prev_by_event = defaultdict(list)

        for slot in raw_slots:
            games_by_event = defaultdict(list)
            for g in slot:
                games_by_event[g.get("event", "")].append(g)

            for ev, games in games_by_event.items():
                # referee_hint が固定されている試合を処理
                fixed_games = []
                normal_games = []

                for g in games:
                    if g.get("referee_hint"):
                        g["referee"] = g["referee_hint"]
                        g["_ref_fixed"] = True
                        self.ref_count[g["referee_hint"]] += 1
                        fixed_games.append(g)
                    else:
                        normal_games.append(g)

                # 通常試合に審判を割当
                if normal_games:
                    ref_gamesets = prev_by_event.get(ev, [])
                    self._assign_to_games(normal_games, ref_gamesets)

            # 次時程用
            new_prev = defaultdict(list)
            for g in slot:
                ev = g.get("event", "")
                teams = ResourceManager.get_teams_only(g)
                if teams:
                    new_prev[ev].append(teams)
            prev_by_event = new_prev

        return raw_slots

    def _assign_to_games(self, games: List[dict], ref_gamesets: List[set]):
        """1種目内の試合に審判を割当"""
        if len(ref_gamesets) >= 2 and len(games) >= 2:
            # 参照元が複数あれば、それぞれから選ぶ
            self._assign_with_reference(games, ref_gamesets)
        else:
            # 通常割当
            for g in games:
                g["referee"] = self._pick_referee(g)
                self.ref_count[g["referee"]] += 1

    def _pick_referee(self, game: dict) -> str:
        """試合に審判を選ぶ"""
        a, b = game.get("teams", (None, None))
        blocked = {a, b}
        candidates = set(self.all_classes) - blocked
        if not candidates:
            candidates = set(self.all_classes)
        return min(candidates, key=lambda t: (self.ref_count[t], self.rng.random()))

    # ✅ 改善
    def _assign_with_reference(self, games: List[dict], 
                               ref_gamesets: List[set]):
        """
        複数の参照セットから審判を選ぶ
        
        規則:
        - 最初の2試合: ref_gamesets[0], [1] から優先選択
        - 残りの試合: 全セットの和から選択
        - 選手が試合に参加している場合は除外
        """
        s0, s1 = ref_gamesets[0], ref_gamesets[1]

        for i, g in enumerate(games[:2]):
            a, b = g.get("teams", (None, None))
            blocked = {a, b}
            ref_set = s0 if i == 0 else s1
            candidates = {t for t in ref_set if t is not None} - blocked
            if not candidates:
                candidates = set(self.all_classes) - blocked
            referee = min(
                candidates or set(self.all_classes),
                key=lambda t: (self.ref_count[t], self.rng.random()),
            )
            g["referee"] = referee
            self.ref_count[referee] += 1

        # 残り試合
        union_set = set().union(*ref_gamesets)
        for g in games[2:]:
            a, b = g.get("teams", (None, None))
            blocked = {a, b}
            candidates = {t for t in union_set if t is not None} - blocked
            if not candidates:
                candidates = set(self.all_classes) - blocked
            referee = min(
                candidates or set(self.all_classes),
                key=lambda t: (self.ref_count[t], self.rng.random()),
            )
            g["referee"] = referee
            self.ref_count[referee] += 1


class TournamentRefereeAssigner:
    """本選の審判を「敗者」で自動設定"""

    @staticmethod
    def assign(raw_slots: List[List[dict]]) -> List[List[dict]]:
        """本選2時程目以降の審判を「前試合敗者」に変更"""
        prev_tourn_names_by_event = defaultdict(list)

        for slot in raw_slots:
            tourn_games_by_event = defaultdict(list)

            for g in slot:
                name = g.get("name", "")
                # 予選・敗者戦・BYEは除外
                if any(x in name for x in ["予選", "敗者戦", "BYE"]):
                    continue

                ev = g.get("event", "")
                tourn_games_by_event[ev].append(g)

            # 上書き（2時程目以降）
            for ev, games in tourn_games_by_event.items():
                prev_names = prev_tourn_names_by_event.get(ev, [])
                if prev_names:
                    for i, g in enumerate(games):
                        ref_src = prev_names[i] if i < len(prev_names) else prev_names[-1]
                        g["referee"] = f"{ref_src} 負け"

            # 次用
            new_prev = defaultdict(list)
            for ev, games in tourn_games_by_event.items():
                for g in games:
                    new_prev[ev].append(g.get("name", "本選"))
            prev_tourn_names_by_event = new_prev

        return raw_slots


# ============================================================
# Event Orchestration（イベント統合）
# ============================================================

class EventScheduler:
    """1種目のスケジュール全体を生成"""

    @staticmethod
    def create_event_schedule(
        event_name: str,
        event_info: dict,
        classes: List[Tuple[str, str]],
        league_seed: int = 0,
    ) -> dict:
        """
        1種目について、リーグ→予選→本選→敗者戦を全て生成
        
        Args:
            event_name: 種目名
            event_info: {participants: [...], gender: 'X'/'M'/'F', min_teams: 3, tournament_max_teams: 8, ...}
            classes: [('1A', '...'), ('1B', '...'), ...]
        """
        participants = event_info.get("participants", [])
        gender = event_info.get("gender", "X")
        min_teams = event_info.get("min_teams", 3)
        max_teams = event_info.get("tournament_max_teams", 8)

        # リーグ分け
        leagues = LeagueManager.split_into_leagues(
            participants, min_teams, seed=league_seed
        )

        # 予選試合
        league_games = LeagueManager.make_league_games(event_name, gender, leagues)

        # 本選進出決定
        advancers, losers_by_rank = TournamentManager.build_advancers(
            leagues, max_teams
        )

        # 本選トーナメント
        tourn_slots = TournamentManager.make_tournament_slots(
            event_name, gender, advancers
        )

        # 敗者戦
        consolation_games = TournamentManager.make_consolation_games(
            event_name, gender, losers_by_rank
        )
        if len(league_games) == 0 and len(participants) >= 2:
            print(f"[WARN] {event_name}: 予選試合が0件")
            print("  participants:", participants)
            print("  min_teams:", min_teams)
            print("  leagues:", {L: len(ts) for L, ts in leagues.items()})

        return {
            "event": event_name,
            "gender": gender,
            "leagues": leagues,
            "league_games": league_games,
            "tournament_slots": tourn_slots,
            "consolation_games": consolation_games,
            "advancers": advancers,
            "losers_by_rank": losers_by_rank,
            "parallel": event_info.get("parallel", 2),
            "consolation_parallel": event_info.get("consolation_parallel", 1),
        }


# ============================================================
# Main Orchestration（メイン統合）
# ============================================================

class TimetableBuilder:
    """全大会のタイムテーブル生成"""

    def __init__(
        self,
        events: Dict[str, dict],
        classes: List[Tuple[str, str]],
        time_config: TimeConfig = None,
        schedule_config: ScheduleConfig = None,
    ):
        self.events = events
        self.classes = classes
        self.time_config = time_config or TimeConfig()
        self.schedule_config = schedule_config or ScheduleConfig()

    def build(self, league_seed: int = 0) -> Tuple[List[List[dict]], dict]:
        """
        タイムテーブル全体を生成
        
        Returns:
            (final_timetable, info)
        """
        # ステップ1：各種目のスケジュール原案を生成
        all_event_results = []
        for event_name, event_info in self.events.items():
            es = EventScheduler.create_event_schedule(
                event_name, event_info, self.classes, league_seed
            )
            all_event_results.append(es)

        # ステップ2：試合キューを構築
        league_q, tourn_q = self._build_queues(all_event_results)

        # ステップ3：予選 + 敗者戦をスケジューリング
        per_event_parallel = {
            ev["event"]: ev.get("parallel", 2) for ev in all_event_results
        }
        per_event_parallel_cons = {
            ev["event"]: ev.get("consolation_parallel", 1)
            for ev in all_event_results
        }

        builder = ScheduleBuilder(
            league_q,
            per_event_parallel,
            per_event_parallel_cons,
            self.schedule_config,
            seed=league_seed + 97,
        )
        prelim_raw = builder.build()
        if sum(len(slot) for slot in prelim_raw) == 0:
            raise RuntimeError("予選+敗者戦の試合が0件です（入力/リーグ分けを確認）")


        # ステップ4：本選開始時刻の調整
        after_prelim = TimeManager.end_time_after_slots(
            self.time_config.start_time,
            len(prelim_raw),
            self.time_config.match_min,
            self.time_config.change_min,
        )

        if self.time_config.enforce_tournament_start:
            over = TimeManager.minutes_between(
                self.time_config.tournament_start_time, after_prelim
            )
            if over > 0:
                raise RuntimeError(
                    f"本選を {self.time_config.tournament_start_time} に開始予定が、"
                    f"予選が {after_prelim} までかかり {over} 分オーバー"
                )
            empty = TimeManager.num_empty_slots_to_reach(
                after_prelim,
                self.time_config.tournament_start_time,
                self.time_config.match_min,
                self.time_config.change_min,
            )
            for _ in range(empty):
                prelim_raw.append([])

        # ステップ5：本選を追加
        tourn_raw = self._schedule_tournaments(tourn_q, per_event_parallel)

        # ステップ6：統合
        raw = prelim_raw + tourn_raw

        # ステップ7：審判割当
        all_class_names = [c[0] for c in self.classes]
        assigner = RefereeAssigner(all_class_names, seed=league_seed + 97)
        raw = assigner.assign(raw)
        raw = TournamentRefereeAssigner.assign(raw)

        # ステップ8：時刻付与
        final = TimeManager.add_times_to_timetable(
            raw,
            self.time_config.start_time,
            self.time_config.match_min,
            self.time_config.change_min,
        )

        return final, {
            "success": True,
            "league_seed": league_seed,
            "num_slots": len(final),
            "all_event_results": all_event_results,
        }

    def _build_queues(
        self, all_event_results
    ) -> Tuple[Dict[str, List[dict]], Dict[str, List[List[dict]]]]:
        """試合キューを構築"""
        league_q = {}
        tourn_q = {}

        for ev in all_event_results:
            name = ev["event"]
            gender = ev.get("gender", "X")

            # 予選 + 敗者戦を連結
            L = []
            for g in ev.get("league_games", []):
                gg = dict(g)
                gg["event"] = name
                gg["gender"] = gg.get("gender", gender)
                L.append(gg)

            C = []
            for g in ev.get("consolation_games", []):
                gg = dict(g)
                gg["event"] = name
                gg["gender"] = gg.get("gender", gender)
                C.append(gg)

            league_q[name] = L + C
            tourn_q[name] = ev.get("tournament_slots", [])

        return league_q, tourn_q

    def _schedule_tournaments(
        self, tourn_q: Dict[str, List[List[dict]]], per_event_parallel: Dict[str, int]
    ) -> List[List[dict]]:
        """本選トーナメントを時程化"""
        events = list(tourn_q.keys())
        streams = {e: tourn_q[e][:] for e in events}
        idx = {e: 0 for e in events}

        out = []
        while any(idx[e] < len(streams[e]) for e in events):
            slot = []
            for e in events:
                if idx[e] < len(streams[e]):
                    slot.extend(streams[e][idx[e]])
                    idx[e] += 1
            if slot:
                out.append(slot)

        return out


# ============================================================
# Retry Logic（リトライロジック）
# ============================================================

class RobustTimetableBuilder:
    """リトライ機能付きタイムテーブルビルダー"""

    def __init__(
        self,
        events: Dict[str, dict],
        classes: List[Tuple[str, str]],
        time_config: TimeConfig = None,
        schedule_config: ScheduleConfig = None,
    ):
        self.events = events
        self.classes = classes
        self.time_config = time_config or TimeConfig()
        self.schedule_config = schedule_config or ScheduleConfig()

    def build_with_retries(
        self, base_seed: int = 0, seed: Optional[int] = None
    ) -> Tuple[Optional[List[List[dict]]], dict]:
        """リトライ付きでタイムテーブルを生成"""
        last_error = None
        prev_leagues = {}
        # 互換：呼び出し側が seed=... を渡してきた場合は base_seed として扱う
        if seed is not None:
            base_seed = seed
        for attempt in range(self.schedule_config.league_attempts):
            league_seed = base_seed + attempt * 10007

            try:
                builder = TimetableBuilder(
                    self.events,
                    self.classes,
                    self.time_config,
                    self.schedule_config,
                )
                tt, info = builder.build(league_seed)

                # リーグ分けが前回と同じならスキップ
                all_event_results = info.get("all_event_results", [])
                current_leagues = {
                    ev["event"]: ev["leagues"] for ev in all_event_results
                }
                if current_leagues == prev_leagues:
                    continue
                prev_leagues = current_leagues

                return tt, {
                    **info,
                    "success": True,
                    "league_attempt": attempt,
                    "league_seed": league_seed,
                    "num_slots": len(tt),
                }

            except (ScheduleError, RuntimeError) as e:
                last_error = str(e)
                continue

        return None, {
            "success": False,
            "league_attempts": self.schedule_config.league_attempts,
            "last_error": last_error,
        }


# ============================================================
# Output（出力）
# ============================================================

class TimetableExporter:
    """タイムテーブルをCSV/DataFrameに出力"""

    @staticmethod
    def to_csv(
        final_timetable: List[List[dict]],
        all_event_results: List[dict],
        classes: List[Tuple[str, str]],
        leagues_csv: str = "leagues.csv",
        timetable_csv: str = "timetable.csv",
    ):
        """CSV形式で出力"""
        # leagues.csv
        with open(leagues_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["event", "league", "team"])
            for ev in sorted(all_event_results, key=lambda x: x["event"]):
                for L in sorted(ev["leagues"].keys()):
                    for team in ev["leagues"][L]:
                        w.writerow([ev["event"], L, team])

        # timetable.csv
        with open(timetable_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([
                "slot_no", "start", "end", "event", "name",
                "team_a", "team_b", "referee", "phase", "gender"
            ])
            for slot_no, slot in enumerate(final_timetable, 1):
                for g in slot:
                    a, b = g.get("display_teams", g.get("teams", (None, None)))
                    w.writerow([
                        slot_no,
                        g.get("start", ""),
                        g.get("end", ""),
                        g.get("event", ""),
                        g.get("name", ""),
                        "" if a is None else str(a),
                        "" if b is None else str(b),
                        g.get("referee", ""),
                        g.get("phase", ""),
                        g.get("gender", ""),
                    ])

        print(f"✅ CSV出力: {leagues_csv}, {timetable_csv}")

    @staticmethod
    def to_dataframes(
        final_timetable: List[List[dict]],
        all_event_results: List[dict],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """DataFrame形式で出力"""
        # leagues_df
        leagues_rows = []
        for ev in sorted(all_event_results, key=lambda x: x["event"]):
            for L in sorted(ev["leagues"].keys()):
                for team in ev["leagues"][L]:
                    leagues_rows.append([ev["event"], L, team])

        leagues_df = pd.DataFrame(
            leagues_rows, columns=["event", "league", "team"]
        )

        # timetable_df
        tt_rows = []
        for slot_no, slot in enumerate(final_timetable, 1):
            for g in slot:
                a, b = g.get("display_teams", g.get("teams", (None, None)))
                tt_rows.append([
                    slot_no,
                    g.get("start", ""),
                    g.get("end", ""),
                    g.get("event", ""),
                    g.get("name", ""),
                    "" if a is None else str(a),
                    "" if b is None else str(b),
                    g.get("referee", ""),
                    g.get("phase", ""),
                    g.get("gender", ""),
                ])

        timetable_df = pd.DataFrame(
            tt_rows,
            columns=[
                "slot_no", "start", "end", "event", "name",
                "team_a", "team_b", "referee", "phase", "gender"
            ],
        )

        return leagues_df, timetable_df


# ============================================================
# Validation（検証）
# ============================================================

class GameValidator:
    """スケジュールの妥当性をチェック"""

    @staticmethod
    def verify_min_games(
        all_event_results: List[dict],
        final_timetable: List[List[dict]],
        min_games: int = 3,
        verbose: bool = True,
    ) -> Tuple[Dict, List[str]]:
        """各チームの試合数が最低値を満たすか確認"""
        game_count = defaultdict(lambda: defaultdict(int))
        warnings = []

        for ev in all_event_results:
            event_name = ev["event"]
            leagues = ev.get("leagues", {})
            advancers = set(ev.get("advancers", []))
            consolation_games = ev.get("consolation_games", [])

            # 予選：各リーグの順位ラベルは (リーグ人数-1) 試合
            for L, teams in leagues.items():
                games_per_team = max(0, len(teams) - 1)
                for rank in range(1, len(teams) + 1):
                    label = f"{L}{rank}位"
                    game_count[event_name][label] += games_per_team

            # 敗者戦
            for g in consolation_games:
                for t in g.get("teams", (None, None)):
                    if t:
                        game_count[event_name][t] += 1

            # 本選（最低保証）
            for label in advancers:
                if label:
                    game_count[event_name][label] += 1

            # 警告チェック
            for label, cnt in game_count[event_name].items():
                if cnt < min_games:
                    warnings.append(
                        f"{event_name} の {label}: {cnt}試合 "
                        f"（最小 {min_games}試合必要）"
                    )

        if verbose:
            print("\n【試合数確認】")
            print("=" * 70)
            for event_name in sorted(game_count.keys()):
                print(f"\n{event_name}:")
                for label in sorted(game_count[event_name].keys()):
                    cnt = game_count[event_name][label]
                    status = "✅" if cnt >= min_games else "⚠️"
                    print(f"  {status} {label}: {cnt}試合")

            print("\n" + "=" * 70)
            if warnings:
                print("❌ 以下が最低試合数を満たしていません:")
                for w in warnings:
                    print(f"  {w}")
            else:
                print(f"✅ 全チームが最低 {min_games}試合以上確保")

        return dict(game_count), warnings


# ============================================================
# Print Utilities（表示ユーティリティ）
# ============================================================

class TimetablePrinter:
    """タイムテーブルを整形して表示"""

    @staticmethod
    def print_timetable(
        timetable: List[List[dict]], title: str = "タイムテーブル"
    ):
        """タイムテーブルを表示"""
        print("=" * 80)
        print(title)
        print("=" * 80)
        for si, slot in enumerate(timetable, 1):
            if not slot:
                continue
            start = slot[0].get("start", "??:??")
            end = slot[0].get("end", "??:??")
            print(f"\n【時程 {si}】{start}-{end}")
            for g in slot:
                ev = g.get("event", "")
                name = g.get("name", "")
                a, b = g.get("display_teams", g.get("teams", (None, None)))
                ref = g.get("referee", "未定")
                print(f"  {ev} | {name}: {a} vs {b} | 審判: {ref}")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # 設定
    time_config = TimeConfig(
        start_time="09:00",
        match_min=10,
        change_min=3,
        tournament_start_time="13:00",
    )
    schedule_config = ScheduleConfig(
        lookahead=80,
        league_attempts=30,
        repair_mode="global",
        global_repair_iters=5000,
        global_repair_max_tries_per_conflict=300,
    )


    # クラス定義
    classes = [
        ("1A", "1年A組"),
        ("1B", "1年B組"),
        ("2A", "2年A組"),
        ("2B", "2年B組"),
    ]

    # 種目定義
    events = {
        "男子シングルス": {
            "gender": "M",
            "participants": ["1A", "1B", "2A", "2B"],
            "min_teams": 2,
            "tournament_max_teams": 4,
            "parallel": 2,
            "consolation_parallel": 1,
        },
        "女子シングルス": {
            "gender": "F",
            "participants": ["1A", "1B", "2A"],
            "min_teams": 2,
            "tournament_max_teams": 4,
            "parallel": 2,
            "consolation_parallel": 1,
        },
    }

    # ビルダーを実行
    builder = RobustTimetableBuilder(
        events, classes, time_config, schedule_config
    )
    timetable, info = builder.build_with_retries(seed=42)

    if info["success"]:
        TimetablePrinter.print_timetable(timetable, "スポーツフェスティバル タイムテーブル")
        TimetableExporter.to_csv(
            timetable,
            info.get("all_event_results", []),
            classes,
        )
        print(f"\n✅ スケジューリング成功（試行回数: {info['league_attempt']}）")
    else:
        print(f"\n❌ スケジューリング失敗: {info['last_error']}")
