"""出牌后的牌型计划播报：自己准备做什么牌型，三家可能在做什么牌型。

只用于日志，不参与决策；所有概率都来自记牌器的条件牌池，是估计而不是透视。
"""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict

from game_baque.ev import ExpectedValueEngine
from game_baque.models import RANKS as RANK_ORDER
from game_baque.models import Card
from game_baque.rules import EIGHT_SPECIAL, SPECIAL_6_2, evaluate_hu
from game_baque.strategy import hand_potential
from game_baque.tracker import CardKey, TableTracker, card_from_key

# 出牌后紧接着行动的是右边（下家），左边在自己之前行动（上家）。
SEAT_NAMES = {"right": "下家", "top": "对家", "left": "上家"}


def _label(key: CardKey) -> str:
    return card_from_key(key).label


def _join_labels(counter: Counter[CardKey], limit: int = 6) -> str:
    items = [f"{_label(key)}×{count}" if count > 1 else _label(key) for key, count in counter.most_common(limit)]
    return "、".join(items) or "无"


def own_plan_line(
    engine: ExpectedValueEngine,
    kept: list[Card],
    gold_ranks: tuple[str, ...],
    tracker: TableTracker,
    *,
    samples: int = 160,
    deadline: float | None = None,
    rng: random.Random | None = None,
) -> str:
    """自己保留 7 张后的计划：已听牌列出胡哪些牌，否则估计几次摸牌内最可能成的牌型。"""
    hand_text = " ".join(card.label for card in kept)
    try:
        unseen = tracker.unseen_counts()
    except RuntimeError:
        unseen = Counter()

    waits: dict[str, Counter[CardKey]] = defaultdict(Counter)
    best_score: dict[str, float] = {}
    for key, count in unseen.items():
        if count <= 0:
            continue
        hu = evaluate_hu(kept + [card_from_key(key)], gold_ranks)
        if hu is None:
            continue
        waits[hu.pattern][key] += count
        # 显示自摸（×2）分值；拿亮牌或别人打出时减半。
        best_score[hu.pattern] = max(
            best_score.get(hu.pattern, 0.0), engine.hu_value(hu, self_draw=True)
        )
    if waits:
        total = sum(sum(counter.values()) for counter in waits.values())
        parts = [
            f"{pattern}（等 {_join_labels(counter)}，约 {best_score[pattern]:.0f} 分）"
            for pattern, counter in sorted(waits.items(), key=lambda item: -best_score[item[0]])
        ]
        return f"自己 [{hand_text}]：已听牌，未见 {total} 张能胡 → " + "；".join(parts)

    # 出牌后要先等三家各摸一张，自己的下一摸是牌堆往下第 4 张。
    draw_dist = tracker.draw_distribution(skip=3)
    horizon = engine.own_draws_left(tracker.deck_count, draw_now=False)
    if not draw_dist or horizon <= 0:
        return f"自己 [{hand_text}]：未听牌，牌堆已不够再摸，只能防守。"
    rng = rng or random.Random()
    keys = list(draw_dist)
    weights = [draw_dist[key] for key in keys]
    patterns: Counter[str] = Counter()
    completed = 0
    for _ in range(samples):
        if completed and deadline is not None and time.monotonic() >= deadline:
            break
        completed += 1
        hand = list(kept)
        for depth in range(horizon):
            hand8 = hand + [card_from_key(rng.choices(keys, weights=weights, k=1)[0])]
            hu = evaluate_hu(hand8, gold_ranks)
            if hu is not None:
                patterns[hu.pattern] += 1
                break
            if depth + 1 < horizon:
                drop = max(
                    range(8),
                    key=lambda i: hand_potential(
                        [c for j, c in enumerate(hand8) if j != i],
                        gold_ranks,
                        prefer_flush=engine.prefer_flush,
                        chase_big=engine.chase_big,
                        prefer_pinghu=engine.prefer_pinghu,
                    ),
                )
                hand = [c for j, c in enumerate(hand8) if j != drop]
    if not patterns:
        return f"自己 [{hand_text}]：未听牌，{horizon} 次摸牌内很难成牌，先整理结构。"
    total = sum(patterns.values()) / completed
    parts = [f"{pattern} {count / completed:.0%}" for pattern, count in patterns.most_common(3)]
    return (
        f"自己 [{hand_text}]：未听牌，{horizon} 次摸牌内成牌约 {total:.0%}，"
        f"最可能做成 " + "、".join(parts)
    )


_SUIT_SYMBOL = {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}


def _best_flush(hand: list[Card]) -> tuple[int, str] | None:
    """最有希望的同花顺子：(已有张数含癞子, 描述)。看 6 张窗口（6+2、5+3 的主体）。"""
    wild = sum(card.wild for card in hand)
    ranks = list(RANK_ORDER)
    best: tuple[int, str] | None = None
    for suit, symbol in _SUIT_SYMBOL.items():
        held = {card.rank for card in hand if not card.wild and card.suit == suit}
        for start in range(0, len(ranks) - 5):
            window = ranks[start : start + 6]
            have = [rank for rank in window if rank in held]
            if len(have) < 2:
                continue
            missing = [rank for rank in window if rank not in held]
            used_wild = min(wild, len(missing))
            covered = len(have) + used_wild
            lack = missing[used_wild:]
            text = (
                f"同花顺 {symbol}{window[0]}~{window[-1]}：已有 {len(have)} 张"
                + (f"+癞子{used_wild}" if used_wild else "")
                + ("，成形" if not lack else f"，还差 {'、'.join(symbol + rank for rank in lack)}")
            )
            if best is None or covered > best[0]:
                best = (covered, text)
    return best


def target_line(engine: ExpectedValueEngine, hand: list[Card], gold_ranks: tuple[str, ...]) -> str:
    """本局主攻方向（与 EV 的同花顺优先/追大牌模式一致），只用于日志。"""
    nonwild = [card for card in hand if not card.wild]
    wild = len(hand) - len(nonwild)
    flush = _best_flush(hand)
    flush_text = flush[1] if flush else "暂无成形的同花顺子"
    if engine.prefer_pinghu:
        ranks = Counter(card.rank for card in nonwild)
        groups = [f"{rank}×{count}" for rank, count in ranks.most_common() if count >= 2]
        group_text = "、".join(groups) if groups else "暂无对子"
        wild_text = f"；手里有癞子 {wild} 张（平胡用不上癞子）" if wild else ""
        return f"平胡优先（对手豆少，胡得快最重要）：对子/三张 {group_text}{wild_text}；备选 {flush_text}"
    if not engine.chase_big:
        mode = "同花顺优先（能胡的牌越多越好，无癞子更好）"
        return f"{mode}：{flush_text}"
    gold = Counter(card.rank for card in nonwild if card.rank in gold_ranks)
    gold_text = (
        f"金牌 {sum(gold.values())} 张（{'、'.join(f'{rank}×{count}' for rank, count in gold.most_common())}）"
        if gold
        else "暂无金牌"
    )
    ranks = Counter(card.rank for card in nonwild)
    special: list[str] = []
    for rank, (name, _multiplier) in EIGHT_SPECIAL.items():
        if ranks[rank] + wild >= 4:
            special.append(f"{name}（{rank}×{ranks[rank]}" + (f"+癞子{wild}" if wild else "") + "）")
    for pair, (name, _multiplier) in SPECIAL_6_2.items():
        first, second = tuple(pair)
        if ranks[first] and ranks[second] and ranks[first] + ranks[second] + wild >= 4:
            special.append(f"{name}（{first}×{ranks[first]}、{second}×{ranks[second]}）")
    big = "；特殊牌型方向：" + "、".join(special) if special else ""
    return f"追大牌（豆够多，全金牌/特殊牌型优先）：{gold_text}{big}；备选 {flush_text}"


def opponent_plan_lines(
    engine: ExpectedValueEngine,
    gold_ranks: tuple[str, ...],
    tracker: TableTracker,
    *,
    deadline: float | None = None,
) -> list[str]:
    """三家：已知存牌、出过的牌，以及下一次摸牌可能胡的牌型与要等的牌。

    决策时算完的推断会命中缓存；没命中时按 deadline 截断采样，只求日志够用。
    """
    # 参数与出牌决策一致，可直接命中刚算过的缓存。
    inference = {
        item.seat: item
        for item in engine.infer_opponents(
            tracker, gold_ranks, top_patterns=4, top_cards=6, deadline=deadline
        )
    }
    lines: list[str] = []
    for seat in ("left", "top", "right"):
        opp = tracker.opponents.get(seat)
        if opp is None:
            continue
        name = f"{SEAT_NAMES[seat]}({seat})"
        known = _join_labels(Counter(opp.known_hand))
        discards = "、".join(_label(key) for key in opp.discards[-4:]) or "无"
        if opp.finished:
            lines.append(f"{name}：已胡牌/离场。")
            continue
        item = inference.get(seat)
        head = f"{name}：存牌[{known}] 未知 {opp.unknown_hand_count} 张，最近打出[{discards}]"
        if item is None or item.any_hu_probability_next_draw <= 0 or not item.patterns:
            lines.append(head + "；暂无听牌迹象。")
            continue
        patterns = "；".join(
            f"{p.pattern} {p.probability_next_draw:.1%}（等 {'、'.join(_label(k) for k in p.dangerous_cards[:4])}）"
            for p in item.patterns[:3]
        )
        danger = "、".join(f"{_label(key)} {mass:.1%}" for key, mass in item.dangerous_cards[:4])
        lines.append(
            f"{head}；下一摸胡牌 {item.any_hu_probability_next_draw:.1%}，可能在做：{patterns}；"
            f"最怕放出：{danger}"
        )
    return lines
