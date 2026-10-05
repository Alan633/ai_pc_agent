"""对照游戏内“番型/推荐”官方番型表（2026-10-05 截图）逐条验证规则引擎。

官方表：
- 特殊番型 ×100：独一无二(8A) 君临天下(8K) 十全十美(8个10) 八方来财(8个8)
- 特殊番型 ×32（6+2 同点）：顶峰相见(K/A) 心心相连(J/Q) 十拿九稳(9/10) 六事兴旺(6/7)
- 八方来贺 ×50：完整 8 张同花顺或 8 炸
- 6+2：比翼为邻(两组相邻同点) ×16、六朝金粉(八张同颜色) ×16、六六大顺 ×8
- 5+3：永恒相随(相邻同点) ×8、五谷丰登(同颜色) ×8、五福临门 ×4
- 4+4：二龙腾飞(相邻同点) ×4、四季发财(同颜色) ×4、四季如春 ×2
- 3+3+2：平胡 ×1，其中 2 必须为一对
"""

from __future__ import annotations

import pytest

from game_baque.models import Card
from game_baque.rules import evaluate_hu

# 金牌取用不到的点数，避免金牌倍数干扰牌型判断。
NO_GOLD = ("6", "7", "8")


def cards(spec: str) -> list[Card]:
    """"A♠ A♥ 10♦" → Card 列表。"""
    suit_of = {"♠": "S", "♥": "H", "♣": "C", "♦": "D"}
    return [Card(rank=token[:-1], suit=suit_of[token[-1]]) for token in spec.split()]


OFFICIAL_EXAMPLES = [
    # 截图“特殊番型”原样示例
    ("A♠ A♠ A♣ A♣ A♥ A♥ A♦ A♦", "独一无二", 100),
    ("K♠ K♠ K♣ K♣ K♥ K♥ K♦ K♦", "君临天下", 100),
    ("10♠ 10♠ 10♣ 10♣ 10♥ 10♥ 10♦ 10♦", "十全十美", 100),
    ("8♠ 8♠ 8♣ 8♣ 8♥ 8♥ 8♦ 8♦", "八方来财", 100),
    ("K♠ K♠ K♥ K♥ K♣ K♣ A♦ A♦", "顶峰相见", 32),
    ("A♠ A♠ A♥ A♥ A♣ A♣ K♦ K♦", "顶峰相见", 32),
    ("J♠ J♠ J♥ J♥ J♣ J♣ Q♦ Q♦", "心心相连", 32),
    ("Q♠ Q♠ Q♥ Q♥ Q♣ Q♣ J♦ J♦", "心心相连", 32),
    ("9♠ 9♠ 9♥ 9♥ 9♣ 9♣ 10♦ 10♦", "十拿九稳", 32),
    ("10♠ 10♠ 10♥ 10♥ 10♣ 10♣ 9♦ 9♦", "十拿九稳", 32),
    ("6♠ 6♠ 6♥ 6♥ 6♣ 6♣ 7♦ 7♦", "六事兴旺", 32),
    ("7♠ 7♠ 7♥ 7♥ 7♣ 7♣ 6♦ 6♦", "六事兴旺", 32),
    # “番型总览”按文字说明构造的示例
    ("6♥ 7♥ 8♥ 9♥ 10♥ J♥ Q♥ K♥", "八方来贺", 50),  # 8 张同花顺
    ("9♠ 9♠ 9♥ 9♥ 9♣ 9♣ 8♦ 8♦", "比翼为邻", 16),  # 6+2 相邻同点
    ("6♥ 7♥ 8♥ 9♥ 10♥ J♥ Q♦ Q♦", "六朝金粉", 16),  # 6+2 八张同色（红）
    ("6♥ 7♥ 8♥ 9♥ 10♥ J♥ Q♠ Q♣", "六六大顺", 8),  # 6+2 杂色、不相邻
    ("9♠ 9♠ 9♥ 9♥ 9♣ 10♦ 10♦ 10♣", "永恒相随", 8),  # 5+3 相邻同点
    ("6♥ 7♥ 8♥ 9♥ 10♥ Q♦ Q♦ Q♥", "五谷丰登", 8),  # 5+3 八张同色
    ("6♥ 7♥ 8♥ 9♥ 10♥ Q♠ Q♣ Q♦", "五福临门", 4),  # 5+3 杂色
    ("9♠ 9♠ 9♥ 9♥ 10♣ 10♣ 10♦ 10♦", "二龙腾飞", 4),  # 4+4 相邻同点
    ("6♥ 7♥ 8♥ 9♥ Q♦ K♦ A♦ J♦", "四季发财", 4),  # 4+4 八张同色
    ("6♥ 7♥ 8♥ 9♥ Q♠ Q♣ Q♦ Q♥", "四季如春", 2),  # 4+4 杂色
    ("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ A♦", "平胡", 1),  # 3+3+2，2 为一对
]


@pytest.mark.parametrize(("spec", "pattern", "multiplier"), OFFICIAL_EXAMPLES)
def test_official_pattern_table(spec: str, pattern: str, multiplier: int) -> None:
    result = evaluate_hu(cards(spec), NO_GOLD)
    assert result is not None, f"{spec} 应能胡（{pattern}）"
    assert (result.pattern, result.pattern_multiplier) == (pattern, multiplier)


def test_ping_hu_requires_the_two_to_be_a_pair() -> None:
    # 3+3+2 的“2”若只是两张不同点数（例如同花相邻），官方规定不算平胡。
    assert evaluate_hu(cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ K♣"), NO_GOLD) is None


def test_gold_count_multiplies_linearly_like_the_settlement_screen() -> None:
    # 结算页实例：两行“被自摸（平胡）230分 3倍” —— 平胡 ×1 × 3 张金牌 = 每次 3 倍；
    # 没有癞子能胡 2 次，合计 ×2。
    result = evaluate_hu(cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ A♦"), ("6", "7", "8"))
    assert result is not None
    assert result.pattern == "平胡"
    assert result.gold_count == 3
    assert result.gold_multiplier == 3
    assert result.hard and result.hard_multiplier == 2
    assert result.total_multiplier // result.hard_multiplier == 3  # 每次支付的“3倍”
    assert result.theoretical_score == result.base_points * 6


def test_hand_with_wild_pays_only_once() -> None:
    hand = cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣") + [Card(wild=True)]
    result = evaluate_hu(hand, ("6", "7", "8"))
    assert result is not None
    assert not result.hard and result.hard_multiplier == 1


def test_two_payments_are_capped_per_payment_and_by_balance() -> None:
    from game_baque.ev import ExpectedValueEngine

    hard = evaluate_hu(cards("6♥ 7♥ 8♥ J♠ Q♠ K♠ A♣ A♦"), ("6", "7", "8"))
    assert hard is not None
    single = hard.theoretical_score / 2
    engine = ExpectedValueEngine()
    # 三家：单次上限都很高，但余额分别只够付 0.5 次、1.5 次、远超 2 次。
    engine.score_caps = (single * 10, single * 10, single * 10)
    engine.seat_balances = (single * 0.5, single * 1.5, single * 100)
    expected = (single * 0.5 + single * 1.5 + single * 2) / 3
    assert engine.hu_value(hard) == pytest.approx(expected)
