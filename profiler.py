# -*- coding: utf-8 -*-
"""
profiler.py — 거래내역 → 5대 행동편향 점수화(투자 심리 프로파일).

입력: 한 검사자(account 'YOU')의 거래내역 + 가격표(prices_lab) + 메타(meta_lab).
출력: 5대 편향 severity(0=없음 … 100=심함), 종합 유형, 손익 비교, 맞춤 솔루션.

5대 편향
  1) 처분효과    : PGR−PLR (오른 건 팔고 내린 건 보유)
  2) 추세 놓침   : 추세주(상승국면)를 이익 중 조기 청산한 비율
  3) 패닉 매도   : 하락한 날(전 스텝 대비 ↓)에 매도한 비율
  4) 과잉 매매   : 평균 보유기간이 짧을수록 ↑ (잦은 손바뀜)
  5) 본전 집착   : 진입가 ±3% 부근에서 매도한 비율(get-even)

측정 엔진(PGR/PLR)은 상위 폴더 disposition_effect.py 재사용.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
# disposition_effect.py 위치: 같은 폴더(배포 저장소) 또는 부모 행동과학/(로컬 개발)
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from disposition_effect import compute_disposition  # noqa: E402

BREAKEVEN_BAND = 0.03    # ±3% 본전 부근


def _clip100(x):
    return float(max(0.0, min(100.0, x)))


def build_price_panel(prices: pd.DataFrame, dates) -> dict:
    panel = {}
    labels = [c for c in prices.columns if c != "step"]
    for s in range(len(prices)):
        for lab in labels:
            panel[(pd.Timestamp(dates[s]).normalize(), lab)] = \
                float(prices.loc[s, lab])
    return panel


def profile(trades: pd.DataFrame, prices: pd.DataFrame, meta: pd.DataFrame,
            dates) -> dict:
    labels = [c for c in prices.columns if c != "step"]
    regime = dict(zip(meta["label"], meta["regime"]))
    n_steps = len(prices) - 1
    panel = build_price_panel(prices, dates)

    sells = trades[trades["side"] == "SELL"].copy()
    n_sells = len(sells)

    # ── 1) 처분효과 (전체 + 국면별) ──
    def de_of(tickers):
        sub = trades[trades["ticker"].isin(tickers)]
        if sub.empty:
            return None
        r = compute_disposition(sub, price_panel=panel, by_account=True)
        return r["pooled"]["PGR_minus_PLR"]

    de_all = de_of(labels)
    de_box = de_of([l for l in labels if regime[l] == "box"])
    de_trend = de_of([l for l in labels if regime[l] == "trend"])
    disposition = _clip100((de_all or 0) * 100) if de_all is not None else 0.0

    # 보조 계산: 각 매도의 (스텝, 진입가, 매도가, 직전가)
    entry = {lab: float(prices.loc[0, lab]) for lab in labels}
    sold_labels = set()
    panic_cnt = breakeven_cnt = 0
    hold_steps = {}
    for _, r in sells.iterrows():
        lab = r["ticker"]
        step = int(pd.Index(dates).get_loc(pd.Timestamp(r["date"])))
        sold_labels.add(lab)
        hold_steps[lab] = step
        prev_px = float(prices.loc[step - 1, lab]) if step > 0 else entry[lab]
        sell_px = float(r["price"])
        if sell_px < prev_px:
            panic_cnt += 1
        if abs(sell_px / entry[lab] - 1) <= BREAKEVEN_BAND:
            breakeven_cnt += 1

    # ── 2) 추세 놓침: 추세주를 이익 중 조기 청산 ──
    trend_labels = [l for l in labels if regime[l] == "trend"]
    trend_sold_in_gain = 0
    for lab in trend_labels:
        if lab in sold_labels:
            step = hold_steps[lab]
            if step < n_steps and float(prices.loc[step, lab]) > entry[lab]:
                trend_sold_in_gain += 1
    counter_trend = _clip100(100 * trend_sold_in_gain / max(1, len(trend_labels)))

    # ── 3) 패닉 매도 ──
    panic = _clip100(100 * panic_cnt / n_sells) if n_sells else 0.0

    # ── 4) 과잉 매매: 평균 보유기간 짧을수록 ↑ ──
    holds = [hold_steps.get(lab, n_steps) for lab in labels]   # 안 판 건 끝까지
    avg_hold = sum(holds) / len(holds)
    overtrade = _clip100(100 * (1 - (avg_hold - 1) / (n_steps - 1)))

    # ── 5) 본전 집착 ──
    anchoring = _clip100(100 * breakeven_cnt / n_sells) if n_sells else 0.0

    traits = {
        "처분효과": disposition,
        "추세 놓침": counter_trend,
        "패닉 매도": panic,
        "과잉 매매": overtrade,
        "본전 집착": anchoring,
    }

    # ── 종합 유형 ──
    TYPE = {
        "처분효과": "성급한 익절·손실 외면형",
        "추세 놓침": "추세 놓침형(승자 조기청산)",
        "패닉 매도": "패닉 매도형",
        "과잉 매매": "과잉 매매형",
        "본전 집착": "본전 집착형",
    }
    if n_sells == 0:
        ptype = "완전 보유형(관망)"
        dom = None
    elif max(traits.values()) < 25:
        ptype = "균형 잡힌 규율형"
        dom = max(traits, key=traits.get)
    else:
        dom = max(traits, key=traits.get)
        ptype = TYPE[dom]

    return {
        "traits": traits,
        "type": ptype,
        "dominant": dom,
        "n_sells": n_sells,
        "de_all": de_all, "de_box": de_box, "de_trend": de_trend,
        "avg_hold": avg_hold,
    }


# ── 손익 비교(편향 vs 보유 vs 규율) ──
def equity_of(trades, prices, dates, qty):
    labels = [c for c in prices.columns if c != "step"]
    n_steps = len(prices) - 1
    held = set(labels)
    cash = 0.0
    for _, r in trades[trades["side"] == "SELL"].iterrows():
        lab = r["ticker"]
        if lab in held:
            cash += float(r["price"]) * qty[lab]
            held.discard(lab)
    final = cash + sum(float(prices.loc[n_steps, lab]) * qty[lab] for lab in held)
    invested = sum(float(prices.loc[0, lab]) * qty[lab] for lab in labels)
    return final, invested


SOLUTIONS = {
    "처분효과": ("진입과 동시에 '목표가·손절가'를 함께 적어두세요. 손실 종목을 "
              "들고 버티는 대신, 정해둔 손절선에서 기계적으로 끊는 규칙이 "
              "처분효과를 가장 직접적으로 줄입니다."),
    "추세 놓침": ("오르는 종목은 '얼마 벌면 판다'(고정 목표가) 대신 추적 손절"
              "(trailing stop)을 쓰세요. 추세가 살아 있는 한 들고, 고점 대비 "
              "일정 % 빠질 때만 파는 방식이 승자를 끝까지 태웁니다."),
    "패닉 매도": ("하락한 날의 즉흥 매도를 막으려면 '쿨다운 규칙'을 두세요. 빨간 "
              "날엔 당일 매도 금지, 다음 날 미리 정한 기준으로만 판단합니다."),
    "과잉 매매": ("매매 횟수에 상한을 두세요(예: 주 1회 점검). 잦은 손바뀜은 "
              "비용과 실수를 늘립니다. 체크리스트를 통과한 경우만 매매합니다."),
    "본전 집착": ("'본전 생각'을 버리세요. 매도 판단은 매수가가 아니라 '지금 이 "
              "종목의 앞으로의 전망'으로 합니다. 진입가는 이미 매몰비용입니다."),
}


def solution_for(prof: dict) -> list[str]:
    traits = prof["traits"]
    ranked = sorted(traits.items(), key=lambda kv: -kv[1])
    out = []
    for name, score in ranked:
        if score >= 25:
            out.append(f"[{name} {score:.0f}점] {SOLUTIONS[name]}")
    if not out:
        out.append("뚜렷한 편향이 약합니다. 지금의 규율(손절·보유 원칙)을 "
                    "유지하세요.")
    return out
