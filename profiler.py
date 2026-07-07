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
        "성급한 익절": disposition,
        "상승주 조기매도": counter_trend,
        "공포 매도": panic,
        "잦은 매매": overtrade,
        "본전 집착": anchoring,
    }

    # ── 종합 유형 (쉬운 말) ──
    TYPE = {
        "성급한 익절": "오른 건 서둘러 팔고, 내린 건 버티는 형",
        "상승주 조기매도": "오를 종목을 너무 일찍 파는 형",
        "공포 매도": "겁나면 던지는 형",
        "잦은 매매": "짧게 자주 사고파는 형",
        "본전 집착": "본전 회복만 기다리는 형",
    }
    if n_sells == 0:
        ptype = "끝까지 보유형 (한 번도 안 팜)"
        dom = None
    elif max(traits.values()) < 25:
        ptype = "균형 잡힌 규율형 (뚜렷한 나쁜 습관 없음)"
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
        # 개인화 솔루션 근거로 쓰는 상세 카운트
        "detail": {
            "n_sells": n_sells,
            "panic_cnt": panic_cnt,
            "breakeven_cnt": breakeven_cnt,
            "trend_sold_in_gain": trend_sold_in_gain,
            "n_trend": len(trend_labels),
            "avg_hold": avg_hold,
            "n_steps": n_steps,
            "de_all": de_all, "de_box": de_box, "de_trend": de_trend,
        },
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


# 각 습관의 한 줄 쉬운 설명 (결과 화면·리포트에서 점수 옆에 함께 표시).
TRAIT_DESC = {
    "성급한 익절": "오른 종목은 서둘러 팔고, 내린 종목은 계속 들고 버티는 습관",
    "상승주 조기매도": "더 오를 종목을 이익 조금에 미리 파는 습관",
    "공포 매도": "하락한 날 겁이 나 즉흥적으로 파는 습관",
    "잦은 매매": "짧게 자주 사고파는 습관",
    "본전 집착": "산 가격 회복만 기다려 판단이 흐려지는 습관",
}


# 습관별 개인 맞춤 처방: (진단 근거 = 당신의 실제 수치) + (구체적 3단계 규칙).
# d = prof['detail'] 를 받아 사용자 수치를 문장에 주입한다.
def _sol_disposition(d) -> str:
    box = ("특히 지지부진하던 장에서 더 심했습니다. "
           if d["de_box"] is not None and d["de_box"] > 0 else "")
    return (f"당신은 오른 종목을 손실 종목보다 먼저 파는 성향이 컸습니다. {box}"
            f"처방 ①살 때 바로 '팔 목표가격=산 값의 1.15배, 손절 가격=산 값의 "
            f"0.90배'를 종목마다 적어두기 ②손절 가격에 닿으면 감정 없이 그날 팔기 "
            f"③이익 종목은 목표가격 전까지 팔지 않기. 다음 검사에선 이 습관 점수를 "
            f"0에 가깝게 낮추는 게 목표입니다.")


def _sol_counter_trend(d) -> str:
    return (f"쭉 오르던 종목 {d['n_trend']}개 중 {d['trend_sold_in_gain']}개를 "
            f"조금 이익 났을 때 미리 팔아, 그 뒤 상승분을 놓쳤습니다. "
            f"처방 ①오르는 종목은 '얼마 벌면 판다' 대신 '가장 높았던 값에서 10% "
            f"떨어지면 판다'로 관리 ②이익 났다고 파는 게 아니라 오름세가 꺾일 때만 "
            f"팔기 ③살 때 '이건 오르는 종목'이라고 표시해 두고 따로 관리하기.")


def _sol_panic(d) -> str:
    return (f"전체 매도 {d['n_sells']}번 중 {d['panic_cnt']}번이 '전날 떨어진 종목'을 "
            f"겁나서 즉흥적으로 판 것이었습니다. "
            f"처방 ①떨어진 날엔 그날 바로 팔지 않기—하루 기다리기 ②팔고 싶으면 이유를 "
            f"한 줄 적고 다음 날 아침에만 실행 ③떨어진 날의 매도는 미리 정한 손절 "
            f"가격에 닿았을 때만 허용. 이 규칙으로 {d['panic_cnt']}번 중 상당수를 "
            f"막을 수 있습니다.")


def _sol_overtrade(d) -> str:
    return (f"한 종목을 평균 {d['avg_hold']:.1f}주밖에 안 들고 있어 사고파는 게 너무 "
            f"잦았습니다. "
            f"처방 ①매매는 '일주일에 한 번 정하는 날'에만 하기 ②사고팔기 전 3가지"
            f"(오름세가 살아있나? 손절 가격에 닿았나? 목표가격에 닿았나?)를 확인하고 "
            f"맞을 때만 ③일주일 매매 횟수 상한을 정해 넘으면 그냥 지켜보기. 잦은 "
            f"매매는 수익보다 수수료와 실수를 키웁니다.")


def _sol_anchoring(d) -> str:
    return (f"매도 {d['n_sells']}번 중 {d['breakeven_cnt']}번이 '산 값 근처(±3%)'에서 "
            f"나왔습니다. 본전 생각에 매여 있었다는 뜻입니다. "
            f"처방 ①팔지 말지는 산 값이 아니라 '지금부터 오를까'만 보고 정하기 "
            f"②차트에서 내가 산 가격 선을 지우고 보기 ③본전 회복을 기다리며 손실을 "
            f"키우지 않기—산 값은 이미 돌아오지 않는 돈입니다.")


SOLUTIONS = {
    "성급한 익절": _sol_disposition,
    "상승주 조기매도": _sol_counter_trend,
    "공포 매도": _sol_panic,
    "잦은 매매": _sol_overtrade,
    "본전 집착": _sol_anchoring,
}


def solution_for(prof: dict) -> list[str]:
    traits = prof["traits"]
    d = prof.get("detail", {})
    ranked = sorted(traits.items(), key=lambda kv: -kv[1])
    out = []
    for name, score in ranked:
        if score >= 25:
            out.append(f"[{name} {score:.0f}점] {SOLUTIONS[name](d)}")
    if not out:
        out.append(
            "뚜렷한 나쁜 습관이 약합니다. 지금의 규율(손절·보유 원칙)을 그대로 "
            "유지하세요. 다음 검사에서도 이 습관을 지키는 게 목표입니다.")
    return out
