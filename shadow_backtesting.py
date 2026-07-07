# -*- coding: utf-8 -*-
"""
shadow_backtesting.py
사용자 거래기록을 '규칙 기반 Shadow 계좌'와 비교해
규칙 위반(조기 익절/지연 손절)을 진단한다.

입력 가정
---------
- trades: account_id, date, ticker, side, quantity, price
- prices: step + 라벨별 가격표
- dates : prices의 step과 1:1 대응하는 날짜 리스트

핵심 아이디어
-------------
1) 실제 계좌에서 종목별 진입/청산 시점을 복원
2) 같은 진입점에서 규칙 기반 Shadow 계좌를 재생
3) 실제 vs Shadow 청산 시점을 비교해 위반 유형 분류
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class StrategyProfile:
    """Shadow 계좌가 따를 규칙 파라미터."""

    stop_loss_pct: float = 0.10
    take_profit_pct: float | None = 0.25
    trailing_stop_pct: float | None = 0.08
    min_hold_steps: int = 1
    max_hold_steps: int | None = None


@dataclass
class ExitDecision:
    ticker: str
    step: int
    date: pd.Timestamp
    price: float
    reason: str


def _issue_label(issue_type: str | None) -> str:
    labels = {
        "early_profit_take": "조기 익절",
        "delayed_stop_loss": "지연 손절",
        "no_stop_loss_execution": "손절 미이행",
    }
    return labels.get(issue_type or "", "해당 없음")


def _event_explanation(issue_type: str, delay_steps: int, opportunity_cost_pct: float) -> tuple[str, str, str]:
    """이벤트 유형별 해설 문구를 생성한다."""
    severity = "low"
    if abs(opportunity_cost_pct) >= 5:
        severity = "high"
    elif abs(opportunity_cost_pct) >= 2:
        severity = "medium"

    if issue_type == "early_profit_take":
        psych = "이익 확정 욕구(후회 회피)로 추세를 끝까지 못 탔을 가능성이 큽니다."
        action = "분할 익절 또는 트레일링 손절 규칙으로 이익 구간 체류 시간을 늘려보세요."
    elif issue_type == "delayed_stop_loss":
        psych = "손실 회피 심리로 규칙 신호 이후에도 버틴 흔적입니다."
        action = "손절 신호 발생 시 즉시 실행하는 자동/체크리스트 규칙을 권장합니다."
    else:
        psych = "손절 규칙을 구조적으로 무시한 패턴으로 손실 비대칭이 커질 수 있습니다."
        action = "최대 손실 한도를 계좌 레벨에서 강제해 규칙 미이행을 차단하세요."

    narrative = (
        f"{_issue_label(issue_type)} 이벤트: 규칙 대비 {delay_steps:+d}주 차이가 났고 "
        f"기회비용은 {opportunity_cost_pct:+.1f}%p 입니다."
    )
    return narrative, psych, action


def _to_step_map(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, int]:
    out = {}
    for i, d in enumerate(dates):
        out[pd.Timestamp(d).normalize()] = i
    return out


def _build_positions(trades: pd.DataFrame, dates: list[pd.Timestamp]) -> dict[str, dict[str, Any]]:
    """종목별 진입/실제 청산 정보를 단순화해서 만든다.

    본 검사 데이터는 종목당 1회 매수/0~1회 매도를 가정한다.
    """
    step_map = _to_step_map(dates)
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()

    positions: dict[str, dict[str, Any]] = {}

    for ticker, g in t.groupby("ticker"):
        g = g.sort_values("date")
        buys = g[g["side"].str.upper() == "BUY"]
        sells = g[g["side"].str.upper() == "SELL"]
        if buys.empty:
            continue

        b = buys.iloc[0]
        entry_date = pd.Timestamp(b["date"]).normalize()
        entry_step = step_map.get(entry_date)
        if entry_step is None:
            continue

        row: dict[str, Any] = {
            "ticker": ticker,
            "entry_step": int(entry_step),
            "entry_date": entry_date,
            "entry_price": float(b["price"]),
            "quantity": float(b.get("quantity", 0) or 0),
            "actual_exit_step": None,
            "actual_exit_date": None,
            "actual_exit_price": None,
        }

        if not sells.empty:
            s = sells.iloc[0]
            exit_date = pd.Timestamp(s["date"]).normalize()
            exit_step = step_map.get(exit_date)
            if exit_step is not None:
                row["actual_exit_step"] = int(exit_step)
                row["actual_exit_date"] = exit_date
                row["actual_exit_price"] = float(s["price"])

        positions[ticker] = row

    return positions


def infer_strategy_profile(trades: pd.DataFrame, prices: pd.DataFrame, dates: list[pd.Timestamp]) -> dict[str, Any]:
    """사용자 실거래로부터 전략 성향(추정)을 프로필화한다."""
    positions = _build_positions(trades, dates)
    if not positions:
        return {
            "style": "insufficient-data",
            "inferred_take_profit_pct": None,
            "inferred_stop_loss_pct": None,
            "avg_holding_steps": None,
            "panic_sell_ratio": None,
            "n_positions": 0,
        }

    take_profits = []
    stop_losses = []
    holds = []
    panic_cnt = 0
    sell_cnt = 0

    for ticker, p in positions.items():
        if p["actual_exit_step"] is None:
            continue

        entry_step = p["entry_step"]
        exit_step = p["actual_exit_step"]
        entry_price = p["entry_price"]
        exit_price = p["actual_exit_price"]

        if exit_step <= entry_step:
            continue

        ret = exit_price / entry_price - 1.0
        holds.append(exit_step - entry_step)

        if ret >= 0:
            take_profits.append(ret)
        else:
            stop_losses.append(abs(ret))

        prev_px = float(prices.loc[max(0, exit_step - 1), ticker])
        if exit_price < prev_px:
            panic_cnt += 1
        sell_cnt += 1

    inferred_tp = float(pd.Series(take_profits).median()) if take_profits else None
    inferred_sl = float(pd.Series(stop_losses).median()) if stop_losses else None
    avg_hold = float(pd.Series(holds).mean()) if holds else None
    panic_ratio = (panic_cnt / sell_cnt) if sell_cnt else None

    style = "balanced"
    if inferred_tp is not None and inferred_tp <= 0.10:
        style = "quick-profit-taker"
    if inferred_sl is not None and inferred_sl >= 0.12:
        style = "loss-delayer"
    if (inferred_tp is not None and inferred_tp <= 0.10) and (inferred_sl is not None and inferred_sl >= 0.12):
        style = "disposition-prone"

    return {
        "style": style,
        "inferred_take_profit_pct": inferred_tp,
        "inferred_stop_loss_pct": inferred_sl,
        "avg_holding_steps": avg_hold,
        "panic_sell_ratio": panic_ratio,
        "n_positions": len(positions),
    }


def _simulate_shadow_exit(
    ticker: str,
    entry_step: int,
    entry_price: float,
    prices: pd.DataFrame,
    dates: list[pd.Timestamp],
    profile: StrategyProfile,
) -> ExitDecision:
    """규칙에 따라 Shadow 청산 시점을 찾는다."""
    n_steps = len(prices) - 1
    peak = entry_price

    for step in range(entry_step + 1, n_steps + 1):
        px = float(prices.loc[step, ticker])
        peak = max(peak, px)

        hold_len = step - entry_step
        if hold_len < profile.min_hold_steps:
            continue

        pnl = px / entry_price - 1.0
        drawdown_from_peak = px / peak - 1.0

        if pnl <= -profile.stop_loss_pct:
            return ExitDecision(ticker, step, pd.Timestamp(dates[step]).normalize(), px, "stop_loss")

        if profile.take_profit_pct is not None and pnl >= profile.take_profit_pct:
            return ExitDecision(ticker, step, pd.Timestamp(dates[step]).normalize(), px, "take_profit")

        if profile.trailing_stop_pct is not None and peak > entry_price and drawdown_from_peak <= -profile.trailing_stop_pct:
            return ExitDecision(ticker, step, pd.Timestamp(dates[step]).normalize(), px, "trailing_stop")

        if profile.max_hold_steps is not None and hold_len >= profile.max_hold_steps:
            return ExitDecision(ticker, step, pd.Timestamp(dates[step]).normalize(), px, "time_exit")

    final_px = float(prices.loc[n_steps, ticker])
    return ExitDecision(
        ticker=ticker,
        step=n_steps,
        date=pd.Timestamp(dates[n_steps]).normalize(),
        price=final_px,
        reason="hold_to_end",
    )


def shadow_backtest(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    dates: list[pd.Timestamp],
    profile: StrategyProfile,
    regime_map: dict[str, str] | None = None,
    profile_by_regime: dict[str, StrategyProfile] | None = None,
) -> dict[str, Any]:
    """실제 매매와 Shadow 규칙 매매를 비교해 위반 이벤트를 반환한다."""
    positions = _build_positions(trades, dates)
    rows = []
    events = []

    for ticker, p in positions.items():
        entry_step = p["entry_step"]
        entry_price = p["entry_price"]
        actual_exit_step = p["actual_exit_step"]

        chosen_profile = profile
        chosen_regime = "all"
        if regime_map and profile_by_regime:
            chosen_regime = regime_map.get(ticker, "all")
            chosen_profile = profile_by_regime.get(chosen_regime, profile)

        shadow = _simulate_shadow_exit(
            ticker=ticker,
            entry_step=entry_step,
            entry_price=entry_price,
            prices=prices,
            dates=dates,
            profile=chosen_profile,
        )

        if actual_exit_step is None:
            actual_step = len(prices) - 1
            actual_reason = "not_sold"
            actual_px = float(prices.loc[actual_step, ticker])
            actual_date = pd.Timestamp(dates[actual_step]).normalize()
        else:
            actual_step = int(actual_exit_step)
            actual_reason = "sold"
            actual_px = float(p["actual_exit_price"])
            actual_date = pd.Timestamp(p["actual_exit_date"]).normalize()

        shadow_ret = shadow.price / entry_price - 1.0
        actual_ret = actual_px / entry_price - 1.0
        delay_steps = actual_step - shadow.step

        issue = None
        note = ""

        if actual_step < shadow.step and actual_ret > 0:
            issue = "early_profit_take"
            note = "규칙 신호 이전에 이익 실현(조기 매도)"
        elif shadow.reason in {"stop_loss", "trailing_stop"} and actual_step > shadow.step:
            issue = "delayed_stop_loss"
            note = "손절 신호 이후 지연 청산"
        elif shadow.reason in {"stop_loss", "trailing_stop"} and actual_reason == "not_sold":
            issue = "no_stop_loss_execution"
            note = "손절 규칙 미이행(종료 시점까지 보유)"

        rows.append(
            {
                "ticker": ticker,
                "regime": chosen_regime,
                "entry_date": p["entry_date"],
                "actual_exit_date": actual_date,
                "shadow_exit_date": shadow.date,
                "actual_exit_reason": actual_reason,
                "shadow_exit_reason": shadow.reason,
                "actual_return_pct": actual_ret * 100,
                "shadow_return_pct": shadow_ret * 100,
                "alpha_vs_shadow_pct": (actual_ret - shadow_ret) * 100,
                "delay_steps_vs_shadow": delay_steps,
                "issue_type": issue,
                "issue_label": _issue_label(issue),
                "issue_note": note,
            }
        )

        if issue is not None:
            events.append(
                {
                    "ticker": ticker,
                    "regime": chosen_regime,
                    "issue_type": issue,
                    "issue_label": _issue_label(issue),
                    "actual_exit_date": actual_date,
                    "shadow_exit_date": shadow.date,
                    "delay_steps": delay_steps,
                    "actual_return_pct": actual_ret * 100,
                    "shadow_return_pct": shadow_ret * 100,
                    "opportunity_cost_pct": (shadow_ret - actual_ret) * 100,
                    "note": note,
                }
            )

    compare_df = pd.DataFrame(rows)
    events_df = pd.DataFrame(events)

    if not events_df.empty:
        narratives = []
        psych_hints = []
        action_hints = []
        severities = []
        for _, row in events_df.iterrows():
            nrt, psy, act = _event_explanation(
                str(row["issue_type"]),
                int(row["delay_steps"]),
                float(row["opportunity_cost_pct"]),
            )
            narratives.append(nrt)
            psych_hints.append(psy)
            action_hints.append(act)
            if abs(float(row["opportunity_cost_pct"])) >= 5:
                severities.append("high")
            elif abs(float(row["opportunity_cost_pct"])) >= 2:
                severities.append("medium")
            else:
                severities.append("low")

        events_df["severity"] = severities
        events_df["narrative"] = narratives
        events_df["psychology_hint"] = psych_hints
        events_df["action_hint"] = action_hints

    violations = len(events_df)
    n = len(compare_df) if len(compare_df) else 1
    adherence = max(0.0, 1.0 - violations / n)

    early_cnt = int((events_df.get("issue_type") == "early_profit_take").sum()) if not events_df.empty else 0
    delayed_cnt = int((events_df.get("issue_type") == "delayed_stop_loss").sum()) if not events_df.empty else 0
    missed_stop_cnt = int((events_df.get("issue_type") == "no_stop_loss_execution").sum()) if not events_df.empty else 0

    total_cost = float(events_df["opportunity_cost_pct"].sum()) if not events_df.empty else 0.0
    avg_delay = float(events_df["delay_steps"].mean()) if not events_df.empty else 0.0

    summary = {
        "n_positions": int(len(compare_df)),
        "n_violations": violations,
        "rule_adherence_pct": adherence * 100,
        "early_profit_take_count": early_cnt,
        "delayed_stop_loss_count": delayed_cnt,
        "no_stop_loss_execution_count": missed_stop_cnt,
        "avg_delay_steps": avg_delay,
        "total_opportunity_cost_pct": total_cost,
    }

    return {
        "summary": summary,
        "comparison": compare_df,
        "events": events_df,
    }
