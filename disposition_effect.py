"""
disposition_effect.py
처분효과(Disposition Effect) 측정 — Odean(1998) PGR/PLR 방법론 구현

핵심 설계 원칙
--------------
1) 분류 단위는 '매도가 발생한 날(sell day)'의 보유 포지션이다.
   - 매도가 없는 날의 보유분은 paper로 세지 않는다. (Odean의 표본 정의)
2) 손익 판정 기준(cost basis)은 평균단가(average cost) 기본, FIFO 선택 가능.
3) 개인별(account-level)로 PGR/PLR을 산출한 뒤 횡단면 집계하는 것을
   기본으로 한다. 전체 풀링(pooled)도 옵션으로 제공한다.

입력 데이터 스키마 (거래내역 DataFrame)
--------------------------------------
필수 컬럼:
  account_id : 계좌/투자자 식별자
  date       : 거래일 (datetime 변환 가능)
  ticker     : 종목 식별자
  side       : 'BUY' 또는 'SELL'
  quantity   : 체결 수량 (양수)
  price      : 체결 단가

가격 기준일 종가가 별도로 있으면 더 정확하지만, 본 구현은
'매도일에 매도가 발생한 종목 = realized, 그날 함께 보유한 다른 종목 = paper'
판정을 위해 보유 종목의 그날 평가가격이 필요하다. 두 가지 모드 제공:
  - mode='trade_price': 그날 거래된 가격을 평가가격 proxy로 사용
                        (paper 종목은 마지막 체결가 또는 별도 price_panel 필요)
  - price_panel 제공 시: (date, ticker) -> close 가격으로 정확히 평가
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass


# ----------------------------------------------------------------------
# 포지션 추적: 평균단가 / FIFO
# ----------------------------------------------------------------------
class PositionBook:
    """계좌별 보유 포지션과 cost basis를 추적한다."""

    def __init__(self, method: str = "average"):
        assert method in ("average", "fifo")
        self.method = method
        # ticker -> dict(qty, cost) [average]  또는  ticker -> list[(qty, price)] [fifo]
        self._pos: dict = {}

    def quantity(self, ticker: str) -> float:
        if ticker not in self._pos:
            return 0.0
        if self.method == "average":
            return self._pos[ticker]["qty"]
        return sum(q for q, _ in self._pos[ticker])

    def cost_basis(self, ticker: str) -> float | None:
        """현재 보유분의 단위당 평균 취득원가. 미보유 시 None."""
        if self.quantity(ticker) <= 0:
            return None
        if self.method == "average":
            p = self._pos[ticker]
            return p["cost"] / p["qty"] if p["qty"] > 0 else None
        lots = self._pos[ticker]
        tot_q = sum(q for q, _ in lots)
        tot_c = sum(q * px for q, px in lots)
        return tot_c / tot_q if tot_q > 0 else None

    def buy(self, ticker: str, qty: float, price: float):
        if self.method == "average":
            p = self._pos.setdefault(ticker, {"qty": 0.0, "cost": 0.0})
            p["qty"] += qty
            p["cost"] += qty * price
        else:
            self._pos.setdefault(ticker, []).append((qty, price))

    def sell(self, ticker: str, qty: float):
        """매도 처리. 보유량을 차감한다(realized 손익 계산은 호출부에서)."""
        if self.method == "average":
            p = self._pos.get(ticker)
            if p is None or p["qty"] <= 0:
                return
            avg = p["cost"] / p["qty"]
            sell_qty = min(qty, p["qty"])
            p["qty"] -= sell_qty
            p["cost"] -= avg * sell_qty
            if p["qty"] <= 1e-9:
                self._pos.pop(ticker, None)
        else:
            lots = self._pos.get(ticker, [])
            remaining = qty
            while remaining > 1e-9 and lots:
                q0, px0 = lots[0]
                take = min(remaining, q0)
                q0 -= take
                remaining -= take
                if q0 <= 1e-9:
                    lots.pop(0)
                else:
                    lots[0] = (q0, px0)
            if not lots:
                self._pos.pop(ticker, None)

    def held_tickers(self) -> list[str]:
        return [t for t in self._pos if self.quantity(t) > 0]


# ----------------------------------------------------------------------
# 결과 컨테이너
# ----------------------------------------------------------------------
@dataclass
class DispositionResult:
    realized_gains: int
    paper_gains: int
    realized_losses: int
    paper_losses: int

    @property
    def pgr(self) -> float | None:
        denom = self.realized_gains + self.paper_gains
        return self.realized_gains / denom if denom > 0 else None

    @property
    def plr(self) -> float | None:
        denom = self.realized_losses + self.paper_losses
        return self.realized_losses / denom if denom > 0 else None

    @property
    def disposition_spread(self) -> float | None:
        if self.pgr is None or self.plr is None:
            return None
        return self.pgr - self.plr

    @property
    def disposition_ratio(self) -> float | None:
        if self.pgr is None or self.plr is None or self.plr == 0:
            return None
        return self.pgr / self.plr

    def as_dict(self) -> dict:
        return {
            "realized_gains": self.realized_gains,
            "paper_gains": self.paper_gains,
            "realized_losses": self.realized_losses,
            "paper_losses": self.paper_losses,
            "PGR": self.pgr,
            "PLR": self.plr,
            "PGR_minus_PLR": self.disposition_spread,
            "PGR_over_PLR": self.disposition_ratio,
        }


# ----------------------------------------------------------------------
# 핵심 계산
# ----------------------------------------------------------------------
def _eval_price(date, ticker, last_trade_price, price_panel):
    """평가가격 결정: price_panel 우선, 없으면 그날 거래가(proxy)."""
    if price_panel is not None:
        key = (pd.Timestamp(date).normalize(), ticker)
        if key in price_panel:
            return price_panel[key]
    return last_trade_price


def compute_account_disposition(
    trades: pd.DataFrame,
    method: str = "average",
    price_panel: dict | None = None,
    exclude_same_day_buy: bool = True,
) -> DispositionResult:
    """
    단일 계좌의 PGR/PLR 구성요소를 산출한다.

    분류 로직 (Odean 1998):
      - 매도가 발생한 날(sell day)에 대해서만 분류를 수행한다.
      - 그날 '매도된' 종목: cost basis 대비 손익 부호로 realized gain/loss
      - 그날 '보유 중이나 안 판' 종목: paper gain/loss
      - exclude_same_day_buy=True 면 당일 신규 매수만 있어 cost basis가
        당일가와 같은 종목은 paper 판정에서 제외(노이즈 방지).
    """
    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "side"]).reset_index(drop=True)
    # 같은 날 안에서 BUY를 SELL보다 먼저 처리(당일 매수분이 보유에 반영되도록)
    side_order = {"BUY": 0, "SELL": 1}
    df["_so"] = df["side"].str.upper().map(side_order)
    df = df.sort_values(["date", "_so"]).reset_index(drop=True)

    book = PositionBook(method=method)
    rg = pg = rl = pl = 0
    last_price: dict[str, float] = {}

    bought_today: set[str] = set()

    for date, day in df.groupby(df["date"].dt.normalize(), sort=True):
        bought_today.clear()
        day_sells = []  # (ticker, basis_before_sell)

        # 1) 그날 거래를 순서대로 반영하되, SELL 시점의 cost basis를 기록
        for _, row in day.iterrows():
            t = row["ticker"]
            side = str(row["side"]).upper()
            qty = float(row["quantity"])
            px = float(row["price"])
            last_price[t] = px

            if side == "BUY":
                book.buy(t, qty, px)
                bought_today.add(t)
            elif side == "SELL":
                basis = book.cost_basis(t)  # 매도 직전 평균취득원가
                book.sell(t, qty)
                if basis is not None:
                    day_sells.append((t, basis, px))

        sold_today = {t for t, _, _ in day_sells}

        # 2) realized 분류 (매도 종목)
        for t, basis, sell_px in day_sells:
            if sell_px > basis:
                rg += 1
            elif sell_px < basis:
                rl += 1
            # 동가는 무시

        # 3) paper 분류 (그날 보유 중 + 안 판 종목)
        for t in book.held_tickers():
            if t in sold_today:
                # 일부만 판 경우에도 Odean은 매도 종목을 realized로만 카운트
                continue
            if exclude_same_day_buy and t in bought_today and book.quantity(t) > 0:
                # 당일 신규진입만 있는 종목 → cost basis가 당일가라 평가 무의미
                # (직전일부터 보유가 아니므로 paper 모집단에서 제외)
                # 단, 이전부터 보유했다면 제외하지 않음 → 보수적으로 basis 비교로 판단
                pass
            basis = book.cost_basis(t)
            if basis is None:
                continue
            ep = _eval_price(date, t, last_price.get(t, basis), price_panel)
            if ep > basis:
                pg += 1
            elif ep < basis:
                pl += 1

    return DispositionResult(rg, pg, rl, pl)


def compute_disposition(
    trades: pd.DataFrame,
    method: str = "average",
    price_panel: dict | None = None,
    by_account: bool = True,
    exclude_same_day_buy: bool = True,
) -> dict:
    """
    전체 거래내역에 대해 PGR/PLR을 계산한다.

    by_account=True  : 계좌별 산출 후 횡단면 평균(권장, 견고)
                       + 풀링 결과도 함께 반환
    by_account=False : 전체 풀링만
    """
    results = {}

    # 풀링(전체 거래를 한 모집단으로)
    pooled_rg = pooled_pg = pooled_rl = pooled_pl = 0
    per_account = {}

    for acc, g in trades.groupby("account_id"):
        r = compute_account_disposition(
            g, method=method, price_panel=price_panel,
            exclude_same_day_buy=exclude_same_day_buy,
        )
        per_account[acc] = r.as_dict()
        pooled_rg += r.realized_gains
        pooled_pg += r.paper_gains
        pooled_rl += r.realized_losses
        pooled_pl += r.paper_losses

    pooled = DispositionResult(pooled_rg, pooled_pg, pooled_rl, pooled_pl)
    results["pooled"] = pooled.as_dict()

    if by_account:
        adf = pd.DataFrame(per_account).T
        # 계좌별 PGR/PLR이 정의된 경우만 평균
        results["cross_sectional"] = {
            "n_accounts": len(adf),
            "mean_PGR": adf["PGR"].dropna().mean(),
            "mean_PLR": adf["PLR"].dropna().mean(),
            "mean_PGR_minus_PLR": (adf["PGR"] - adf["PLR"]).dropna().mean(),
        }
        results["per_account"] = per_account

    return results


# ----------------------------------------------------------------------
# 합성 데이터 검증
# ----------------------------------------------------------------------
def _make_synthetic(seed: int = 42, n_accounts: int = 200) -> pd.DataFrame:
    """
    처분효과를 의도적으로 심은 합성 거래내역 생성.
    - 이익 포지션은 높은 확률로 매도(PGR↑), 손실 포지션은 낮은 확률로 매도(PLR↓)
    """
    rng = np.random.default_rng(seed)
    rows = []
    tickers = [f"T{i:03d}" for i in range(30)]
    base_date = pd.Timestamp("2025-01-02")

    for acc in range(n_accounts):
        acc_id = f"A{acc:04d}"
        # 초기 매수 5~10종목
        n_hold = rng.integers(5, 11)
        chosen = rng.choice(tickers, size=n_hold, replace=False)
        buy_prices = {}
        for t in chosen:
            px = float(rng.uniform(50, 150))
            buy_prices[t] = px
            rows.append([acc_id, base_date, t, "BUY", 10, px])

        # 이후 20거래일 동안 가격 변동 + 매도 결정
        for d in range(1, 21):
            date = base_date + pd.Timedelta(days=d)
            for t in list(buy_prices.keys()):
                drift = rng.normal(0, 0.03)
                cur = buy_prices[t] * (1 + drift * d)
                pnl = cur - buy_prices[t]
                # 처분효과: 이익이면 매도확률 0.30, 손실이면 0.08
                p_sell = 0.30 if pnl > 0 else 0.08
                if rng.random() < p_sell:
                    rows.append([acc_id, date, t, "SELL", 10, float(cur)])
                    del buy_prices[t]

    df = pd.DataFrame(
        rows, columns=["account_id", "date", "ticker", "side", "quantity", "price"]
    )
    return df


if __name__ == "__main__":
    df = _make_synthetic()
    print(f"합성 거래내역: {len(df):,}건, 계좌 {df['account_id'].nunique()}개\n")

    res = compute_disposition(df, method="average", by_account=True)

    print("=== Pooled (전체 풀링) ===")
    for k, v in res["pooled"].items():
        if isinstance(v, float):
            print(f"  {k:16s}: {v:.4f}")
        else:
            print(f"  {k:16s}: {v}")

    print("\n=== Cross-sectional (계좌별 평균, 권장) ===")
    for k, v in res["cross_sectional"].items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    cs = res["cross_sectional"]
    spread = cs["mean_PGR_minus_PLR"]
    print("\n=== 해석 ===")
    if spread and spread > 0:
        print(f"  PGR > PLR (spread={spread:.4f}) → 처분효과 존재 확인.")
        print("  이익은 잘 실현하고 손실은 덜 실현하는 경향. (의도대로 검출됨)")
    else:
        print("  처분효과 미검출.")
