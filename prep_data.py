# -*- coding: utf-8 -*-
"""
prep_data.py — 투자심리검사용 10종목 가격 캐시 생성.

박스권(2014–15) 5종목 + 추세상승(2020–21) 5종목을 실제로 받아,
각 종목을 진입가=100,000원으로 정규화(블라인드·동일금액)하고,
10스텝으로 표본화해 익명(A~J)으로 섞어 저장한다.

출력:
  prices_lab.csv : step, A..J (정규화 종가)  — 검사용(블라인드)
  meta_lab.csv   : code, name, regime, period, label(A..J)  — 결과 공개용

사용: python prep_data.py
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_STEPS = 10                      # 0..9 (진입 포함 10점)
ENTRY = 100_000                   # 정규화 진입가(원)
SEED = 7

BOX = {                           # 박스권 2014–15
    "005930.KS": "삼성전자", "017670.KS": "SK텔레콤", "105560.KS": "KB금융",
    "055550.KS": "신한지주", "030200.KS": "KT",
}
TREND = {                         # 추세상승 2020–21
    "066570.KS": "LG전자", "005490.KS": "POSCO", "005380.KS": "현대차",
    "035420.KS": "NAVER", "000270.KS": "기아",
}
BOX_WIN = ("2014-01-01", "2016-01-01")
TREND_WIN = ("2020-03-01", "2021-07-01")


def _fetch_norm(codes, win):
    import yfinance as yf
    d = yf.download(list(codes), start=win[0], end=win[1], interval="1wk",
                    group_by="ticker", auto_adjust=True, progress=False)
    out = {}
    for code in codes:
        s = d[code]["Close"].dropna()
        idx = [int(round(x)) for x in np.linspace(0, len(s) - 1, N_STEPS)]
        samp = s.iloc[idx].values
        out[code] = (samp / samp[0] * ENTRY).round(-2)   # 진입=100,000 정규화
    return out


def main():
    print("· 박스권/추세 실데이터 수신 중(야후)...")
    box = _fetch_norm(BOX, BOX_WIN)
    trend = _fetch_norm(TREND, TREND_WIN)

    # 메타 + 익명 라벨(A~J) 섞기
    items = []
    for code, nm in BOX.items():
        items.append((code, nm, "box", "2014–15", box[code]))
    for code, nm in TREND.items():
        items.append((code, nm, "trend", "2020–21", trend[code]))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(items))
    labels = [chr(ord("A") + i) for i in range(len(items))]

    price_cols = {}
    meta_rows = []
    for label, oi in zip(labels, order):
        code, nm, regime, period, series = items[oi]
        price_cols[label] = series
        meta_rows.append({"label": label, "code": code, "name": nm,
                          "regime": regime, "period": period})

    prices = pd.DataFrame(price_cols)
    prices.insert(0, "step", range(N_STEPS))
    prices.to_csv(HERE / "prices_lab.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(meta_rows).to_csv(HERE / "meta_lab.csv", index=False,
                                   encoding="utf-8-sig")

    # 검증 출력
    print("저장: prices_lab.csv, meta_lab.csv")
    rep = []
    for r in meta_rows:
        s = price_cols[r["label"]]
        rep.append((r["label"], r["name"], r["regime"],
                    round(s[-1] / s[0] * 100 - 100, 1),
                    round((max(s) / min(s) - 1) * 100, 1)))
    df = pd.DataFrame(rep, columns=["라벨", "종목", "국면", "최종%", "고저폭%"])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
