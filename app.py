# -*- coding: utf-8 -*-
"""
app.py — 투자 심리검사 대시보드 (Streamlit)

10종목(박스권 5 + 추세장 5, 블라인드)을 10주간 직접 매매 → 거래내역으로
5대 행동편향을 진단 → 국면별 처분효과·성과 비교 → 맞춤 솔루션 + docx 리포트.

실행:
  ..\..\pykrx_venv\Scripts\python.exe -m streamlit run app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profiler as P          # noqa: E402
import shadow_backtesting as SB  # noqa: E402
from report import make_docx  # noqa: E402

# 한글 폰트 — 로컬(Windows)은 Malgun, 클라우드(Linux)는 NanumGothic 자동 선택.
# 클라우드에선 packages.txt의 fonts-nanum가 설치돼 아래 후보에 잡힌다.
import matplotlib.font_manager as _fm  # noqa: E402
for _fp in _fm.findSystemFonts():
    if any(k in _fp.lower() for k in ("nanum", "malgun")):
        try:
            _fm.fontManager.addfont(_fp)
        except Exception:
            pass
_avail = {f.name for f in _fm.fontManager.ttflist}
for f in ("Malgun Gothic", "맑은 고딕", "NanumGothic", "NanumBarunGothic"):
    if f in _avail:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

BUDGET_PER = 2_000_000
ACCOUNT = "YOU"

st.set_page_config(page_title="투자 심리검사", page_icon="🧠", layout="centered")


@st.cache_data
def load_data():
    prices = pd.read_csv(HERE / "prices_lab.csv", encoding="utf-8-sig")
    meta = pd.read_csv(HERE / "meta_lab.csv", encoding="utf-8-sig")
    return prices, meta


PRICES, META = load_data()
LABELS = [c for c in PRICES.columns if c != "step"]
N = len(PRICES) - 1
DATES = list(pd.bdate_range("2020-01-06", periods=len(PRICES)))
ENTRY = {l: float(PRICES.loc[0, l]) for l in LABELS}
QTY = {l: max(1, round(BUDGET_PER / ENTRY[l])) for l in LABELS}
REGIME = dict(zip(META["label"], META["regime"]))


def init_state():
    ss = st.session_state
    ss.started = False
    ss.finished = False
    ss.step = 1
    ss.holdings = set(LABELS)
    ss.trades = [{"account_id": ACCOUNT, "date": DATES[0], "ticker": l,
                  "side": "BUY", "quantity": QTY[l], "price": ENTRY[l]}
                 for l in LABELS]


if "started" not in st.session_state:
    init_state()


def record_sells(picked, step):
    for lab in picked:
        if lab in st.session_state.holdings:
            px = float(PRICES.loc[step, lab])
            st.session_state.trades.append({
                "account_id": ACCOUNT, "date": DATES[step], "ticker": lab,
                "side": "SELL", "quantity": QTY[lab], "price": px})
            st.session_state.holdings.discard(lab)


def radar_png(traits: dict) -> bytes:
    keys = list(traits)
    vals = [traits[k] for k in keys]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    vals += vals[:1]
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    ax.plot(ang, vals, color="#C0392B", linewidth=2)
    ax.fill(ang, vals, color="#C0392B", alpha=0.25)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(keys, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color="gray")
    fig.tight_layout()
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def sim_strategy(kind):
    """비교용 전략 시뮬 → 거래내역 DataFrame."""
    held = set(LABELS)
    trades = [{"account_id": "X", "date": DATES[0], "ticker": l, "side": "BUY",
               "quantity": QTY[l], "price": ENTRY[l]} for l in LABELS]
    for step in range(1, N + 1):
        for l in list(held):
            pct = PRICES.loc[step, l] / ENTRY[l] - 1
            sell = (kind == "disciplined" and pct <= -0.10)
            if sell:
                trades.append({"account_id": "X", "date": DATES[step],
                               "ticker": l, "side": "SELL",
                               "quantity": QTY[l], "price": float(PRICES.loc[step, l])})
                held.discard(l)
    return pd.DataFrame(trades)


def recommended_shadow_params(style: str) -> dict:
    """전략 성향별 Shadow 규칙 추천값(%)을 반환한다."""
    base = {
        "stop_loss": 10,
        "take_profit": 25,
        "trailing": 8,
        "min_hold": 2,
        "use_regime": True,
        "b_sl": 8,
        "b_tp": 18,
        "b_tr": 6,
        "t_sl": 12,
        "t_tp": 35,
        "t_tr": 10,
    }
    if style == "quick-profit-taker":
        base.update({"take_profit": 30, "trailing": 7, "min_hold": 3,
                     "b_tp": 20, "t_tp": 40})
    elif style == "loss-delayer":
        base.update({"stop_loss": 8, "trailing": 7, "min_hold": 2,
                     "b_sl": 7, "t_sl": 10})
    elif style == "disposition-prone":
        base.update({"stop_loss": 8, "take_profit": 30, "trailing": 7,
                     "min_hold": 3, "b_sl": 7, "b_tp": 20,
                     "t_sl": 10, "t_tp": 40})
    return base


# ───────────────────────── 화면 ─────────────────────────
st.title("🧠 투자 심리검사")

if not st.session_state.started:
    st.markdown(
        "당신은 지금 **10개 종목**을 각 200만 원씩(총 2,000만 원) 보유하고 "
        "있습니다. 앞으로 **10주** 동안 매주 가격이 바뀝니다. "
        "**팔고 싶은 종목을 직접 고르세요.**\n\n"
        "- 종목 정체는 검사가 끝나면 공개됩니다(블라인드).\n"
        "- 당신의 매매 기록으로 **5대 투자심리 편향**을 진단하고, "
        "**맞춤 솔루션**과 개인 리포트를 드립니다.\n\n"
        "> 5종목은 횡보(박스권), 5종목은 상승추세에서 가져왔습니다. "
        "어느 게 어느 쪽인지는 비밀입니다.")
    if st.button("검사 시작 ▶", type="primary"):
        st.session_state.started = True
        st.rerun()

elif not st.session_state.finished:
    step = st.session_state.step
    st.progress((step - 1) / N, text=f"{step}주차 / {N}주")
    st.caption("이번 주 가격입니다. 매도할 종목을 체크하고 '다음 주'를 누르세요.")

    picked = []
    for lab in LABELS:
        if lab not in st.session_state.holdings:
            continue
        now = float(PRICES.loc[step, lab])
        pct = now / ENTRY[lab] - 1
        c1, c2, c3, c4 = st.columns([1.2, 2, 2, 1.4])
        c1.markdown(f"### {lab}")
        c2.metric("현재가", f"{now:,.0f}원", f"{pct*100:+.1f}%")
        c3.caption(f"매수가 {ENTRY[lab]:,.0f}원 · {QTY[lab]}주")
        if c4.checkbox("매도", key=f"sell_{lab}_{step}"):
            picked.append(lab)
        st.divider()

    col_a, col_b = st.columns(2)
    if col_a.button("다음 주 ▶", type="primary"):
        record_sells(picked, step)
        if step >= N or not st.session_state.holdings:
            st.session_state.finished = True
        else:
            st.session_state.step += 1
        st.rerun()
    if col_b.button("지금 검사 종료"):
        record_sells(picked, step)
        st.session_state.finished = True
        st.rerun()

else:
    tdf = pd.DataFrame(st.session_state.trades)
    prof = P.profile(tdf, PRICES, META, DATES)

    st.header(f"🔎 진단 결과 — {prof['type']}")
    st.caption(f"매매 {prof['n_sells']}회 · 평균 보유 {prof['avg_hold']:.1f}주")

    rp = radar_png(prof["traits"])
    cL, cR = st.columns([1, 1])
    cL.image(rp, caption="5대 행동편향 (0~100, 높을수록 심함)")
    with cR:
        for k, v in prof["traits"].items():
            st.write(f"**{k}** — {v:.0f}점")
            st.progress(v / 100)

    # 성과 비교
    user_final, inv = P.equity_of(tdf, PRICES, DATES, QTY)
    bh_final, _ = P.equity_of(sim_strategy("buyhold"), PRICES, DATES, QTY)
    dc_final, _ = P.equity_of(sim_strategy("disciplined"), PRICES, DATES, QTY)
    up = lambda f: (f / inv - 1) * 100
    st.subheader("📈 성과 비교 (같은 시장, 다른 습관)")
    st.table(pd.DataFrame({
        "전략": ["당신의 매매", "그냥 보유(무편향)", "손절 규율(-10%)"],
        "수익률": [f"{up(user_final):+.1f}%", f"{up(bh_final):+.1f}%",
                f"{up(dc_final):+.1f}%"],
    }))
    missed = None
    if up(bh_final) - up(user_final) > 1:
        missed = (f"그냥 들고만 있었어도 {up(bh_final):+.1f}%였습니다. "
                  f"매매로 {up(bh_final)-up(user_final):.1f}%p를 깎아먹었습니다.")
        st.warning(missed)

    # 전략 프로필 + Shadow Backtesting
    st.subheader("🕶️ Shadow Backtesting (규칙 준수 점검)")
    inferred = SB.infer_strategy_profile(tdf, PRICES, DATES)
    rec = recommended_shadow_params(inferred["style"])

    if "shadow_stop_loss" not in st.session_state:
        st.session_state.shadow_stop_loss = rec["stop_loss"]
        st.session_state.shadow_take_profit = rec["take_profit"]
        st.session_state.shadow_trailing = rec["trailing"]
        st.session_state.shadow_min_hold = min(max(1, rec["min_hold"]), max(1, N))
        st.session_state.shadow_use_regime_rules = rec["use_regime"]
        st.session_state.shadow_box_sl = rec["b_sl"]
        st.session_state.shadow_box_tp = rec["b_tp"]
        st.session_state.shadow_box_tr = rec["b_tr"]
        st.session_state.shadow_trend_sl = rec["t_sl"]
        st.session_state.shadow_trend_tp = rec["t_tp"]
        st.session_state.shadow_trend_tr = rec["t_tr"]

    cset1, cset2 = st.columns([1.6, 1])
    if cset1.button("추천값 적용", key="shadow_apply_rec"):
        st.session_state.shadow_stop_loss = rec["stop_loss"]
        st.session_state.shadow_take_profit = rec["take_profit"]
        st.session_state.shadow_trailing = rec["trailing"]
        st.session_state.shadow_min_hold = min(max(1, rec["min_hold"]), max(1, N))
        st.session_state.shadow_use_regime_rules = rec["use_regime"]
        st.session_state.shadow_box_sl = rec["b_sl"]
        st.session_state.shadow_box_tp = rec["b_tp"]
        st.session_state.shadow_box_tr = rec["b_tr"]
        st.session_state.shadow_trend_sl = rec["t_sl"]
        st.session_state.shadow_trend_tp = rec["t_tp"]
        st.session_state.shadow_trend_tr = rec["t_tr"]
        st.rerun()
    cset2.caption(
        f"추천(스타일={inferred['style']}) · SL {rec['stop_loss']}% / TP {rec['take_profit']}% / TR {rec['trailing']}%"
    )

    with st.expander("Shadow 규칙 설정", expanded=False):
        stop_loss = st.slider("손절 규칙 (%)", 3, 30, 10, key="shadow_stop_loss") / 100
        take_profit = st.slider("익절 규칙 (%)", 5, 60, 25, key="shadow_take_profit") / 100
        trailing = st.slider("트레일링 손절 (%)", 3, 30, 8, key="shadow_trailing") / 100
        min_hold = st.slider("최소 보유(주)", 1, max(1, N), 1, key="shadow_min_hold")
        use_regime_rules = st.checkbox("국면별 규칙 분리 적용(박스/추세)", key="shadow_use_regime_rules")

        box_profile = None
        trend_profile = None
        if use_regime_rules:
            st.caption("박스권 종목 규칙")
            b_sl = st.slider("박스 손절 (%)", 3, 30, 8, key="shadow_box_sl") / 100
            b_tp = st.slider("박스 익절 (%)", 5, 60, 18, key="shadow_box_tp") / 100
            b_tr = st.slider("박스 트레일링 (%)", 3, 30, 6, key="shadow_box_tr") / 100

            st.caption("추세장 종목 규칙")
            t_sl = st.slider("추세 손절 (%)", 3, 30, 12, key="shadow_trend_sl") / 100
            t_tp = st.slider("추세 익절 (%)", 5, 60, 35, key="shadow_trend_tp") / 100
            t_tr = st.slider("추세 트레일링 (%)", 3, 30, 10, key="shadow_trend_tr") / 100

            box_profile = SB.StrategyProfile(
                stop_loss_pct=b_sl,
                take_profit_pct=b_tp,
                trailing_stop_pct=b_tr,
                min_hold_steps=min_hold,
            )
            trend_profile = SB.StrategyProfile(
                stop_loss_pct=t_sl,
                take_profit_pct=t_tp,
                trailing_stop_pct=t_tr,
                min_hold_steps=min_hold,
            )

    shadow_profile = SB.StrategyProfile(
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        trailing_stop_pct=trailing,
        min_hold_steps=min_hold,
    )
    profile_by_regime = None
    regime_map = None
    if use_regime_rules:
        profile_by_regime = {"box": box_profile, "trend": trend_profile}
        regime_map = REGIME

    shadow = SB.shadow_backtest(
        tdf,
        PRICES,
        DATES,
        shadow_profile,
        regime_map=regime_map,
        profile_by_regime=profile_by_regime,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("규칙 준수율", f"{shadow['summary']['rule_adherence_pct']:.1f}%")
    c2.metric("위반 건수", f"{shadow['summary']['n_violations']}건")
    c3.metric("기회비용 합계", f"{shadow['summary']['total_opportunity_cost_pct']:+.1f}%p")

    st.caption(
        "추정 전략 프로필 · "
        f"스타일 {inferred['style']} · "
        f"추정 익절 {((inferred['inferred_take_profit_pct'] or 0) * 100):.1f}% · "
        f"추정 손절 {((inferred['inferred_stop_loss_pct'] or 0) * 100):.1f}%"
    )

    if not shadow["events"].empty:
        st.write("규칙 위반/감정 매매 의심 지점")
        cols = [
            "ticker", "regime", "issue_label", "severity",
            "actual_exit_date", "shadow_exit_date",
            "delay_steps", "actual_return_pct", "shadow_return_pct",
            "opportunity_cost_pct", "narrative", "psychology_hint", "action_hint",
        ]
        st.dataframe(shadow["events"][cols].sort_values("opportunity_cost_pct", ascending=False),
                     use_container_width=True)
    else:
        st.success("Shadow 규칙 기준으로 뚜렷한 위반 지점이 발견되지 않았습니다.")

    # 실제 vs Shadow 청산 타임라인
    if not shadow["comparison"].empty:
        st.write("실제 청산 vs Shadow 청산 타임라인")
        ticker_pick = st.selectbox("종목 선택", shadow["comparison"]["ticker"].tolist(), key="shadow_timeline_ticker")
        row = shadow["comparison"][shadow["comparison"]["ticker"] == ticker_pick].iloc[0]

        x_dates = [pd.Timestamp(d).normalize() for d in DATES]
        y = [float(PRICES.loc[i, ticker_pick]) for i in range(len(DATES))]

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.plot(x_dates, y, color="#1A2332", linewidth=2, label="가격")

        entry_d = pd.Timestamp(row["entry_date"]).normalize()
        actual_d = pd.Timestamp(row["actual_exit_date"]).normalize()
        shadow_d = pd.Timestamp(row["shadow_exit_date"]).normalize()
        entry_y = float(PRICES.loc[x_dates.index(entry_d), ticker_pick]) if entry_d in x_dates else y[0]
        actual_y = float(PRICES.loc[x_dates.index(actual_d), ticker_pick]) if actual_d in x_dates else y[-1]
        shadow_y = float(PRICES.loc[x_dates.index(shadow_d), ticker_pick]) if shadow_d in x_dates else y[-1]

        ax.scatter([entry_d], [entry_y], color="#2E86C1", s=70, marker="o", label="진입")
        ax.scatter([actual_d], [actual_y], color="#C0392B", s=85, marker="X", label="실제 청산")
        ax.scatter([shadow_d], [shadow_y], color="#117A65", s=85, marker="D", label="Shadow 청산")
        ax.set_title(f"{ticker_pick} 청산 타이밍 비교", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 국면별 처분효과
    st.subheader("🧪 국면별 처분효과 (PGR−PLR)")
    def fmt(x):
        return "측정불가" if x is None else f"{x*100:+.0f}%p"
    st.write(f"- 박스권(2014–15): **{fmt(prof['de_box'])}** · "
             f"추세장(2020–21): **{fmt(prof['de_trend'])}** · "
             f"전체: **{fmt(prof['de_all'])}**")

    # 정체 공개
    st.subheader("🎭 종목 정체 공개")
    reveal_rows = []
    rev_lines = []
    for _, m in META.iterrows():
        lab = m["label"]
        fin_pct = PRICES.loc[N, lab] / ENTRY[lab] * 100 - 100
        tag = "📦박스" if m["regime"] == "box" else "📈추세"
        reveal_rows.append({"라벨": lab, "종목": m["name"], "국면": tag,
                            "기간": m["period"], "최종": f"{fin_pct:+.1f}%"})
        rev_lines.append(f"{lab} = {m['name']} ({tag} {m['period']}, "
                         f"{fin_pct:+.1f}%)")
    st.table(pd.DataFrame(reveal_rows))

    # 솔루션
    st.subheader("💡 당신을 위한 솔루션")
    sols = P.solution_for(prof)
    for s in sols:
        st.markdown(f"- {s}")

    # docx
    extras = {
        "pnl_table": [("당신의 매매", up(user_final)),
                      ("그냥 보유(무편향)", up(bh_final)),
                      ("손절 규율(-10%)", up(dc_final))],
        "missed_note": missed,
        "reveal": rev_lines,
        "solutions": sols,
        "shadow": {
            "inferred": inferred,
            "summary": shadow["summary"],
        },
    }
    docx_bytes = make_docx(prof, extras, rp)
    st.download_button("📄 개인 진단 리포트(docx) 내려받기", docx_bytes,
                       file_name="투자심리검사_진단리포트.docx",
                       mime=("application/vnd.openxmlformats-officedocument"
                             ".wordprocessingml.document"))

    if st.button("다시 검사하기"):
        init_state()
        st.rerun()
