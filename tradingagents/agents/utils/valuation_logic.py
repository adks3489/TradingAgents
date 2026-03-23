"""
Valuation Logic Engine for TradingAgents
=========================================
Provides deterministic, formula-based valuation calculations.
This module ensures reproducible, auditable results independent of LLM arithmetic.

Supported Methods:
  - DCF (Discounted Cash Flow) with multi-scenario analysis
  - Comparable Multiples Valuation (P/E, EV/EBITDA, P/S, P/B)
  - Reverse DCF (implied growth rate from current price)
  - Sensitivity tables for WACC and terminal growth rate
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DCFInputs:
    """All inputs needed for a DCF valuation."""
    # Core financials
    free_cash_flow_ttm: float          # Latest TTM FCF ($M)
    shares_outstanding: float          # Shares outstanding (M)
    net_debt: float                    # Total debt minus cash ($M); negative = net cash

    # Growth assumptions (decimal, e.g., 0.20 = 20%)
    growth_rate_base: float = 0.15     # Base-case FCF growth for projection period
    growth_rate_bull: float = 0.25     # Bull-case FCF growth
    growth_rate_bear: float = 0.05     # Bear-case FCF growth
    projection_years: int = 5          # Explicit projection period

    # Discount rate (WACC components or direct override)
    wacc: Optional[float] = None       # If provided, used directly
    risk_free_rate: float = 0.043      # 10-yr US Treasury yield
    equity_risk_premium: float = 0.055 # ERP (Damodaran US estimate)
    beta: float = 1.0                  # Company beta

    # Terminal value
    terminal_growth_rate: float = 0.03  # Long-run perpetuity growth rate

    # Multiples inputs (for cross-validation)
    eps_ttm: Optional[float] = None
    ebitda_ttm: Optional[float] = None
    revenue_ttm: Optional[float] = None
    book_value_per_share: Optional[float] = None
    current_price: Optional[float] = None
    peer_pe: Optional[float] = None
    peer_ev_ebitda: Optional[float] = None
    peer_ps: Optional[float] = None


@dataclass
class ValuationResult:
    """Output from valuation calculations."""
    # DCF results
    dcf_base: float
    dcf_bull: float
    dcf_bear: float
    dcf_midpoint: float

    # Reverse DCF
    implied_growth_rate: Optional[float]

    # Multiples results
    pe_target: Optional[float]
    ev_ebitda_target: Optional[float]
    ps_target: Optional[float]

    # Summary
    consensus_low: float
    consensus_high: float
    consensus_midpoint: float
    discount_to_midpoint_pct: Optional[float]   # positive = undervalued
    verdict: str                                  # "UNDERVALUED" / "FAIRLY VALUED" / "OVERVALUED"

    # Sensitivity table rows: list of (wacc, terminal_g, intrinsic_value)
    sensitivity_rows: list = field(default_factory=list)

    # Formatted report
    formatted_report: str = ""


# ---------------------------------------------------------------------------
# Core Calculation Helpers
# ---------------------------------------------------------------------------

def _compute_wacc(inputs: DCFInputs) -> float:
    """Compute WACC using CAPM if not directly provided."""
    if inputs.wacc is not None:
        return inputs.wacc
    return inputs.risk_free_rate + inputs.beta * inputs.equity_risk_premium


def _dcf_equity_value(
    fcf: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    years: int,
    net_debt: float,
    shares: float,
) -> float:
    """
    Two-stage DCF: explicit growth phase + Gordon Growth terminal value.
    Returns intrinsic value per share.
    """
    if wacc <= terminal_growth:
        # Guard against mathematical impossibility
        raise ValueError(
            f"WACC ({wacc:.2%}) must be greater than terminal growth rate ({terminal_growth:.2%})."
        )

    # Stage 1: Explicit projection
    pv_fcfs = 0.0
    for yr in range(1, years + 1):
        projected_fcf = fcf * ((1 + growth_rate) ** yr)
        pv_fcfs += projected_fcf / ((1 + wacc) ** yr)

    # Stage 2: Terminal value (Gordon Growth Model)
    terminal_fcf = fcf * ((1 + growth_rate) ** years) * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)

    # Enterprise value → Equity value
    enterprise_value = pv_fcfs + pv_terminal
    equity_value = enterprise_value - net_debt
    return equity_value / shares


def _reverse_dcf(
    current_price: float,
    fcf_per_share: float,
    wacc: float,
    terminal_growth: float,
    years: int,
) -> float:
    """
    Binary-search for the growth rate that makes DCF value == current price.
    Returns implied annual FCF growth rate.
    """
    if fcf_per_share <= 0:
        return float("nan")

    lo, hi = -0.30, 1.50
    for _ in range(60):  # 60 iterations → precision < 0.0001%
        mid = (lo + hi) / 2
        try:
            val = _dcf_equity_value(
                fcf=fcf_per_share,
                growth_rate=mid,
                wacc=wacc,
                terminal_growth=terminal_growth,
                years=years,
                net_debt=0,       # already per-share
                shares=1,
            )
        except ValueError:
            break
        if val < current_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Multiples Helpers
# ---------------------------------------------------------------------------

def _pe_target(eps: float, peer_pe: float) -> Optional[float]:
    if eps and peer_pe and eps > 0:
        return round(eps * peer_pe, 2)
    return None


def _ev_ebitda_target(
    ebitda: float,
    peer_ev_ebitda: float,
    net_debt: float,
    shares: float,
) -> Optional[float]:
    if ebitda and peer_ev_ebitda and ebitda > 0 and shares > 0:
        ev = ebitda * peer_ev_ebitda
        equity_val = ev - net_debt
        return round(equity_val / shares, 2)
    return None


def _ps_target(
    revenue: float,
    peer_ps: float,
    shares: float,
) -> Optional[float]:
    if revenue and peer_ps and shares > 0:
        return round((revenue * peer_ps) / shares, 2)
    return None


# ---------------------------------------------------------------------------
# Sensitivity Table
# ---------------------------------------------------------------------------

def _build_sensitivity_table(
    fcf: float,
    growth_rate: float,
    years: int,
    net_debt: float,
    shares: float,
) -> list:
    """
    Generate a 3x3 sensitivity table varying WACC and terminal growth.
    Returns list of (wacc, tgr, intrinsic_value) tuples.
    """
    wacc_range = [0.07, 0.09, 0.11]
    tgr_range  = [0.02, 0.03, 0.04]
    rows = []
    for w in wacc_range:
        for tg in tgr_range:
            if w <= tg:
                rows.append((w, tg, float("nan")))
                continue
            try:
                val = _dcf_equity_value(fcf, growth_rate, w, tg, years, net_debt, shares)
                rows.append((w, tg, round(val, 2)))
            except Exception:
                rows.append((w, tg, float("nan")))
    return rows


# ---------------------------------------------------------------------------
# Verdict Helper
# ---------------------------------------------------------------------------

def _verdict(discount_pct: Optional[float]) -> str:
    if discount_pct is None:
        return "UNCERTAIN"
    if discount_pct > 15:
        return "UNDERVALUED"
    if discount_pct < -15:
        return "OVERVALUED"
    return "FAIRLY VALUED"


# ---------------------------------------------------------------------------
# Main Public API
# ---------------------------------------------------------------------------

def run_valuation(inputs: DCFInputs) -> ValuationResult:
    """
    Execute all valuation analyses and return a structured ValuationResult.

    Args:
        inputs: DCFInputs with financial data and growth assumptions.

    Returns:
        ValuationResult with per-share intrinsic value estimates and a
        pre-formatted text report suitable for inclusion in the fundamentals report.
    """
    wacc = _compute_wacc(inputs)

    # --- DCF ---
    kwargs = dict(
        fcf=inputs.free_cash_flow_ttm,
        wacc=wacc,
        terminal_growth=inputs.terminal_growth_rate,
        years=inputs.projection_years,
        net_debt=inputs.net_debt,
        shares=inputs.shares_outstanding,
    )
    dcf_base = round(_dcf_equity_value(growth_rate=inputs.growth_rate_base, **kwargs), 2)
    dcf_bull = round(_dcf_equity_value(growth_rate=inputs.growth_rate_bull, **kwargs), 2)
    dcf_bear = round(_dcf_equity_value(growth_rate=inputs.growth_rate_bear, **kwargs), 2)
    dcf_midpoint = round((dcf_base + dcf_bull + dcf_bear) / 3, 2)

    # --- Reverse DCF ---
    implied_g = None
    if inputs.current_price and inputs.shares_outstanding > 0:
        fcf_ps = inputs.free_cash_flow_ttm / inputs.shares_outstanding
        implied_g = round(
            _reverse_dcf(inputs.current_price, fcf_ps, wacc,
                         inputs.terminal_growth_rate, inputs.projection_years),
            4,
        )

    # --- Multiples ---
    pe_t       = _pe_target(inputs.eps_ttm or 0, inputs.peer_pe or 0)
    eveb_t     = _ev_ebitda_target(inputs.ebitda_ttm or 0, inputs.peer_ev_ebitda or 0,
                                    inputs.net_debt, inputs.shares_outstanding)
    ps_t       = _ps_target(inputs.revenue_ttm or 0, inputs.peer_ps or 0,
                             inputs.shares_outstanding)

    # --- Consensus Range ---
    candidates = [v for v in [dcf_bear, dcf_base, pe_t, eveb_t, ps_t] if v is not None and not math.isnan(v)]
    consensus_low  = round(min(candidates), 2) if candidates else dcf_bear
    consensus_high = round(max(candidates), 2) if candidates else dcf_bull
    consensus_mid  = round((consensus_low + consensus_high) / 2, 2)

    discount_pct = None
    if inputs.current_price:
        discount_pct = round((consensus_mid - inputs.current_price) / consensus_mid * 100, 1)

    # --- Sensitivity ---
    sensitivity = _build_sensitivity_table(
        inputs.free_cash_flow_ttm,
        inputs.growth_rate_base,
        inputs.projection_years,
        inputs.net_debt,
        inputs.shares_outstanding,
    )

    verdict = _verdict(discount_pct)

    result = ValuationResult(
        dcf_base=dcf_base,
        dcf_bull=dcf_bull,
        dcf_bear=dcf_bear,
        dcf_midpoint=dcf_midpoint,
        implied_growth_rate=implied_g,
        pe_target=pe_t,
        ev_ebitda_target=eveb_t,
        ps_target=ps_t,
        consensus_low=consensus_low,
        consensus_high=consensus_high,
        consensus_midpoint=consensus_mid,
        discount_to_midpoint_pct=discount_pct,
        verdict=verdict,
        sensitivity_rows=sensitivity,
    )

    result.formatted_report = _format_report(inputs, wacc, result)
    return result


# ---------------------------------------------------------------------------
# Report Formatter
# ---------------------------------------------------------------------------

def _format_report(inputs: DCFInputs, wacc: float, r: ValuationResult) -> str:
    """Generate a Markdown-formatted valuation section for the fundamentals report."""

    lines = [
        "## Quantitative Valuation Analysis",
        "",
        "> **Methodology**: All calculations are deterministic (Python-computed). "
        "Growth assumptions are derived from historical FCF CAGR and analyst consensus. "
        "WACC is computed via CAPM.",
        "",
        "### Key Inputs",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| FCF (TTM) | ${inputs.free_cash_flow_ttm:.1f}M |",
        f"| Shares Outstanding | {inputs.shares_outstanding:.1f}M |",
        f"| Net Debt (Debt − Cash) | ${inputs.net_debt:.1f}M |",
        f"| WACC (CAPM: Rf {inputs.risk_free_rate:.1%} + β{inputs.beta:.2f} × ERP {inputs.equity_risk_premium:.1%}) | {wacc:.2%} |",
        f"| Terminal Growth Rate | {inputs.terminal_growth_rate:.1%} |",
        f"| Projection Period | {inputs.projection_years} years |",
        f"| Current Market Price | ${inputs.current_price:.2f} |" if inputs.current_price else "",
        "",
        "### DCF Valuation — Multi-Scenario",
        f"| Scenario | FCF Growth Assumption | Intrinsic Value / Share |",
        f"|----------|-----------------------|------------------------|",
        f"| 🐻 Bear Case | {inputs.growth_rate_bear:.0%} / yr | ${r.dcf_bear:.2f} |",
        f"| 📊 Base Case | {inputs.growth_rate_base:.0%} / yr | ${r.dcf_base:.2f} |",
        f"| 🐂 Bull Case | {inputs.growth_rate_bull:.0%} / yr | ${r.dcf_bull:.2f} |",
        f"| **DCF Midpoint** | — | **${r.dcf_midpoint:.2f}** |",
        "",
    ]

    if r.implied_growth_rate is not None:
        lines += ["### Reverse DCF — Market-Implied Growth Rate"]
        if math.isnan(r.implied_growth_rate):
            lines.append(f"At the current price of **${inputs.current_price:.2f}**, the market-implied growth rate cannot be calculated because the current Free Cash Flow is negative or zero.")
            lines.append("> The Reverse DCF model requires a positive trailing FCF to derive a meaningful implied growth rate.")
        else:
            lines.append(f"At the current price of **${inputs.current_price:.2f}**, the market is pricing in an FCF growth rate of **{r.implied_growth_rate:.1%} per year** over {inputs.projection_years} years.")
            lines.append("> A rate below historical CAGR suggests the market may be underestimating growth.")
        lines.append("")

    multiples_rows = []
    if r.pe_target:
        multiples_rows.append(f"| P/E (peer avg {inputs.peer_pe:.1f}x) | EPS ${inputs.eps_ttm:.2f} | ${r.pe_target:.2f} |")
    if r.ev_ebitda_target:
        multiples_rows.append(f"| EV/EBITDA (peer avg {inputs.peer_ev_ebitda:.1f}x) | EBITDA ${inputs.ebitda_ttm:.1f}M | ${r.ev_ebitda_target:.2f} |")
    if r.ps_target:
        multiples_rows.append(f"| P/S (peer avg {inputs.peer_ps:.1f}x) | Revenue ${inputs.revenue_ttm:.1f}M | ${r.ps_target:.2f} |")
    if multiples_rows:
        lines += [
            "### Comparable Multiples Valuation",
            "| Method | Input | Implied Price / Share |",
            "|--------|-------|-----------------------|",
        ] + multiples_rows + [""]

    # Sensitivity table
    lines += [
        "### DCF Sensitivity Table (Base-Case Growth)",
        "Intrinsic Value per Share at different WACC / Terminal Growth combinations:",
        "",
        "| WACC \\ Terminal g | 2.0% | 3.0% | 4.0% |",
        "|-------------------|------|------|------|",
    ]
    for wacc_v in [0.07, 0.09, 0.11]:
        row_vals = {tg: v for (w, tg, v) in r.sensitivity_rows if w == wacc_v}
        def fmt(v):
            return f"${v:.2f}" if not math.isnan(v) else "N/A"
        lines.append(
            f"| {wacc_v:.0%} | {fmt(row_vals.get(0.02, float('nan')))} | "
            f"{fmt(row_vals.get(0.03, float('nan')))} | "
            f"{fmt(row_vals.get(0.04, float('nan')))} |"
        )

    # Verdict
    lines += [
        "",
        "### Valuation Verdict",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Consensus Low | ${r.consensus_low:.2f} |",
        f"| Consensus High | ${r.consensus_high:.2f} |",
        f"| **Consensus Midpoint** | **${r.consensus_midpoint:.2f}** |",
    ]
    if inputs.current_price:
        lines.append(f"| Current Price | ${inputs.current_price:.2f} |")
    if r.discount_to_midpoint_pct is not None:
        sign = "+" if r.discount_to_midpoint_pct > 0 else ""
        lines.append(f"| **Discount / Premium to Midpoint** | **{sign}{r.discount_to_midpoint_pct:.1f}%** |")
    lines.append(f"| **Verdict** | **{r.verdict}** |")
    lines.append("")

    return "\n".join(l for l in lines if l is not None)
