"""
Quick smoke test for valuation_logic.py
Uses a VITL-like dataset to verify calculation correctness.
"""
import sys
from tradingagents.agents.utils.valuation_logic import DCFInputs, run_valuation

# --- VITL-like numbers (from actual fundamentals_report.md) ---
inputs = DCFInputs(
    free_cash_flow_ttm=46.0,
    shares_outstanding=46.1,
    net_debt=-59.8,
    current_price=14.51,
    growth_rate_base=0.20,
    growth_rate_bull=0.30,
    growth_rate_bear=0.08,
    projection_years=5,
    beta=1.207,
    risk_free_rate=0.043,
    equity_risk_premium=0.055,
    terminal_growth_rate=0.03,
    eps_ttm=1.44,
    ebitda_ttm=102.9,
    revenue_ttm=759.4,
    peer_pe=18.0,
    peer_ev_ebitda=14.0,
    peer_ps=1.5,
)

result = run_valuation(inputs)

print("=" * 60)
print("VITL Valuation Smoke Test")
print("=" * 60)
print(f"  Bear DCF:          ${result.dcf_bear:.2f}")
print(f"  Base DCF:          ${result.dcf_base:.2f}")
print(f"  Bull DCF:          ${result.dcf_bull:.2f}")
print(f"  DCF Midpoint:      ${result.dcf_midpoint:.2f}")
print(f"  P/E Target:        ${result.pe_target:.2f}" if result.pe_target else "  P/E Target:        N/A")
print(f"  EV/EBITDA Target:  ${result.ev_ebitda_target:.2f}" if result.ev_ebitda_target else "  EV/EBITDA Target:  N/A")
print(f"  P/S Target:        ${result.ps_target:.2f}" if result.ps_target else "  P/S Target:        N/A")
print(f"  Implied Growth:    {result.implied_growth_rate:.1%}" if result.implied_growth_rate else "  Implied Growth:    N/A")
print(f"  Consensus Low:     ${result.consensus_low:.2f}")
print(f"  Consensus High:    ${result.consensus_high:.2f}")
print(f"  Consensus Mid:     ${result.consensus_midpoint:.2f}")
print(f"  Market Price:      $14.51")
print(f"  Discount:          {result.discount_to_midpoint_pct:+.1f}%" if result.discount_to_midpoint_pct else "")
print(f"  VERDICT:           {result.verdict}")
print()

assert result.dcf_base > 1, f"Base DCF too low: {result.dcf_base}"
assert result.dcf_bull > result.dcf_base > result.dcf_bear, "Scenario ordering wrong"
assert result.consensus_low <= result.consensus_midpoint <= result.consensus_high, "Consensus range wrong"
assert result.verdict in ("UNDERVALUED", "FAIRLY VALUED", "OVERVALUED", "UNCERTAIN"), "Invalid verdict"
print("All assertions passed!")
print()
print("--- Sample report output (first 40 lines) ---")
for i, line in enumerate(result.formatted_report.splitlines()[:40]):
    print(line)

print("\n" + "=" * 60)
print("Negative FCF Valuation Test (ONDS Scenario)")
print("=" * 60)
inputs_neg = DCFInputs(
    free_cash_flow_ttm=-35.0,
    shares_outstanding=463.0,
    net_debt=-415.0,
    current_price=10.90,
    growth_rate_base=0.10,
    growth_rate_bull=0.20,
    growth_rate_bear=0.0,
    projection_years=5,
    beta=2.58,
    risk_free_rate=0.043,
    equity_risk_premium=0.055,
    terminal_growth_rate=0.03
)
result_neg = run_valuation(inputs_neg)
print("--- Sample report output ---")
for i, line in enumerate(result_neg.formatted_report.splitlines()[:50]):
    print(line)
