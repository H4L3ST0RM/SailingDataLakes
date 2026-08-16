import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

cost_of_false_positive = 250
cost_of_false_negative = 4000
daily_applicants = 5000
raw_fraud_rate = 0.02

legit_applicants = daily_applicants * (1 - raw_fraud_rate)
fraud_applicants = daily_applicants * raw_fraud_rate
ltv_per_legit_customer = 250
baseline_value = legit_applicants * ltv_per_legit_customer - fraud_applicants * cost_of_false_negative

breakeven_precision = cost_of_false_positive / (cost_of_false_positive + cost_of_false_negative)

precision = np.linspace(0.03, 1, 300)
recall = np.linspace(0, 1, 300)
P, R = np.meshgrid(precision, recall)

TP = R * fraud_applicants
FP = TP * (1 - P) / P
value_added = TP * cost_of_false_negative - FP * cost_of_false_positive
total_value = baseline_value + value_added

span = 500000
vmin, vmax = baseline_value - span, baseline_value + span
norm = TwoSlopeNorm(vmin=vmin, vcenter=baseline_value, vmax=vmax)
total_value_clipped = np.clip(total_value, vmin, vmax)

fig, ax = plt.subplots(figsize=(6.8, 5.2))
levels = np.linspace(vmin, vmax, 15)
cs = ax.contourf(P * 100, R * 100, total_value_clipped, levels=levels, cmap="RdBu", norm=norm, extend="both")
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label("Daily portfolio value ($, capped for display)")

be = ax.contour(P * 100, R * 100, total_value, levels=[baseline_value], colors="black", linewidths=2, linestyles="--")

ax.axvline(breakeven_precision * 100, color="black", linewidth=0, alpha=0)  # keep for reference, no visual dup
ax.text(breakeven_precision * 100 + 2, 6, f"break-even\nprecision ≈ {breakeven_precision:.1%}", fontsize=9, color="black")

ax.set_title("Onboarding Control: Portfolio Value by Precision & Recall")
ax.set_xlabel("Control Precision (%)")
ax.set_ylabel("Control Recall (%)")
fig.tight_layout()
fig.savefig("breakeven_precision.png", dpi=150)
print(f"baseline_value=${baseline_value:,.0f}  breakeven_precision={breakeven_precision:.1%}")
