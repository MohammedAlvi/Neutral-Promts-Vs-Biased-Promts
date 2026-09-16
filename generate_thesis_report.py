import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = "sycophancy_pairwise_evaluated_100.xlsx"
df = pd.read_excel(FILE_PATH)

# Ensure numeric types
for col in ["Phi4_Pairwise_Sycophancy_100", "Falcon3_Pairwise_Sycophancy_100"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

valid = df.dropna(subset=["Phi4_Pairwise_Sycophancy_100", "Falcon3_Pairwise_Sycophancy_100"]).copy()
phi = valid["Phi4_Pairwise_Sycophancy_100"].to_numpy()
falcon = valid["Falcon3_Pairwise_Sycophancy_100"].to_numpy()
diff = falcon - phi

# 1. Macro Statistics
def get_ci95(arr):
    n = len(arr)
    mean = np.mean(arr)
    se = stats.sem(arr)
    return stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)

phi_ci = get_ci95(phi)
falcon_ci = get_ci95(falcon)

t_stat, t_pval = stats.ttest_rel(falcon, phi)
w_stat, w_pval = stats.wilcoxon(falcon, phi)
diff_sd = np.std(diff, ddof=1)
cohens_d = np.mean(diff) / diff_sd if diff_sd != 0 else 0.0

print("\n" + "=" * 75)
print("             PAIRWISE SYCOPHANCY MACRO RESULTS (0-100 SCALE)")
print("=" * 75)
stats_df = pd.DataFrame([
    {
        "Model": "Phi-4 (14B)",
        "Mean (0-100)": np.mean(phi),
        "SD": np.std(phi, ddof=1),
        "95% CI Lower": phi_ci[0],
        "95% CI Upper": phi_ci[1]
    },
    {
        "Model": "Falcon 3 (7B)",
        "Mean (0-100)": np.mean(falcon),
        "SD": np.std(falcon, ddof=1),
        "95% CI Lower": falcon_ci[0],
        "95% CI Upper": falcon_ci[1]
    }
])
print(stats_df.round(2).to_string(index=False))

print("\n--- INFERENTIAL HYPOTHESIS TESTS (Falcon 3 vs. Phi-4) ---")
print(f"Mean Difference (Falcon - Phi): {np.mean(diff):.2f}")
print(f"Paired t-test statistic:        {t_stat:.4f} (p = {t_pval:.4e})")
print(f"Wilcoxon signed-rank stat:      {w_stat:.1f} (p = {w_pval:.4e})")
print(f"Cohen's d (Effect Size):        {cohens_d:.4f}")

# 2. Per-Style Breakdown
print("\n" + "=" * 75)
print("             BREAKDOWN BY COGNITIVE PRESSURE STYLE")
print("=" * 75)
style_table = valid.groupby("Style").agg({
    "Phi4_Pairwise_Sycophancy_100": ["mean", "std"],
    "Falcon3_Pairwise_Sycophancy_100": ["mean", "std"]
}).round(2)
print(style_table)

# 3. Direct Sycophancy Rates Plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

plot_data = valid.melt(
    id_vars=["Style"],
    value_vars=["Phi4_Pairwise_Sycophancy_100", "Falcon3_Pairwise_Sycophancy_100"],
    var_name="Model", value_name="Sycophancy_Score"
)
plot_data["Model"] = plot_data["Model"].map({
    "Phi4_Pairwise_Sycophancy_100": "Phi-4 (14B)",
    "Falcon3_Pairwise_Sycophancy_100": "Falcon 3 (7B)"
})

palette = {"Phi-4 (14B)": "#2b83ba", "Falcon 3 (7B)": "#d7191c"}

sns.barplot(
    data=plot_data, x="Style", y="Sycophancy_Score", hue="Model",
    palette=palette, ax=ax, edgecolor="black", linewidth=1.0,
    errorbar=None
)

for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    xytext=(0, 4), textcoords="offset points")

ax.set_title("Empirical Sycophancy Rate by Pressure Style (LLM-as-a-Judge)", fontsize=13, fontweight="bold", pad=14)
ax.set_ylabel("Pairwise Sycophancy Score (0-100%)", fontsize=11, fontweight="bold")
ax.set_xlabel("Pressure Vector Style", fontsize=11, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(title="Architecture", frameon=True, loc="upper left")

plt.tight_layout()
plt.savefig("standard_sycophancy_rates.png", dpi=300)
print("\n[SUCCESS] Clean chart saved as 'standard_sycophancy_rates.png'")
plt.show()