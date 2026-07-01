import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT = "/home/claude/covid"

# ---------- LOAD & CLEAN ----------
df = pd.read_csv(f"{OUT}/covid_data.csv", parse_dates=["date"])

# Data cleaning steps
df = df.drop_duplicates()
df = df.sort_values(["country", "date"]).reset_index(drop=True)
df["daily_cases"] = df["daily_cases"].clip(lower=0)
df["daily_deaths"] = df["daily_deaths"].clip(lower=0)
df["daily_recovered"] = df["daily_recovered"].clip(lower=0)
df["active_cases"] = df["active_cases"].clip(lower=0)
assert df.isnull().sum().sum() == 0, "Found missing values"

global_df = df.groupby("date", as_index=False)[
    ["daily_cases", "daily_deaths", "daily_recovered", "total_cases",
     "total_deaths", "total_recovered", "active_cases"]
].sum()
global_df["case_7day_avg"] = global_df["daily_cases"].rolling(7).mean()
global_df["death_7day_avg"] = global_df["daily_deaths"].rolling(7).mean()
global_df["recovered_7day_avg"] = global_df["daily_recovered"].rolling(7).mean()
global_df["cfr_pct"] = (global_df["total_deaths"] / global_df["total_cases"] * 100)
global_df["recovery_rate_pct"] = (global_df["total_recovered"] / global_df["total_cases"] * 100)

# ---------- SUMMARY STATS ----------
summary = {
    "total_cases": int(global_df["total_cases"].iloc[-1]),
    "total_deaths": int(global_df["total_deaths"].iloc[-1]),
    "total_recovered": int(global_df["total_recovered"].iloc[-1]),
    "active_cases": int(global_df["active_cases"].iloc[-1]),
    "case_fatality_rate": round(global_df["cfr_pct"].iloc[-1], 2),
    "recovery_rate": round(global_df["recovery_rate_pct"].iloc[-1], 2),
    "peak_daily_cases": int(global_df["daily_cases"].max()),
    "peak_daily_cases_date": str(global_df.loc[global_df["daily_cases"].idxmax(), "date"].date()),
    "peak_daily_deaths": int(global_df["daily_deaths"].max()),
    "peak_daily_deaths_date": str(global_df.loc[global_df["daily_deaths"].idxmax(), "date"].date()),
}
country_totals = (
    df.groupby("country")
    .agg(total_cases=("daily_cases", "sum"),
         total_deaths=("daily_deaths", "sum"),
         total_recovered=("daily_recovered", "sum"))
    .assign(cfr_pct=lambda d: (d.total_deaths / d.total_cases * 100).round(2))
    .sort_values("total_cases", ascending=False)
)
print(summary)
print(country_totals)

import json
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
country_totals.to_csv(f"{OUT}/country_totals.csv")

# ============ CHART 1: Daily new cases (global) with 7-day avg ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(global_df["date"], global_df["daily_cases"], color="#a8c8e8", width=1, label="Daily new cases")
ax.plot(global_df["date"], global_df["case_7day_avg"], color="#1f4e79", linewidth=2, label="7-day average")
ax.set_title("Global Daily New COVID-19 Cases", fontsize=13, fontweight="bold")
ax.set_ylabel("New cases")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/chart1_daily_cases.png")
plt.close()

# ============ CHART 2: Daily deaths with 7-day avg ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(global_df["date"], global_df["daily_deaths"], color="#f4a9a8", width=1, label="Daily deaths")
ax.plot(global_df["date"], global_df["death_7day_avg"], color="#a31515", linewidth=2, label="7-day average")
ax.set_title("Global Daily COVID-19 Deaths", fontsize=13, fontweight="bold")
ax.set_ylabel("Deaths")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/chart2_daily_deaths.png")
plt.close()

# ============ CHART 3: Cumulative cases/deaths/recovered (stacked view) ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(global_df["date"], global_df["total_cases"], label="Total cases", color="#1f77b4", linewidth=2)
ax.plot(global_df["date"], global_df["total_recovered"], label="Total recovered", color="#2ca02c", linewidth=2)
ax.plot(global_df["date"], global_df["total_deaths"], label="Total deaths", color="#d62728", linewidth=2)
ax.set_title("Cumulative Cases, Recoveries, and Deaths", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/chart3_cumulative.png")
plt.close()

# ============ CHART 4: Active cases over time ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.fill_between(global_df["date"], global_df["active_cases"], color="#f0b429", alpha=0.6)
ax.plot(global_df["date"], global_df["active_cases"], color="#a3720a", linewidth=1.5)
ax.set_title("Active Cases Over Time (Global)", fontsize=13, fontweight="bold")
ax.set_ylabel("Active cases")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/chart4_active_cases.png")
plt.close()

# ============ CHART 5: Country comparison bar chart ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ct = country_totals.reset_index()
x = range(len(ct))
ax.bar(x, ct["total_cases"], color=sns.color_palette("Blues_d", len(ct)))
ax.set_xticks(list(x))
ax.set_xticklabels(ct["country"])
ax.set_title("Total Confirmed Cases by Country", fontsize=13, fontweight="bold")
ax.set_ylabel("Total cases")
for i, v in enumerate(ct["total_cases"]):
    ax.text(i, v, f"{v/1e6:.1f}M", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/chart5_country_comparison.png")
plt.close()

# ============ CHART 6: Case fatality rate trend ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(global_df["date"], global_df["cfr_pct"], color="#8e44ad", linewidth=2, label="Case Fatality Rate (%)")
ax.plot(global_df["date"], global_df["recovery_rate_pct"], color="#16a085", linewidth=2, label="Recovery Rate (%)")
ax.set_title("Case Fatality Rate vs Recovery Rate Over Time", fontsize=13, fontweight="bold")
ax.set_ylabel("Percent (%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/chart6_rates.png")
plt.close()

# ============ CHART 7: Heatmap of monthly cases by country ============
df["month"] = df["date"].dt.to_period("M").astype(str)
pivot = df.pivot_table(index="country", columns="month", values="daily_cases", aggfunc="sum")
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Monthly cases"})
ax.set_title("Monthly New Cases Heatmap by Country", fontsize=13, fontweight="bold")
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/chart7_heatmap.png")
plt.close()

print("All charts saved.")
