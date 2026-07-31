# Predictive Electricity Grid Generation

An end-to-end multivariate time-series forecasting project to predict hourly U.S. electricity generation and fuel mix. This repository implements a deep learning pipeline to answer a critical grid question: 

**"How clean is the electricity grid right now, and what will it be in the next hour?"**

This project bridges the gap between historical data analysis and predictive analytics, using the operational framework of GridPulse.us as its foundational reference.

### ⚡ Project Blueprint & Context
This project evolves the capabilities of GridPulse, an energy analytics platform built using a rapid AI development methodology.

* **What GridPulse Does Today:** Operates on real AWS infrastructure to provide daily graph snapshots, fuel mix tracking (natural gas, coal, nuclear, renewables), and automated prose narratives detailing grid stress during extreme weather.
* **The Goal of this Repo:** Transitioning GridPulse from a descriptive analytics tool into a predictive application by forecasting next-day grid composition.

###  The Dataset: EIA-930
The project utilizes real, public data from the EIA Hourly Electric Grid Monitor (EIA-930).

* **Data Scope:** Tracks hourly U.S. electricity demand and generation by fuel type across every Balancing Authority (grid region).
* **Target Problem:** Multivariate time-series forecasting. The model ingests multi-dimensional historical inputs (parallel fuel streams, time-of-day encodings, regional demand) to predict future energy generation mixes.

### Methodology & Architecture
Following a structured machine learning lifecycle, this project relies on a SaaS-ready, scalable modeling approach:

1. **Research & Approach Selection:** While traditional statistical models (ARIMA, Exponential Smoothing, SARIMAX) are highly effective for univariate series, they struggle to capture the complex, non-linear relationships inherent in multivariate grid data. A deep learning approach was selected to better handle simultaneous input arrays.
2. **Modeling Stack:** 
    * **Baseline Model:** A rigorous Naive Persistence Baseline that calculates a rolling average of the last seven days.
    * **Core Architecture:** A PyTorch LSTM (Long Short-Term Memory) network. LSTMs natively track sequential temporal patterns and long-term dependencies across multiple parallel fuel types and grid variables.

### Case Study 1: Wind Forecasting in a High-Penetration Grid (ERCOT, July 18, 2026)
*   **LSTM MAE**: 3378.31 MW | **Baseline MAE**: 5686.02 MW
*   **Net Advantage**: **+40.6% lower error**
*   **Engineering Insight**: ERCOT carries the largest wind fleet of any region in this project, and wind output is driven mainly by persistent weather trends rather than time of day. That plays to the LSTM's actual strength — its 168-hour lookback lets it track ongoing conditions, while the diurnal-mean baseline has no way to react to what's happening right now. This result holds up in aggregate too: across the full held-out validation set, the LSTM beats the baseline on ERCOT wind by roughly 24% on average. Day-to-day results do vary — wind is a genuinely volatile signal, and some individual days still favor the baseline — but the typical-case advantage is real and consistent.

### Case Study 2: A Known Limitation — Solar Forecasting (CISO, July 31, 2026)
*   **LSTM MAE**: 1591.11 MW | **Baseline MAE**: 379.54 MW
*   **Net Advantage**: **-319.2% (Baseline Wins, by a wide margin)**
*   **Engineering Insight**: The model has no explicit signal telling it what hour of day it is — it only sees raw fuel-generation values. The diurnal-mean baseline, by contrast, is built entirely around hour-of-day matching, so it gets that information for free. CISO has the largest, sharpest solar swing of any tracked region (near-zero overnight, tens of thousands of MW at midday), so this is exactly where that gap shows up worst. This isn't a one-off: the same weakness appears across every region with meaningful solar capacity in the full validation set. The clear next step to close this gap is adding explicit time-of-day (and likely day-of-week) features to the model's input — not yet implemented here.

### Case Study 3: A Genuinely Hard Signal — Petroleum (PJM, full validation set)
*   **LSTM MAE**: 414.77 MW | **Baseline MAE**: 333.46 MW
*   **Net Advantage**: **-24.4% (Baseline Wins, narrowly)**
*   **Engineering Insight**: Petroleum is a marginal, rarely-dispatched fuel source (mostly emergency/peaker backup), and both models explain very little of its variance (R² ≈ 0.30 for each). This is presented as an aggregate statistic across the full held-out period rather than a single day, since the point here isn't "the model loses" — it's that this particular signal is close to equally hard for a sophisticated model and a naive average alike.

### Case Study 4: Out-of-Distribution Stress Test (Winter Storm Elliott, PJM Natural Gas, Dec 23, 2022)
*   **LSTM MAE**: 5487.16 MW | **Baseline MAE**: 2137.60 MW
*   **Net Advantage**: **-156.7% (Baseline Wins)**
*   **Engineering Insight**: Important context for this one: the model's training data begins January 1, 2023 — this test date predates the entire training window. Rather than typical held-out validation, this is asking the model to forecast a period it has zero learned exposure to, of any kind. The loss here reflects the general limitation of any static, non-retrained model when pushed outside its training era, not a specific architectural flaw like the diurnal gap above. Included deliberately, as an honest example of where the model's generalization breaks down.
