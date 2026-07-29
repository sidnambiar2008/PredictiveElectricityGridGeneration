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

### Case Study 1: Stable Summer Dynamics (July 25, 2026)
*   **LSTM MAE**: 309.65 MW | **Baseline MAE**: 1546.15 MW
*   **Net Advantage**: **+80.0% lower error**
*   **Engineering Insight**: When grid conditions follow established mid-summer meteorological trends learned over the model's multi-year historical training sequence, the LSTM tracks intra-day generation momentum flawlessly, while the simple 7-day average baseline model completely collapses under the variance.

### Case Study 2: Out-of-Sample Generalization (July 28, 2026)
*   **LSTM MAE**: 1047.27 MW | **Baseline MAE**: 1195.64 MW
*   **Net Advantage**: **+12.4% lower error**
*   **Engineering Insight**: Evaluated on raw, unseen data points frozen completely beyond the network's compilation window, the LSTM secures a true real-world victory. This confirms robust generalizability and proves that the system extracts meaningful tracking dependencies on fresh data streams without relying on training memory.

### Case Study 3: Volatile Spring Transitions (April 15, 2026)
*   **LSTM MAE**: 2270.00 MW | **Baseline MAE**: 1895.94 MW
*   **Net Advantage**: **-19.7% (Baseline Wins)**
*   **Engineering Insight**: Exposes the mathematical limits of a purely history-based autoregressive time-series model. Chaotic, fast-moving spring storm fronts are highly irregular year-over-year. Because the network lacks a live weather forecast feed, it experiences an autoregressive tracking lag during sudden atmospheric pressure shifts, allowing a safe, conservative rolling average baseline to win the aggregate block score.

### Case Study: Climate Shock & Grid Emergency (Winter Storm Elliott - Dec 21, 2022)
*   **LSTM MAE**: 1647.63 MW | **Baseline MAE**: 3723.26 MW
*   **Net Advantage**: **+55.7% lower error**
*   **Engineering Insight**: During extreme climate anomalies, historical rolling averages suffer complete tracking failure due to structural drift. The PyTorch LSTM successfully leverages its long-term recurrent layers to contain error propagation, correctly anticipating the timing and slope of the late-day wind generation recovery curve.

### Case Study 5: Systemic Grid Stress & Thermal Ramp (Winter Storm Elliott - Natural Gas)
*   **LSTM MAE**: 2483.71 MW | **Baseline MAE**: 5417.66 MW
*   **Net Advantage**: **+54.2% lower error**
*   **Grid Infrastructure Analysis**: As variable wind generation cratered during the December 2022 freeze, the PJM grid relied on rapid thermal natural gas deployment to sustain the massive heating load surge. The PyTorch LSTM successfully anticipated the non-linear scaling of this thermal ramp-up, capturing the core peak demand curve while the rolling baseline under-predicted generation capacity across the entire horizon.
