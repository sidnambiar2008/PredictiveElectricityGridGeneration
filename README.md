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
