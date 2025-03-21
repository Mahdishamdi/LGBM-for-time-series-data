Site Demand Forecasting with LightGBM
This project demonstrates how to forecast site demand using time series data and the LightGBM model. The goal is to predict the site demand based on historical data, time-based features, and lag/rolling features.

Overview
The project involves the following steps:

Data Loading and Preparation: The dataset is loaded from a CSV file, and the timestamp is converted to a datetime object. The data is sorted by timestamp to preserve the time order.

Feature Engineering: Time-based features (hour, minute, day of the week) are extracted. Lag features and rolling statistics (mean and standard deviation) are created to capture temporal patterns.

Model Training: A LightGBM model is trained using the engineered features. The model is trained with a focus on minimizing the Mean Absolute Error (MAE).

Evaluation: The model's performance is evaluated using MAE, and the results are visualized by plotting the actual vs. predicted values.

Features
Input Features
Time-based Features: hour, minute, dayofweek

Lag Features: demand_lag1, demand_lag2, pv_lag1

Rolling Features: demand_rolling_mean_3, demand_rolling_std_3

Other Features: meter/pv_power

Target Variable
meter/site_demand: The site demand to be predicted.
