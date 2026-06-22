import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =============================
# Historical Price Visualization
# =============================

data = {
    'Date': [
        '2024-09-06','2024-09-05','2024-09-04','2024-09-03',
        '2024-09-02','2024-09-01','2024-08-31','2024-08-30',
        '2024-08-29','2024-08-28','2024-08-27','2024-08-26',
        '2024-08-25','2024-08-24','2024-08-23','2024-08-22',
        '2024-08-21','2024-08-20'
    ],

    'Open': [
        56161.13,57971.70,57430.35,59106.19,57326.97,58969.80,
        59117.48,59388.60,59027.47,59507.93,62879.71,64342.23,
        64176.37,64103.87,60380.95,61168.32,59014.99,59493.45
    ],

    'High': [
        56902.66,58300.58,58151.57,59815.06,59403.07,59062.07,
        59432.59,59896.89,61184.08,60236.45,63120.80,64879.71,
        64996.42,64153.79,64947.06,61408.11,61834.35,61639.36
    ],

    'Low': [
        53806.17,55712.45,55673.16,57425.17,57136.03,57217.82,
        58768.79,57768.53,58876.23,57890.68,58116.75,62849.56,
        63833.52,63619.92,60372.05,59815.25,58234.15,58610.88
    ],

    'Close': [
        53894.66,56160.49,57791.54,57431.02,59112.48,57325.49,
        58969.90,59119.48,59388.18,59027.63,59054.13,62880.66,
        64333.54,64178.99,64094.36,63811.91,61175.19,59102.79
    ],

    'Volume': [
        40414846796,310302806656,35662780312,266661961053,
        270364544524,245924499997,12403470760,322927564405,
        322249909582,402895664698,391038822198,276820406931,
        188276835555,214305856163,425309509233,276257347374,
        273715154078,316134000000
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 20))

# Open Price
axs[0].plot(df.index, df['Open'], color='blue', label='Open Price')
axs[0].set_title('Open Price Over Time')
axs[0].set_ylabel('Price in USD')
axs[0].legend()

# High Price
axs[1].plot(df.index, df['High'], color='green', label='High Price')
axs[1].set_title('High Price Over Time')
axs[1].set_ylabel('Price in USD')
axs[1].legend()

# Low Price
axs[2].plot(df.index, df['Low'], color='red', label='Low Price')
axs[2].set_title('Low Price Over Time')
axs[2].set_ylabel('Price in USD')
axs[2].legend()

# Close Price
axs[3].plot(df.index, df['Close'], color='purple', label='Close Price')
axs[3].set_title('Close Price Over Time')
axs[3].set_ylabel('Price in USD')
axs[3].legend()

plt.tight_layout()
plt.show()

# ==========================
# Predictive Modeling
# ==========================

df_model = df.copy()

df_model['Days'] = (df_model.index - df_model.index[0]).days

X = df_model[['Days']]
y = df_model['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Scatter Plot
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Days Since Start')
plt.ylabel('Price in USD')
plt.legend()
plt.show()

# Bar Plot
plt.figure(figsize=(10,5))

bar_width = 0.35
index = np.arange(len(y_test))

plt.bar(index, y_test, bar_width,
        label='Actual Prices', color='blue')

plt.bar(index + bar_width, predictions,
        bar_width, label='Predicted Prices', color='red')

plt.xlabel('Test Samples')
plt.ylabel('Price in USD')
plt.title('Actual vs Predicted Bitcoin Prices')

plt.xticks(
    index + bar_width/2,
    [f'Sample {i+1}' for i in range(len(y_test))]
)

plt.legend()
plt.tight_layout()
plt.show()

# Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
