# Stock Market Prediction
This project leverages historical stock data to predict future stock prices. The data is pulled using the yfinance library, and Python is used for data processing, analysis, and prediction. The goal of this project is to provide an insight into stock price trends using machine learning techniques.
## Table of Contents

- **Usage**

- **Data Sources**

- **Prediction Model**

- **Results**


## Technologies Used
- **Python:** The primary language used for the project.

- **yfinance:** A Python library used to fetch historical stock data.

- **Pandas:** For data manipulation and analysis.

- **NumPy:** For numerical operations.

- **Matplotlib / Seaborn:** For data visualization.

- **Scikit-learn:** For implementing machine learning models.

- **TensorFlow/Keras:** If using deep learning models for predictions.


## Usage
### Import the necessary libraries:
To run this project, make sure you have the following Python libraries installed:

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### Pull stock data using yfinance:

```python
ticker = "AAPL"  # Example: Apple stock
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")
```
### Preprocess the data (e.g., handling missing values, creating features):
```python
data['Date'] = pd.to_datetime(data.index)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
```

### Train a model (e.g., Linear Regression):
```python
X = data[['Year', 'Month', 'Day']]  # Example features
y = data['Close']  # Target variable: closing price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```
### Make predictions and evaluate the model:

```python
predictions = model.predict(X_test)

# Plot results
plt.plot(y_test.values, label='True Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()
```

## Data Sources
The stock data is fetched from Yahoo Finance using the yfinance library. The data includes:

- Open, High, Low, Close, Volume, and Adjusted Close prices.

- The data is retrieved from Yahoo Finance by specifying the stock ticker and the date range.


## Prediction Model
The project uses Linear Regression as a basic model for stock price prediction. However, depending on the complexity and accuracy needed, you can explore more advanced models like:

- Random Forests

- Support Vector Machines

- Neural Networks (using TensorFlow/Keras)

## Results
The model provides predictions based on historical stock data. The results can be evaluated by comparing the predicted stock prices against actual prices. Various performance metrics like Mean Squared Error (MSE) and R-squared can be used to evaluate the accuracy of the model.

## Author
[Tridha Mukherjee](https://github.com/TridhaMukherjee)
