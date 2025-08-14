import pandas as pd
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate the sum of the past 5 days' closing prices
    sum_past_5 = df['close'].rolling(window=5).sum()
    
    # Predict the next 5 days' closing prices using a linear regression model
    predictions = []
    for i in range(5, len(df)):
        X = pd.DataFrame({'day': list(range(i-4, i+1))})
        y = df['close'][i-5:i]
        model = LinearRegression().fit(X, y)
        X_future = pd.DataFrame({'day': [i+1, i+2, i+3, i+4, i+5]})
        prediction = model.predict(X_future)
        predictions.append(prediction.sum())
    
    sum_future_5 = pd.Series(predictions, index=df.index[5:])
    sum_future_5 = sum_future_5.reindex(df.index, method='bfill')
    
    # Compute the ratio and then subtract the median volume over the last 10 days
    heuristics_matrix = (sum_past_5 / sum_future_5) - df['volume'].rolling(window=10).median()
    
    return heuristics_matrix
