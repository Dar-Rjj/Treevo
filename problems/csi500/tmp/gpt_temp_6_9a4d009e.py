import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def heuristics_v2(df):
    # Calculate Dynamic Price Momentum
    df['Close_10_days_ago'] = df['close'].shift(10)
    df['Price_Diff'] = df['close'] - df['Close_10_days_ago']
    df['Smoothed_Momentum'] = df['Price_Diff'].ewm(span=10, adjust=False).mean()

    # Incorporate Volume Dynamics
    df['Volume_Variability'] = df['volume'].rolling(window=10).std()

    # Evaluate Adaptive Short-Term Trend
    df['5_day_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Trend_Strength'] = (df['close'] > df['5_day_EMA']).astype(int)
    df['Avg_Range_10_days'] = (df['high'] - df['low']).rolling(window=10).mean()
    df.loc[df['Trend_Strength'] == 1, 'Trend_Strength'] = (df['close'] - df['5_day_EMA']) / df['Avg_Range_10_days']

    # Combine Momentum and Volume
    df['Combined_Momentum_Volume'] = df['Smoothed_Momentum'] * df['Volume_Variability']

    # Integrate Short-Term Price Oscillations
    df['Gain'] = np.where(df['close'] > df['close'].shift(1), df['close'] - df['close'].shift(1), 0)
    df['Loss'] = np.where(df['close'] < df['close'].shift(1), df['close'].shift(1) - df['close'], 0)
    df['Avg_Gain_14'] = df['Gain'].rolling(window=14).mean()
    df['Avg_Loss_14'] = df['Loss'].rolling(window=14).mean()
    df['RS_14'] = df['Avg_Gain_14'] / df['Avg_Loss_14']
    df['RSI_14'] = 100 - (100 / (1 + df['RS_14']))

    # Incorporate Adaptive Market Microstructure
    df['Spread'] = (df['high'] - df['low']).rolling(window=10).mean()
    df['Order_Book_Imbalance'] = ((df['high'] + df['low']) / 2 - df['close'])
    df['Microstructure_Factor'] = df['Spread'] * df['Order_Book_Imbalance']

    # Determine Preliminary Factor Value
    df['Preliminary_Factor'] = (df['Combined_Momentum_Volume'] * df['Trend_Strength'] 
                                + df['RSI_14'] + df['Microstructure_Factor'] + 0.3)

    # Enhance with Machine Learning
    X = df[['Smoothed_Momentum', 'Volume_Variability', 'Trend_Strength', 'RSI_14', 'Microstructure_Factor']].dropna()
    y = df['close'].pct_change().shift(-1).fillna(0)[X.index]  # Future returns
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    predicted_returns = model.predict(X)
    df.loc[X.index, 'Adjusted_Final_Factor'] = df.loc[X.index, 'Preliminary_Factor'] + predicted_returns

    return df['Adjusted_Final_Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
# print(factor_values)
