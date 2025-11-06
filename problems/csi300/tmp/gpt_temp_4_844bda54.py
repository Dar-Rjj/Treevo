import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate 5-day Price Momentum
    df['momentum_5d'] = (df['close'] / df['close'].shift(5)) - 1
    
    # Calculate Recent Return Distribution and Skewness
    returns = df['close'].pct_change()
    df['returns_10d'] = returns.rolling(window=10, min_periods=5).apply(
        lambda x: x.skew() if len(x.dropna()) >= 5 else np.nan, raw=False
    )
    
    # Apply Skewness Multiplier to Momentum
    df['skewness_multiplier'] = 1 + df['returns_10d']
    df['skew_adjusted_momentum'] = df['momentum_5d'] * df['skewness_multiplier']
    
    # Compute Dynamic Liquidity Proxy
    # Price Range Efficiency
    df['price_range_efficiency'] = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)
    
    # Volume-to-Range Ratio
    df['volume_to_range'] = df['volume'] / (df['high'] - df['low'] + 1e-8)
    
    # Detect Liquidity Regime
    df['avg_liquidity_10d'] = df['volume_to_range'].rolling(window=10, min_periods=5).mean()
    
    def classify_liquidity(current, avg):
        if current > 1.5 * avg:
            return 1.2
        elif current >= 0.7 * avg:
            return 1.0
        else:
            return 0.8
    
    df['liquidity_coefficient'] = df.apply(
        lambda row: classify_liquidity(row['volume_to_range'], row['avg_liquidity_10d']) 
        if not pd.isna(row['volume_to_range']) and not pd.isna(row['avg_liquidity_10d']) 
        else 1.0, axis=1
    )
    
    # Apply Liquidity Multiplier
    df['liquidity_adjusted_momentum'] = df['skew_adjusted_momentum'] * df['liquidity_coefficient']
    
    # Incorporate Volatility Adjustment
    # Calculate Realized Volatility
    df['volatility_10d'] = returns.rolling(window=10, min_periods=5).std()
    
    # Calculate 20-day median volatility
    df['median_volatility_20d'] = df['volatility_10d'].rolling(window=20, min_periods=10).median()
    
    def apply_volatility_scaling(current_vol, median_vol):
        if pd.isna(current_vol) or pd.isna(median_vol):
            return 1.0
        if current_vol > 1.2 * median_vol:
            return 1.3
        elif current_vol < 0.8 * median_vol:
            return 0.8
        else:
            return 1.0
    
    df['volatility_scaling'] = df.apply(
        lambda row: apply_volatility_scaling(row['volatility_10d'], row['median_volatility_20d']), 
        axis=1
    )
    
    # Combine Components
    df['final_alpha'] = df['liquidity_adjusted_momentum'] * df['volatility_scaling']
    
    return df['final_alpha']
