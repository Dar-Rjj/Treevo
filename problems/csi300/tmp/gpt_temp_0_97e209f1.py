import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency Framework
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['directional_efficiency'] = np.sign(df['close'] - df['open']) * df['price_efficiency']
    df['efficiency_persistence'] = df['directional_efficiency'] - df['directional_efficiency'].shift(2)
    df['efficiency_volatility'] = df['directional_efficiency'].rolling(window=5, min_periods=3).std()
    
    # Volume Asymmetry Analysis
    df['buy_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)) * df['volume']
    df['sell_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)) * df['volume']
    df['asymmetry_ratio'] = df['buy_pressure'] / (df['buy_pressure'] + df['sell_pressure']).replace(0, np.nan)
    df['asymmetry_momentum'] = df['asymmetry_ratio'] - df['asymmetry_ratio'].shift(5)
    
    # Market State Identification
    df['price_range_ratio'] = (df['high'] - df['low']) / df['close']
    df['range_median'] = df['price_range_ratio'].rolling(window=20, min_periods=10).median()
    df['state_score'] = df['price_range_ratio'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)).astype(int) if len(x.dropna()) >= 5 else np.nan
    )
    
    # Liquidity-Adjusted Gap Signals
    df['gap_signal'] = (df['open'] / df['close'].shift(1) - 1) * (df['close'] / df['open'] - 1)
    df['amount_mean_10d'] = df['amount'].rolling(window=10, min_periods=5).mean()
    df['liquidity_filter'] = df['amount'] / df['amount_mean_10d'].shift(1)
    df['adjusted_gap_factor'] = -df['gap_signal'] * df['liquidity_filter']
    
    # Momentum Acceleration Framework
    df['momentum_signal'] = (df['close'] / df['close'].shift(5) - 1) - (df['close'] / df['close'].shift(10) - 1)
    df['volume_mean_5d'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['volume_mean_20d'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_signal'] = (df['volume'] / df['volume_mean_5d'].shift(1)) - (df['volume'] / df['volume_mean_20d'].shift(1))
    df['momentum_acceleration'] = df['momentum_signal'] * df['volume_signal']
    
    # Adaptive Signal Construction
    df['state_weighted_efficiency'] = df['efficiency_persistence'] * (1 + df['state_score'] * 0.5)
    df['volatility_scaled_asymmetry'] = df['asymmetry_momentum'] / df['efficiency_volatility'].replace(0, np.nan)
    
    # Multi-Dimensional Integration
    df['efficiency_asymmetry_blend'] = (df['state_weighted_efficiency'] * 0.6 + 
                                       df['volatility_scaled_asymmetry'] * 0.4)
    
    df['directional_confirmation'] = df['efficiency_asymmetry_blend'] * np.sign(df['directional_efficiency'])
    
    df['momentum_confirmed_signal'] = df['directional_confirmation'] * np.tanh(df['momentum_acceleration'] * 2)
    
    df['final_factor'] = (df['momentum_confirmed_signal'] * 0.7 + 
                         df['adjusted_gap_factor'] * 0.3)
    
    return df['final_factor']
