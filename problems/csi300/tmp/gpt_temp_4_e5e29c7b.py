import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Reversal Signal Component
    df = df.copy()
    
    # Calculate Intraday Return Extremes
    df['intraday_high_return'] = df['high'] / df['close'].shift(1) - 1
    df['intraday_low_return'] = df['low'] / df['close'].shift(1) - 1
    df['intraday_range'] = np.abs(df['intraday_high_return'] - df['intraday_low_return'])
    
    # Compute Volume-Weighted Intraday Pressure
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_pressure'] = df['volume'] / (df['volume_ma_5'] + 0.001)
    df['bullish_pressure'] = df['intraday_high_return'] * df['volume_pressure']
    df['bearish_pressure'] = df['intraday_low_return'] * df['volume_pressure']
    df['net_intraday_pressure'] = df['bullish_pressure'] + df['bearish_pressure']
    
    # Generate Raw Intraday Reversal Signal
    df['reversal_signal'] = -df['net_intraday_pressure'] * df['intraday_range']
    
    # Volatility Scaling Mechanism
    # Calculate Short-Term Volatility
    df['daily_return'] = df['close'] / df['close'].shift(1) - 1
    df['volatility'] = df['daily_return'].rolling(window=10).std()
    
    # Compute Volatility Adjustment
    df['volatility_scaling_factor'] = 1 / (df['volatility'] + 0.0001)
    
    # Apply Volatility Scaling
    df['scaled_reversal'] = df['reversal_signal'] * df['volatility_scaling_factor']
    
    # Multi-Timeframe Regime Detection
    # Trend Regime Classification
    df['short_term_trend'] = df['close'] / df['close'].shift(5) - 1
    df['medium_term_trend'] = df['close'] / df['close'].shift(15) - 1
    df['trend_alignment'] = np.sign(df['short_term_trend']) * np.sign(df['medium_term_trend'])
    
    # Volatility Regime Classification
    df['historical_volatility'] = df['daily_return'].rolling(window=20).std()
    df['volatility_percentile'] = df['historical_volatility'].rolling(window=50).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)) if len(x.dropna()) > 0 else 0
    )
    df['high_volatility_flag'] = (df['volatility_percentile'] > 0.7).astype(int)
    
    # Regime Adjustment Factors
    df['trend_adjustment'] = np.where(df['trend_alignment'] > 0, 1.0, 0.7)
    df['volatility_adjustment'] = np.where(df['high_volatility_flag'] == 1, 1.2, 0.9)
    df['combined_regime_factor'] = df['trend_adjustment'] * df['volatility_adjustment']
    
    # Volume Confirmation Filter
    # Volume Trend Analysis
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_10']
    df['volume_trend'] = df['volume_ma_5'] / df['volume_ma_10'] - 1
    df['volume_momentum'] = np.sign(df['volume_trend']) * df['volume_ratio']
    
    # Volume-Signal Alignment
    df['volume_confirmation'] = np.where(
        np.sign(df['volume_momentum']) == np.sign(df['scaled_reversal']), 1.0, 0.5
    )
    
    # Final Alpha Factor Construction
    df['base_factor'] = df['scaled_reversal'] * df['combined_regime_factor']
    df['volume_adjusted_factor'] = df['base_factor'] * df['volume_confirmation']
    
    return df['volume_adjusted_factor']
