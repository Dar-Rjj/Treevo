import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Weighted High-Low Price Difference
    df['vol_weighted_high_low'] = (df['high'] - df['low']) * df['volume']
    
    # Daily Price Range and Net Movement
    df['daily_price_range'] = df['high'] - df['low']
    df['net_movement'] = df['close'] - df['open']
    
    # Simple Moving Average (SMA) and Close to SMA Ratio
    N_short = 5
    N_long = 20
    df['sma_5'] = df['close'].rolling(window=N_short).mean()
    df['sma_20'] = df['close'].rolling(window=N_long).mean()
    df['close_to_sma_5_ratio'] = df['close'] / df['sma_5']
    df['close_to_sma_20_ratio'] = df['close'] / df['sma_20']
    
    # Historical Volatility
    df['historical_volatility'] = df['close'].rolling(window=N_long).std()
    
    # Volume Change %
    df['volume_change_pct'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['volume_change_pct'].fillna(0, inplace=True)
    
    # Volume Impact Factor
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['volume_impact_factor'] = df['volume'] * abs(df['price_change'])
    
    # Trading Volume Spike
    df['volume_spike'] = df['volume'] - df['volume'].shift(1)
    df['volume_spike_indicator'] = np.where(df['volume_spike'] > 0, 1, -1)
    
    # Short Term Composite Indicator (N=5)
    df['short_term_composite'] = (df['daily_price_range'] + df['net_movement'] + df['close_to_sma_5_ratio'] + df['volume_change_pct']) / 4
    
    # Medium Term Composite Indicator (N=20)
    df['medium_term_composite'] = (df['daily_price_range'] + df['net_movement'] + df['close_to_sma_20_ratio'] + df['volume_change_pct']) / 4
    
    # Historical High-Low Range and Momentum
    df['hist_high_low_momentum'] = df['vol_weighted_high_low'].rolling(window=5).sum() * df['price_change']
    
    # Overnight Sentiment
    df['overnight_return'] = np.log(df['volume']) * (df['open'] / df['close'].shift(1))
    df['overnight_sentiment'] = df['avg_high_low_close_open'] - df['overnight_return']
    
    # Intraday and Overnight Signals
    df['avg_high_low_close_open'] = (df['high'] + df['low'] + df['close'] + df['open']) / 4
    df['volume_adjusted_indicator'] = df['volume'] - df['volume'].rolling(window=5).mean()
    
    # Volume Trend and Reversal Potential
    df['volume_direction'] = np.where(df['volume'] > df['volume'].shift(1), 1, -1)
    df['intraday_high_low_diff'] = df['high'] - df['low']
    df['weighted_reversal_potential'] = df['intraday_high_low_diff'] * df['volume_direction'] * df['price_change'].diff()
    
    # Final Alpha Factor
    df['alpha_factor'] = (
        df['vol_weighted_high_low'] +
        df['net_movement'] +
        df['close_to_sma_5_ratio'] +
        df['volume_change_pct'] +
        df['volume_impact_factor'] +
        df['volume_spike_indicator'] * df['volume_spike'] +
        df['short_term_composite'] +
        df['medium_term_composite'] +
        df['hist_high_low_momentum'] +
        df['overnight_sentiment'] +
        df['volume_adjusted_indicator'] +
        df['weighted_reversal_potential']
    ) / 12
    
    return df['alpha_factor']
