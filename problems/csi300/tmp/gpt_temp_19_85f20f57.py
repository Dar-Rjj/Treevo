import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Volume Breakout Momentum factor
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Volume Breakout Detection
    # Calculate 20-day average volume (excluding current day)
    data['avg_volume_20'] = data['volume'].shift(1).rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_20']
    
    # Volume breakout classification
    data['volume_breakout_strength'] = 0
    data.loc[data['volume_ratio'] > 2.0, 'volume_breakout_strength'] = 2.0  # Strong breakout
    data.loc[(data['volume_ratio'] > 1.5) & (data['volume_ratio'] <= 2.0), 'volume_breakout_strength'] = 1.0  # Moderate
    data.loc[data['volume_ratio'] <= 1.5, 'volume_breakout_strength'] = 0.5  # Normal
    
    # Volatility Regime Classification
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate 20-day rolling standard deviation of True Range
    data['tr_volatility'] = data['true_range'].rolling(window=20, min_periods=10).std()
    
    # Calculate volatility percentiles
    data['volatility_percentile'] = data['tr_volatility'].rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) * 2 + 
                 (x.iloc[-1] > np.percentile(x.dropna(), 20)) * 1 if len(x.dropna()) >= 10 else 1, 
        raw=False
    )
    
    # Volatility adjustment factors
    data['volatility_multiplier'] = 1.0
    data.loc[data['volatility_percentile'] == 3, 'volatility_multiplier'] = 0.6  # High volatility: reduce 40%
    data.loc[data['volatility_percentile'] == 1, 'volatility_multiplier'] = 1.4  # Low volatility: amplify 40%
    
    # Price Momentum Confirmation
    # Calculate multi-timeframe returns
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['ret_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Assess momentum consistency
    data['momentum_direction_3d'] = np.sign(data['ret_3d'])
    data['momentum_direction_5d'] = np.sign(data['ret_5d'])
    data['momentum_direction_10d'] = np.sign(data['ret_10d'])
    
    # Count consistent directions
    data['positive_momentum_count'] = (
        (data['momentum_direction_3d'] > 0).astype(int) + 
        (data['momentum_direction_5d'] > 0).astype(int) + 
        (data['momentum_direction_10d'] > 0).astype(int)
    )
    
    data['negative_momentum_count'] = (
        (data['momentum_direction_3d'] < 0).astype(int) + 
        (data['momentum_direction_5d'] < 0).astype(int) + 
        (data['momentum_direction_10d'] < 0).astype(int)
    )
    
    # Momentum consistency score
    data['momentum_consistency'] = 0
    # Strong momentum: all timeframes same direction
    data.loc[data['positive_momentum_count'] == 3, 'momentum_consistency'] = 2.0
    data.loc[data['negative_momentum_count'] == 3, 'momentum_consistency'] = -2.0
    # Moderate momentum: majority same direction
    data.loc[data['positive_momentum_count'] == 2, 'momentum_consistency'] = 1.0
    data.loc[data['negative_momentum_count'] == 2, 'momentum_consistency'] = -1.0
    
    # Generate Composite Alpha Factor
    # Base signal = Volume ratio × Momentum consistency
    data['base_signal'] = data['volume_ratio'] * data['momentum_consistency']
    
    # Volatility adjustment
    data['adjusted_signal'] = data['base_signal'] * data['volatility_multiplier']
    
    # Price Action Filter
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['bullish_confirmation'] = ((data['close'] > data['open']) & (data['close'] > data['mid_price'])).astype(int)
    data['bearish_confirmation'] = ((data['close'] < data['open']) & (data['close'] < data['mid_price'])).astype(int)
    
    # Direction confirmation multiplier
    data['direction_multiplier'] = 1.0
    data.loc[data['bullish_confirmation'] == 1, 'direction_multiplier'] = 1.2
    data.loc[data['bearish_confirmation'] == 1, 'direction_multiplier'] = 0.8
    
    # Final factor
    data['factor'] = data['adjusted_signal'] * data['direction_multiplier']
    
    # Clean up intermediate columns
    result = data['factor'].copy()
    
    return result
