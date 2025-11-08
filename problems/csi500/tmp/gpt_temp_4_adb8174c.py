import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum & Acceleration Analysis
    # Multi-timeframe Price Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Price Acceleration
    data['accel_5d'] = data['momentum_5d'] - data['momentum_5d'].shift(5)
    data['accel_20d'] = data['momentum_20d'] - data['momentum_20d'].shift(20)
    
    # Momentum Alignment Assessment
    data['momentum_alignment'] = np.sign(data['momentum_5d']) == np.sign(data['momentum_20d'])
    
    # Volume Momentum & Acceleration Analysis
    # Volume Momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_context'] = data['volume'] / data['volume_20d_avg'] - 1
    
    # Volume Acceleration
    data['volume_accel_5d'] = data['volume_momentum_5d'] - data['volume_momentum_5d'].shift(5)
    
    # Volume Quality Filter
    data['dollar_volume'] = data['close'] * data['volume']
    data['dollar_volume_10d_avg'] = data['dollar_volume'].rolling(window=10, min_periods=5).mean()
    data['volume_quality'] = data['dollar_volume'] / data['dollar_volume_10d_avg']
    
    # Acceleration Divergence Detection
    data['positive_divergence'] = (data['accel_5d'] > 0) & (data['volume_accel_5d'] < 0)
    data['negative_divergence'] = (data['accel_5d'] < 0) & (data['volume_accel_5d'] > 0)
    data['confirmed_move'] = (data['accel_5d'] * data['volume_accel_5d']) > 0
    
    # Volatility & Range Context
    # Daily Range Strength
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_20d_avg'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    data['range_context'] = data['daily_range'] / data['range_20d_avg'] - 1
    
    # Volatility Regime
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    data['volatility_60d_percentile'] = data['volatility_20d'].rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x[:-1], 80)) if len(x) > 1 else np.nan
    )
    
    def get_volatility_regime(row):
        if pd.isna(row['volatility_20d']) or pd.isna(row['volatility_60d_percentile']):
            return 'normal'
        if row['volatility_60d_percentile'] > 0.8:
            return 'high'
        elif row['volatility_60d_percentile'] < 0.2:
            return 'low'
        else:
            return 'normal'
    
    data['volatility_regime'] = data.apply(get_volatility_regime, axis=1)
    
    # Multi-timeframe Signal Generation
    def generate_signal(row):
        if pd.isna(row['accel_5d']) or pd.isna(row['volume_accel_5d']):
            return 0
        
        # Strong Bullish Signal
        if (row['positive_divergence'] and 
            row['accel_5d'] > 0.02 and row['accel_20d'] > 0.01 and
            row['momentum_alignment'] and
            row['volatility_regime'] == 'low' and
            row['volume_quality'] > 0.8):
            return 2.0
        
        # Strong Bearish Signal
        elif (row['negative_divergence'] and 
              row['accel_5d'] < -0.02 and row['accel_20d'] < -0.01 and
              not row['momentum_alignment'] and
              row['volatility_regime'] == 'low' and
              row['volume_quality'] > 0.8):
            return -2.0
        
        # Moderate Bullish Signal
        elif (row['positive_divergence'] and 
              row['accel_5d'] > 0.01 and
              row['momentum_alignment'] and
              row['volatility_regime'] == 'normal' and
              row['volume_quality'] > 0.6):
            return 1.0
        
        # Moderate Bearish Signal
        elif (row['negative_divergence'] and 
              row['accel_5d'] < -0.01 and
              not row['momentum_alignment'] and
              row['volatility_regime'] == 'normal' and
              row['volume_quality'] > 0.6):
            return -1.0
        
        # Weak Signal - Confirmed Move
        elif row['confirmed_move'] and row['volatility_regime'] == 'high':
            return 0.5 * np.sign(row['accel_5d'])
        
        # No clear signal
        else:
            return 0
    
    data['signal'] = data.apply(generate_signal, axis=1)
    
    return data['signal']
