import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR for different periods
    df['atr_3'] = df['true_range'].rolling(window=3, min_periods=3).mean()
    df['atr_10'] = df['true_range'].rolling(window=10, min_periods=10).mean()
    
    # Price Momentum Acceleration
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_acceleration'] = df['momentum_3d'] - df['momentum_10d']
    
    # Range Efficiency Acceleration
    df['range_efficiency_3d'] = df['true_range'] / df['atr_3']
    df['range_efficiency_10d'] = df['true_range'] / df['atr_10']
    df['range_acceleration'] = df['range_efficiency_3d'] - df['range_efficiency_10d']
    
    # Volume Trend Analysis
    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=5).mean()
    df['volume_momentum'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    
    # Volume Persistence
    df['above_avg_volume'] = (df['volume'] > df['volume_ma_5']).astype(int)
    df['volume_persistence'] = df['above_avg_volume'].rolling(window=5, min_periods=5).sum()
    
    # Liquidity Assessment
    df['trade_size'] = df['amount'] / df['volume']
    df['liquidity_efficiency'] = df['true_range'] / df['volume']
    
    # Volume-Amplitude Alignment
    df['amplitude_strength'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * df['true_range']
    
    # Calculate Volume-Range Correlation
    volume_range_corr = []
    for i in range(len(df)):
        if i >= 9:
            window_vol = df['volume'].iloc[i-9:i+1]
            window_tr = df['true_range'].iloc[i-9:i+1]
            if len(window_vol) == 10 and len(window_tr) == 10:
                corr = np.corrcoef(window_vol, window_tr)[0, 1]
                volume_range_corr.append(corr if not np.isnan(corr) else 0)
            else:
                volume_range_corr.append(0)
        else:
            volume_range_corr.append(0)
    df['volume_range_correlation'] = volume_range_corr
    
    # Acceleration Persistence Analysis
    df['acceleration_sign'] = np.sign(df['price_acceleration'])
    df['sign_change'] = (df['acceleration_sign'] != df['acceleration_sign'].shift(1)).astype(int)
    
    # Calculate days since acceleration sign change
    days_since_change = []
    current_streak = 0
    for change in df['sign_change']:
        if change == 1:
            current_streak = 0
        else:
            current_streak += 1
        days_since_change.append(current_streak)
    df['direction_persistence'] = days_since_change
    
    # Acceleration Consistency
    df['acceleration_consistency'] = df['price_acceleration'].rolling(window=5, min_periods=5).std()
    
    # Adaptive Composite Signal
    # Multi-Layer Confirmation
    df['price_vol_confirmation'] = df['price_acceleration'] * df['volume_persistence']
    df['range_vol_confirmation'] = df['range_acceleration'] * df['volume_persistence']
    
    # Persistence-Weighted Construction
    df['persistence_weighted_accel'] = df['price_acceleration'] * df['direction_persistence']
    
    # Apply consistency weighting
    consistency_weights = 1 / (df['acceleration_consistency'] + 1e-8)
    df['weighted_by_consistency'] = df['persistence_weighted_accel'] * consistency_weights
    
    # Final Predictive Factor - Combined Acceleration Components
    df['composite_acceleration'] = (
        df['price_acceleration'] * 0.4 + 
        df['range_acceleration'] * 0.3 + 
        df['weighted_by_consistency'] * 0.3
    )
    
    # Apply volume-liquidity confirmation
    volume_confirmation = (
        df['volume_momentum'].fillna(0) + 
        df['volume_range_correlation'].fillna(0) + 
        df['amplitude_strength'].fillna(0)
    ) / 3
    
    # Final factor with multi-timeframe confirmation
    final_factor = df['composite_acceleration'] * (1 + volume_confirmation)
    
    # Clean up intermediate columns
    result = final_factor.copy()
    
    return result
