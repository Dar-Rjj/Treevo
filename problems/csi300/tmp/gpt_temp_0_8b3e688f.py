import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility-Regime Adaptive Price-Volume Divergence
    # Multi-Timeframe Volatility Regimes
    data['hl_range'] = (data['high'] - data['low']) / data['close']
    data['realized_vol_5d'] = data['hl_range'].rolling(window=5).std()
    
    data['close_returns'] = data['close'].pct_change()
    data['realized_vol_20d'] = data['close_returns'].rolling(window=20).std()
    
    data['vol_regime_ratio'] = data['realized_vol_5d'] / data['realized_vol_20d']
    
    # Price-Volume Relationship Shifts
    data['volume_change'] = data['volume'].pct_change()
    data['price_volume_corr_3d'] = data['close_returns'].rolling(window=3).corr(data['volume_change'])
    data['price_volume_corr_10d'] = data['close_returns'].rolling(window=10).corr(data['volume_change'])
    data['corr_divergence'] = data['price_volume_corr_3d'] - data['price_volume_corr_10d']
    
    # Regime-Adaptive Signal
    # Weight correlation divergence by volatility regime
    regime_weights = np.where(data['vol_regime_ratio'] > 1.2, 1.5,  # High volatility: amplify
                             np.where(data['vol_regime_ratio'] < 0.8, 0.7, 1.0))  # Low volatility: dampen
    
    data['abs_price_return_3d'] = data['close_returns'].rolling(window=3).apply(lambda x: np.abs(x).sum())
    
    # Generate final signal
    signal = data['corr_divergence'] * regime_weights * data['abs_price_return_3d']
    
    # Direction consistency adjustment
    data['price_trend_3d'] = data['close'].rolling(window=3).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    data['price_trend_10d'] = data['close'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    
    trend_alignment = np.where(data['price_trend_3d'] == data['price_trend_10d'], 1.2, 0.8)
    
    final_signal = signal * trend_alignment
    
    return final_signal
