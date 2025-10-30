import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Adjusted Momentum Framework
    # Short-Term (3-7 days)
    data['momentum_3d'] = data['close'] - data['close'].shift(3)
    data['volatility_3d'] = data['close'].diff().abs().rolling(window=3, min_periods=2).sum()
    data['vol_adj_momentum_3d'] = data['momentum_3d'] / (data['volatility_3d'] + 1e-8)
    
    data['momentum_7d'] = data['close'] - data['close'].shift(7)
    data['volatility_7d'] = data['close'].diff().abs().rolling(window=7, min_periods=5).sum()
    data['vol_adj_momentum_7d'] = data['momentum_7d'] / (data['volatility_7d'] + 1e-8)
    
    # Medium-Term (10-20 days)
    data['momentum_10d'] = data['close'] - data['close'].shift(10)
    data['volatility_10d'] = data['close'].diff().abs().rolling(window=10, min_periods=8).sum()
    data['vol_adj_momentum_10d'] = data['momentum_10d'] / (data['volatility_10d'] + 1e-8)
    
    data['momentum_20d'] = data['close'] - data['close'].shift(20)
    data['volatility_20d'] = data['close'].diff().abs().rolling(window=20, min_periods=15).sum()
    data['vol_adj_momentum_20d'] = data['momentum_20d'] / (data['volatility_20d'] + 1e-8)
    
    # Volatility-Adjusted Momentum Divergence
    data['momentum_divergence'] = np.abs(data['vol_adj_momentum_7d'] - data['vol_adj_momentum_20d'])
    data['sign_divergence'] = (np.sign(data['vol_adj_momentum_7d']) != np.sign(data['vol_adj_momentum_20d'])).astype(int)
    data['momentum_acceleration'] = data['vol_adj_momentum_3d'] - data['vol_adj_momentum_7d']
    
    # Volatility Regime Context
    vol_20d_avg = data['volatility_20d'].rolling(window=20, min_periods=15).mean()
    vol_percentile = data['volatility_20d'].rolling(window=60, min_periods=40).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) > 0 else 0, raw=False
    )
    data['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
    data['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
    
    # Volume-Amount Efficiency Analysis
    # Volume Efficiency Metrics
    data['daily_volume_eff'] = np.abs(data['close'].diff()) / (data['volume'] + 1e-8)
    data['avg_volume_eff_5d'] = data['daily_volume_eff'].rolling(window=5, min_periods=3).mean()
    data['volume_eff_trend'] = data['daily_volume_eff'] / (data['avg_volume_eff_5d'] + 1e-8)
    
    data['volume_momentum_5d'] = data['volume'] / (data['volume'].shift(5) + 1e-8)
    data['volume_momentum_quality'] = np.sign(data['volume_momentum_5d'] - 1) * np.sign(data['momentum_7d'])
    
    avg_volume_20d = data['volume'].rolling(window=20, min_periods=15).mean()
    data['high_volume_regime'] = (data['volume'] > 1.5 * avg_volume_20d).astype(int)
    data['low_volume_regime'] = (data['volume'] < 0.7 * avg_volume_20d).astype(int)
    
    # Amount Efficiency Framework
    data['daily_amount_eff'] = np.abs(data['close'].diff()) / (data['amount'] + 1e-8)
    data['avg_amount_eff_5d'] = data['daily_amount_eff'].rolling(window=5, min_periods=3).mean()
    data['amount_eff_trend'] = data['daily_amount_eff'] / (data['avg_amount_eff_5d'] + 1e-8)
    
    data['amount_momentum_5d'] = data['amount'] / (data['amount'].shift(5) + 1e-8)
    data['amount_volume_coherence'] = np.corrcoef(
        data['volume_eff_trend'].rolling(window=5).mean().fillna(0),
        data['amount_eff_trend'].rolling(window=5).mean().fillna(0)
    )[0, 1] if len(data) > 5 else 0
    
    # Volume-Amount Composite Efficiency
    data['composite_efficiency'] = (
        0.6 * data['volume_eff_trend'] + 0.4 * data['amount_eff_trend']
    ) * (1 + 0.2 * np.abs(data['amount_volume_coherence']))
    
    eff_percentile = data['composite_efficiency'].rolling(window=60, min_periods=40).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 75)) if len(x.dropna()) > 0 else 0, raw=False
    )
    data['high_efficiency_regime'] = (eff_percentile > 0.75).astype(int)
    data['low_efficiency_regime'] = (eff_percentile < 0.25).astype(int)
    
    # Intraday Range Quality Assessment
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_utilization'] = np.abs(data['close'] - data['open']) / (data['intraday_range'] + 1e-8)
    data['volume_concentration'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=3).mean() + 1e-8)
    data['intraday_efficiency'] = data['intraday_utilization'] * data['volume_concentration']
    
    # Dynamic Price Level Context
    data['high_10d'] = data['high'].rolling(window=10, min_periods=8).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=8).min()
    data['price_position'] = (data['close'] - data['low_10d']) / (data['high_10d'] - data['low_10d'] + 1e-8)
    
    # Adaptive Alpha Factor Construction
    # Core Momentum-Efficiency Signal
    primary_momentum = 0.6 * data['vol_adj_momentum_7d'] + 0.4 * data['vol_adj_momentum_20d']
    momentum_with_acceleration = primary_momentum * (1 + 0.3 * np.tanh(data['momentum_acceleration']))
    
    # Volatility regime adjustment
    vol_regime_adj = np.where(data['high_vol_regime'] == 1, 0.7, 
                             np.where(data['low_vol_regime'] == 1, 1.3, 1.0))
    
    # Efficiency confirmation
    efficiency_adj = momentum_with_acceleration * data['composite_efficiency']
    
    # Intraday adjustment
    intraday_adj = efficiency_adj * (1 + 0.2 * np.tanh(data['intraday_efficiency'] - 1))
    
    # Price level context
    level_adj = intraday_adj * (1 + 0.3 * np.where(
        (data['price_position'] > 0.8) | (data['price_position'] < 0.2), 1.2, 1.0
    ))
    
    # Signal quality assessment
    momentum_alignment = np.sign(data['vol_adj_momentum_7d']) == np.sign(data['vol_adj_momentum_20d'])
    efficiency_alignment = data['composite_efficiency'] > data['composite_efficiency'].rolling(window=10).mean()
    signal_quality = (momentum_alignment.astype(int) + efficiency_alignment.astype(int)) / 2
    
    # Final alpha factor
    alpha_factor = level_adj * vol_regime_adj * (0.8 + 0.4 * signal_quality)
    
    # Handle NaN values
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
