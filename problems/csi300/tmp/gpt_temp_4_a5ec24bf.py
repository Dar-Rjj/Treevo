import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Fractal Volatility Regime Efficiency with Volume-Price Alignment
    """
    data = df.copy()
    
    # Basic calculations
    data['prev_close'] = data['close'].shift(1)
    data['daily_range'] = data['high'] - data['low']
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['gap'] = abs(data['open'] - data['prev_close'])
    
    # 1. Fractal Volatility Regime Classification
    # Short-term volatility (5-day)
    data['range_efficiency'] = data['daily_range'] / abs(data['close'] - data['prev_close']).replace(0, np.nan)
    data['gap_absorption'] = data['gap'] / data['daily_range'].replace(0, np.nan)
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    data['vol_clustering'] = (data['daily_range'] > data['avg_range_5d']).rolling(window=5).sum()
    
    # Medium-term volatility (20-day)
    data['atr_20d'] = data['true_range'].rolling(window=20).mean()
    data['avg_daily_range_20d'] = data['daily_range'].rolling(window=20).mean()
    data['atr_efficiency_ratio'] = data['atr_20d'] / data['avg_daily_range_20d'].replace(0, np.nan)
    
    # Range autocorrelation (20-day)
    data['range_autocorr'] = data['daily_range'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    
    # Volatility asymmetry
    data['upper_range'] = data['high'] - data['close']
    data['lower_range'] = data['close'] - data['low']
    data['vol_asymmetry'] = (data['upper_range'] - data['lower_range']) / data['daily_range'].replace(0, np.nan)
    
    # Fractal regime classification
    data['short_term_vol'] = data['daily_range'].rolling(window=5).mean()
    data['medium_term_vol'] = data['daily_range'].rolling(window=20).mean()
    
    regime_conditions = [
        (data['short_term_vol'] < data['medium_term_vol']) & (data['range_efficiency'] > data['range_efficiency'].rolling(20).mean()),
        (data['short_term_vol'] > data['medium_term_vol']) & (data['range_efficiency'] < data['range_efficiency'].rolling(20).mean()),
        (abs(data['short_term_vol'].pct_change()) > 0.1) | (abs(data['medium_term_vol'].pct_change()) > 0.05)
    ]
    regime_choices = [2, 0, 1]  # 2: High efficiency, 0: Low efficiency, 1: Transition
    data['vol_regime'] = np.select(regime_conditions, regime_choices, default=1)
    
    # 2. Multi-Scale Efficiency Momentum
    # Multi-timeframe returns
    for period in [3, 5, 10]:
        data[f'return_{period}d'] = data['close'].pct_change(period)
    
    # Momentum acceleration
    data['momentum_accel_3_5'] = data['return_5d'] - data['return_3d']
    data['momentum_accel_5_10'] = data['return_10d'] - data['return_5d']
    
    # Momentum direction consistency
    data['momentum_consistency'] = (
        (np.sign(data['return_3d']) == np.sign(data['return_5d'])).astype(int) +
        (np.sign(data['return_5d']) == np.sign(data['return_10d'])).astype(int)
    )
    
    # Daily price efficiency
    data['daily_efficiency'] = abs(data['close'] - data['open']) / data['daily_range'].replace(0, np.nan)
    data['efficiency_momentum'] = data['daily_efficiency'] / data['daily_efficiency'].rolling(window=5).mean()
    
    # Regime-adaptive momentum
    high_eff_mask = data['vol_regime'] == 2
    low_eff_mask = data['vol_regime'] == 0
    trans_mask = data['vol_regime'] == 1
    
    data['regime_momentum'] = 0
    data.loc[high_eff_mask, 'regime_momentum'] = data.loc[high_eff_mask, 'return_5d'] * data.loc[high_eff_mask, 'momentum_consistency']
    data.loc[low_eff_mask, 'regime_momentum'] = data.loc[low_eff_mask, 'return_3d'] / data.loc[low_eff_mask, 'short_term_vol'].replace(0, np.nan)
    data.loc[trans_mask, 'regime_momentum'] = data.loc[trans_mask, 'return_3d'] * data.loc[trans_mask, 'efficiency_momentum']
    
    # 3. Volume Fractal Confirmation Analysis
    # Volume concentration
    data['volume_20d_median'] = data['volume'].rolling(window=20).median()
    data['high_volume_day'] = (data['volume'] > data['volume_20d_median']).astype(int)
    data['volume_clustering'] = data['high_volume_day'].rolling(window=5).sum()
    
    # Volume-volatility correlation
    data['volume_range_corr'] = data['volume'].rolling(window=10).corr(data['daily_range'])
    
    # Volume efficiency metrics
    data['volume_stability'] = data['volume'].rolling(window=10).mean() / data['volume'].rolling(window=10).std()
    data['volume_momentum'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['volume_efficiency'] = data['amount'] / (data['volume'] * data['true_range']).replace(0, np.nan)
    
    # Volume-price alignment
    data['volume_ma_3d'] = data['volume'].rolling(window=3).mean()
    data['volume_trend'] = data['volume_ma_3d'].diff(2)  # 3-day slope
    data['return_volume_alignment'] = (np.sign(data['return_3d']) == np.sign(data['volume_trend'])).astype(int)
    
    # Volume-efficiency momentum
    data['volume_efficiency_momentum'] = data['volume_efficiency'].pct_change(3)
    
    # 4. Fractal Divergence Detection
    # Multi-timeframe efficiency divergence
    data['efficiency_divergence'] = (
        data['daily_efficiency'].rolling(window=5).mean() - 
        data['daily_efficiency'].rolling(window=10).mean()
    )
    
    # Volume-efficiency asymmetry
    data['volume_efficiency_asymmetry'] = (
        data['volume_efficiency'].rolling(window=5).mean() - 
        data['volume_efficiency'].rolling(window=10).mean()
    ) * data['vol_asymmetry']
    
    # 5. Adaptive Signal Construction
    # Regime-appropriate signals
    data['fractal_signal'] = 0
    data.loc[high_eff_mask, 'fractal_signal'] = (
        data.loc[high_eff_mask, 'regime_momentum'] * 
        data.loc[high_eff_mask, 'return_volume_alignment'] * 
        data.loc[high_eff_mask, 'volume_efficiency_momentum']
    )
    data.loc[low_eff_mask, 'fractal_signal'] = (
        data.loc[low_eff_mask, 'regime_momentum'] / 
        data.loc[low_eff_mask, 'short_term_vol'].replace(0, np.nan) * 
        data.loc[low_eff_mask, 'efficiency_divergence']
    )
    data.loc[trans_mask, 'fractal_signal'] = (
        data.loc[trans_mask, 'regime_momentum'] * 
        data.loc[trans_mask, 'volume_efficiency_asymmetry'] * 
        data.loc[trans_mask, 'volume_clustering']
    )
    
    # Volume confirmation filters
    data['volume_confirmation'] = (
        data['volume_efficiency_momentum'] * 
        data['volume_clustering'].pct_change(3) * 
        data['return_volume_alignment']
    )
    
    # Final composite factor
    data['composite_factor'] = (
        data['fractal_signal'] * 0.6 + 
        data['volume_confirmation'] * 0.4
    ) * data['volume_stability'].replace([np.inf, -np.inf], np.nan)
    
    # Normalize and handle outliers
    factor = data['composite_factor'].replace([np.inf, -np.inf], np.nan)
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor
