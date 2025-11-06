import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Price-Volume Divergence Spectrum factor
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate Multi-Timeframe Price Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(4) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(9) - 1
    data['momentum_gradient'] = data['momentum_5d'] - data['momentum_10d']
    
    # Calculate Volume Momentum Patterns
    data['turnover'] = data['volume'] * data['close']
    data['turnover_5d_avg'] = data['turnover'].rolling(window=5).mean()
    data['turnover_15d_avg'] = data['turnover'].rolling(window=15).mean()
    data['turnover_ratio'] = data['turnover_5d_avg'] / data['turnover_15d_avg'] - 1
    
    # Measure Volume Distribution Asymmetry
    # Up-day identification and volume concentration
    data['price_change'] = data['close'] / data['close'].shift(1) - 1
    data['is_up_day'] = (data['price_change'] > 0).astype(int)
    
    # Rolling calculation for up-day volume concentration
    def calc_up_volume_ratio(window):
        up_days = window['is_up_day'] == 1
        if up_days.sum() == 0:
            return 0
        up_volume_avg = window.loc[up_days, 'volume'].mean()
        total_volume_avg = window['volume'].mean()
        return up_volume_avg / total_volume_avg if total_volume_avg > 0 else 0
    
    # Apply rolling calculation
    up_volume_ratios = []
    for i in range(len(data)):
        if i < 9:
            up_volume_ratios.append(0)
            continue
        window_data = data.iloc[i-9:i+1][['is_up_day', 'volume']].copy()
        ratio = calc_up_volume_ratio(window_data)
        up_volume_ratios.append(ratio)
    
    data['up_volume_ratio'] = up_volume_ratios
    
    # Volume clustering intensity - consecutive high volume periods
    data['volume_rank'] = data['volume'].rolling(window=20).rank(pct=True)
    data['high_volume'] = (data['volume_rank'] > 0.7).astype(int)
    
    # Volume persistence (consecutive high volume days)
    data['volume_persistence'] = 0
    persistence_count = 0
    for i in range(len(data)):
        if data['high_volume'].iloc[i] == 1:
            persistence_count += 1
        else:
            persistence_count = 0
        data.loc[data.index[i], 'volume_persistence'] = persistence_count
    
    # Quantify Price Path Efficiency
    data['overnight_gap'] = abs(data['open'] / data['close'].shift(1) - 1)
    data['intraday_range'] = abs(data['high'] / data['low'] - 1)
    
    # Calculate actual price path length (5-day window)
    data['actual_path'] = 0
    for i in range(len(data)):
        if i < 4:
            continue
        path_sum = 0
        for j in range(i-4, i+1):
            if j == i-4:
                # First element uses overnight gap
                path_sum += data['overnight_gap'].iloc[j] if not pd.isna(data['overnight_gap'].iloc[j]) else 0
            else:
                # Subsequent elements use intraday range
                path_sum += data['intraday_range'].iloc[j] if not pd.isna(data['intraday_range'].iloc[j]) else 0
        data.loc[data.index[i], 'actual_path'] = path_sum
    
    # Straight-line price movement (5-day)
    data['straight_line'] = abs(data['close'] / data['close'].shift(4) - 1)
    data['efficiency_ratio'] = data['straight_line'] / (data['actual_path'] + 1e-8)
    
    # Detect Divergence Regime Classification
    # Strength divergence: momentum vs volume magnitude
    data['strength_divergence'] = data['momentum_gradient'] - data['turnover_ratio']
    
    # Timing divergence: correlation between volume peaks and price changes
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['price_volume_corr'] = data['price_change'].rolling(window=10).corr(data['volume_change'])
    
    # Duration divergence: momentum persistence vs volume persistence
    momentum_persistence = []
    persistence_count = 0
    for i in range(len(data)):
        if i == 0:
            momentum_persistence.append(0)
            continue
        if data['momentum_gradient'].iloc[i] * data['momentum_gradient'].iloc[i-1] > 0:
            persistence_count += 1
        else:
            persistence_count = 0
        momentum_persistence.append(persistence_count)
    
    data['momentum_persistence'] = momentum_persistence
    data['duration_divergence'] = data['momentum_persistence'] - data['volume_persistence']
    
    # Apply Volatility Spectrum Adjustment
    # True Range calculation
    data['tr1'] = abs(data['high'] - data['low'])
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_20d'] = data['true_range'].rolling(window=20).mean()
    
    # High-Low Range Spectrum
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    data['range_ratio'] = data['high_20d'] / data['low_20d'] - 1
    
    # Generate Predictive Alpha Factors
    # Early Warning Divergence Signals
    data['momentum_exhaustion'] = (
        (data['momentum_gradient'].abs() > data['momentum_gradient'].rolling(window=20).std()) & 
        (data['turnover_ratio'] < 0) & 
        (data['up_volume_ratio'] < 0.5)
    ).astype(int)
    
    # Accumulation/Distribution patterns
    data['accumulation_pattern'] = (
        (data['momentum_gradient'] > 0) & 
        (data['up_volume_ratio'] > 0.6) & 
        (data['efficiency_ratio'] > 0.5)
    ).astype(int)
    
    data['distribution_pattern'] = (
        (data['momentum_gradient'] < 0) & 
        (data['up_volume_ratio'] < 0.4) & 
        (data['efficiency_ratio'] < 0.3)
    ).astype(int)
    
    # Volatility-Weighted Convergence Factors
    volatility_weight = 1 / (data['atr_20d'] / data['close'] + 1e-8)
    
    # Trend strength validation
    data['trend_strength'] = (
        data['momentum_gradient'].abs() * 
        data['up_volume_ratio'] * 
        volatility_weight
    )
    
    # Breakout authenticity in low volatility
    low_vol_regime = (data['atr_20d'] / data['close'] < data['atr_20d'].rolling(window=60).quantile(0.3))
    data['breakout_authenticity'] = (
        (data['momentum_5d'] > data['momentum_5d'].rolling(window=20).quantile(0.8)) & 
        low_vol_regime & 
        (data['volume'] > data['volume'].rolling(window=20).mean() * 1.5)
    ).astype(int)
    
    # Multi-Timeframe Divergence Analysis - Final Alpha Factor
    alpha_factor = (
        # Short-term noise filtering with volatility adjustment
        data['strength_divergence'] * volatility_weight * 
        # Medium-term trend context with volume persistence
        np.sign(data['momentum_gradient']) * data['volume_persistence'] * 
        # Long-term regime awareness with range spectrum
        (1 / (data['range_ratio'] + 1e-8)) * 
        # Early warning signals
        (1 - 2 * data['momentum_exhaustion']) * 
        # Accumulation/distribution patterns
        (1 + data['accumulation_pattern'] - data['distribution_pattern']) * 
        # Trend sustainability
        data['trend_strength'] * 
        # Breakout validation
        (1 + data['breakout_authenticity'])
    )
    
    # Clean up and return
    result = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
