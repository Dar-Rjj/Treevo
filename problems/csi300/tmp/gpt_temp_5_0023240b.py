import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Regime Alpha Factor
    Combines volatility regime classification with momentum, efficiency, volume, and correlation signals
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Volatility-Regime Momentum Persistence
    # Multi-timeframe momentum calculation
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_8d'] = data['close'].pct_change(8)
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_8d']
    
    # Volatility-regime classification
    # Calculate True Range
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['vol_3d'] = data['tr'].rolling(3).mean() / data['close']
    data['vol_8d'] = data['tr'].rolling(8).mean() / data['close']
    data['vol_regime'] = data['vol_3d'] / data['vol_8d']
    
    # Range Efficiency Volume Alignment
    # Multi-period range efficiency
    data['daily_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency_3d_avg'] = data['daily_efficiency'].rolling(3).mean()
    data['efficiency_trend'] = data['daily_efficiency'] / data['efficiency_3d_avg']
    
    # Count high efficiency days (efficiency > 0.7)
    data['high_eff_count'] = (data['daily_efficiency'] > 0.7).rolling(3).sum()
    
    # Volume-efficiency relationship
    data['high_eff_volume'] = np.where(data['daily_efficiency'] > 0.7, data['volume'], 0)
    data['eff_vol_corr'] = data['daily_efficiency'].rolling(3).corr(data['volume'])
    data['volume_trend'] = data['volume'] / data['volume'].rolling(3).mean()
    data['eff_vol_momentum'] = data['efficiency_trend'] * data['volume_trend']
    
    # Extreme Return Volume Regime
    # Volatility-adjusted extreme detection
    data['rolling_vol_5d'] = data['close'].pct_change().rolling(5).std()
    data['extreme_threshold'] = 1.5 * data['rolling_vol_5d']
    data['extreme_return'] = abs(data['close'].pct_change()) > data['extreme_threshold']
    
    # Volume regime analysis
    data['volume_percentile'] = data['volume'].rolling(5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Volume persistence (consecutive extreme volume days)
    data['high_volume'] = data['volume_percentile'] > 0.8
    data['vol_persistence'] = data['high_volume'].rolling(3).apply(
        lambda x: x.sum() if x.all() else 0
    )
    
    data['vol_return_alignment'] = np.where(
        (data['close'].pct_change() > 0) == (data['volume'] > data['volume'].rolling(5).mean()),
        1, -1
    )
    
    # Amount Flow Direction Persistence
    # Multi-timeframe directional flow
    data['daily_flow'] = np.sign(data['close'] - data['close'].shift(1)) * data['amount']
    data['flow_3d_avg'] = data['daily_flow'].rolling(3).mean()
    data['flow_momentum'] = data['daily_flow'] / data['flow_3d_avg']
    
    # Flow persistence (consecutive same-direction flow days)
    data['flow_direction'] = np.sign(data['daily_flow'])
    data['flow_persistence'] = data['flow_direction'].rolling(3).apply(
        lambda x: 3 if len(set(x)) == 1 else 0
    )
    
    # Flow-volatility relationship
    data['flow_efficiency'] = data['daily_flow'] / (data['high'] - data['low'])
    data['flow_acceleration'] = data['daily_flow'].diff(3)
    
    # Opening Gap Volatility Regime
    # Gap characteristics
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_persistence'] = np.sign(data['overnight_gap']).rolling(3).apply(
        lambda x: 3 if len(set(x)) == 1 else 0
    )
    data['gap_to_vol'] = abs(data['overnight_gap']) / data['rolling_vol_5d']
    
    # Intraday gap behavior
    data['gap_fill_efficiency'] = abs(data['close'] - data['open']) / abs(data['overnight_gap'])
    data['gap_volume_intensity'] = np.where(
        abs(data['overnight_gap']) > data['rolling_vol_5d'],
        data['volume'] / data['volume'].rolling(5).mean(),
        0
    )
    
    # Price-Volume Correlation Persistence
    # Dynamic correlation regimes
    data['price_vol_corr'] = data['close'].pct_change().rolling(5).corr(data['volume'].pct_change())
    data['corr_persistence'] = np.sign(data['price_vol_corr']).rolling(3).apply(
        lambda x: 3 if len(set(x)) == 1 else 0
    )
    data['corr_strength'] = abs(data['price_vol_corr'])
    
    # Regime-specific alpha signals
    data['high_corr_momentum'] = data['momentum_3d'] * data['corr_persistence']
    data['low_corr_mean_rev'] = -data['momentum_3d'] * (1 - data['corr_strength'])
    data['corr_vol_interaction'] = data['price_vol_corr'] * data['vol_regime']
    
    # Combine all components into final alpha factor
    # Weight components based on volatility regime
    high_vol_regime = data['vol_regime'] > 1.2
    low_vol_regime = data['vol_regime'] < 0.8
    
    # High volatility regime: emphasize momentum and correlation signals
    high_vol_alpha = (
        0.3 * data['momentum_divergence'] +
        0.2 * data['high_corr_momentum'] +
        0.2 * data['eff_vol_momentum'] +
        0.15 * data['flow_momentum'] +
        0.15 * data['gap_to_vol']
    )
    
    # Low volatility regime: emphasize mean reversion and efficiency
    low_vol_alpha = (
        0.3 * data['low_corr_mean_rev'] +
        0.25 * data['efficiency_trend'] +
        0.2 * data['vol_return_alignment'] +
        0.15 * data['flow_efficiency'] +
        0.1 * data['gap_fill_efficiency']
    )
    
    # Normal volatility regime: balanced approach
    normal_vol_alpha = (
        0.2 * data['momentum_divergence'] +
        0.2 * data['eff_vol_momentum'] +
        0.15 * data['flow_persistence'] +
        0.15 * data['vol_persistence'] +
        0.15 * data['corr_vol_interaction'] +
        0.15 * data['gap_persistence']
    )
    
    # Combine based on volatility regime
    alpha = np.where(
        high_vol_regime, high_vol_alpha,
        np.where(low_vol_regime, low_vol_alpha, normal_vol_alpha)
    )
    
    # Final normalization and cleaning
    alpha_series = pd.Series(alpha, index=data.index)
    alpha_series = (alpha_series - alpha_series.rolling(20).mean()) / alpha_series.rolling(20).std()
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return alpha_series
