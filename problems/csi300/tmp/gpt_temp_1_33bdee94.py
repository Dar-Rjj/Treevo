import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Price-Volume Convergence with Fractal Market Structure
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Fractal Market Structure Component
    # Multi-Timeframe Volatility Structure
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    data['short_term_vol'] = data['returns'].rolling(window=3, min_periods=2).std()
    data['medium_term_vol'] = data['returns'].rolling(window=10, min_periods=5).std()
    
    # Volatility scaling factors
    data['vol_ratio'] = data['intraday_vol'] / data['short_term_vol'].replace(0, np.nan)
    data['vol_persistence'] = data['short_term_vol'] / data['medium_term_vol'].replace(0, np.nan)
    
    # Price-Volume Fractal Dimension
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['price_complexity'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=3, min_periods=2).sum()
    
    # Volume persistence (correlation of recent volumes)
    def volume_correlation(x):
        if len(x) < 6:
            return np.nan
        recent = x[-3:]  # t-2 to t
        previous = x[-6:-3]  # t-5 to t-3
        return np.corrcoef(recent, previous)[0, 1] if len(recent) == 3 and len(previous) == 3 else np.nan
    
    data['volume_persistence'] = data['volume'].rolling(window=6, min_periods=6).apply(volume_correlation, raw=True)
    
    # Combined fractal scores
    data['fractal_score_1'] = data['price_efficiency'] * data['volume_concentration']
    data['fractal_score_2'] = data['price_complexity'] * data['volume_persistence']
    
    # Market Microstructure Patterns
    data['avg_range_5'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    data['vol_contraction'] = (data['high'] - data['low']) / data['avg_range_5'].replace(0, np.nan)
    data['volume_compression'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Structural break signals
    data['breakout'] = ((data['close'] > data['high'].shift(1)) & 
                       (data['volume'] > data['volume'].shift(1))).astype(float)
    data['breakdown'] = ((data['close'] < data['low'].shift(1)) & 
                        (data['volume'] > data['volume'].shift(1))).astype(float)
    
    # 2. Smart Convergence Component
    # Volatility-Weighted Price Signals
    data['raw_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['vol_scaled_momentum'] = data['raw_momentum'] / data['short_term_vol'].replace(0, np.nan)
    
    data['trend_slope'] = data['close'] / data['close'].shift(10) - 1
    data['efficiency_trend'] = data['trend_slope'] * data['price_efficiency']
    
    # Volume-Intelligence Signals
    data['volume_burst'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    data['sustained_volume'] = ((data['volume'] > data['volume'].rolling(window=5, min_periods=3).mean()) & 
                               (data['volume'].shift(1) > data['volume'].rolling(window=5, min_periods=3).mean().shift(1))).astype(float)
    
    # Volume-price correlation
    def volume_price_corr(x_vol, x_ret):
        if len(x_vol) < 4 or len(x_ret) < 4:
            return np.nan
        return np.corrcoef(x_vol[-4:], x_ret[-4:])[0, 1]
    
    data['volume_price_corr'] = data['volume'].rolling(window=4, min_periods=4).apply(
        lambda x: volume_price_corr(x, data['returns'].iloc[len(data)-len(x):len(data)]), raw=False)
    
    data['volume_clustering'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)).replace(0, np.nan)
    
    # Convergence Strength
    data['signal_alignment'] = np.sign(data['raw_momentum']) * np.sign(data['volume_burst'] - 1)
    
    # 3. Adaptive Regime Mapping
    # Market Regime Types
    data['ma_20'] = data['close'].rolling(window=20, min_periods=10).mean()
    data['trend_slope_20'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Volatility regimes
    data['high_vol_regime'] = (data['short_term_vol'] > 2 * data['medium_term_vol']).astype(float)
    data['low_vol_regime'] = (data['short_term_vol'] < 0.5 * data['medium_term_vol']).astype(float)
    data['normal_vol_regime'] = ((~data['high_vol_regime'].astype(bool)) & 
                                (~data['low_vol_regime'].astype(bool))).astype(float)
    
    # Structural regimes
    low_range_threshold = data['avg_range_5'].quantile(0.3)
    high_range_threshold = data['avg_range_5'].quantile(0.7)
    low_volume_threshold = data['volume'].rolling(window=20, min_periods=10).mean().quantile(0.3)
    high_volume_threshold = data['volume'].rolling(window=20, min_periods=10).mean().quantile(0.7)
    
    data['compressed_market'] = ((data['avg_range_5'] < low_range_threshold) & 
                               (data['volume'] < low_volume_threshold)).astype(float)
    data['expanding_market'] = ((data['avg_range_5'] > high_range_threshold) & 
                              (data['volume'] > high_volume_threshold)).astype(float)
    data['transitional_market'] = ((~data['compressed_market'].astype(bool)) & 
                                 (~data['expanding_market'].astype(bool))).astype(float)
    
    # 4. Final Alpha Factor Integration
    # Dynamic parameters based on regime
    data['adaptive_lookback'] = np.where(data['high_vol_regime'] == 1, 5, 
                                       np.where(data['low_vol_regime'] == 1, 15, 10))
    
    # Signal sensitivity adjustment
    data['sensitivity_factor'] = np.where(data['high_vol_regime'] == 1, 0.7,
                                        np.where(data['low_vol_regime'] == 1, 1.3, 1.0))
    
    # Multi-scale signal fusion
    # Fractal structure component (weighted average)
    data['fractal_component'] = (
        0.4 * data['fractal_score_1'].fillna(0) +
        0.3 * data['fractal_score_2'].fillna(0) +
        0.2 * (data['breakout'] - data['breakdown']) +
        0.1 * (1 - data['vol_contraction'])
    )
    
    # Convergence component
    data['convergence_component'] = (
        0.5 * data['vol_scaled_momentum'].fillna(0) +
        0.3 * data['efficiency_trend'].fillna(0) +
        0.2 * data['signal_alignment'].fillna(0)
    )
    
    # Regime context component
    data['regime_component'] = (
        0.4 * data['trend_slope_20'].fillna(0) +
        0.3 * (data['expanding_market'] - data['compressed_market']) +
        0.3 * (data['normal_vol_regime'] - 0.5 * data['high_vol_regime'] - 0.5 * data['low_vol_regime'])
    )
    
    # Final alpha factor with regime adaptation
    data['alpha_factor'] = (
        data['sensitivity_factor'] * 
        (0.4 * data['fractal_component'] + 
         0.4 * data['convergence_component'] + 
         0.2 * data['regime_component'])
    )
    
    # Volatility scaling for final output
    volatility_scale = 1 / data['medium_term_vol'].replace(0, np.nan)
    data['final_alpha'] = data['alpha_factor'] * volatility_scale
    
    # Clean up and return
    result = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
