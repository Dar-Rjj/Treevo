import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Multi-Timeframe Price Impact & Momentum Structure
    # Price Impact Asymmetry Analysis
    data['opening_impact'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_impact'] = (data['close'] - data['open']) / data['open']
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['closing_impact'] = (data['close'] - data['mid_price']) / data['mid_price']
    data['impact_asymmetry'] = data['intraday_impact'] / data['opening_impact'].replace(0, np.nan)
    
    # Momentum Persistence Across Timeframes
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(8) - 1
    momentum_diff_1 = data['ultra_short_momentum'] - data['short_term_momentum']
    momentum_diff_2 = data['short_term_momentum'] - data['medium_term_momentum']
    data['momentum_gradient'] = momentum_diff_1 / momentum_diff_2.replace(0, np.nan)
    
    # Microstructure Efficiency Patterns
    data['price_range'] = data['high'] - data['low']
    data['opening_efficiency'] = abs(data['open'] - data['close'].shift(1)) / data['price_range'].replace(0, np.nan)
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / data['price_range'].replace(0, np.nan)
    data['closing_efficiency'] = abs(data['close'] - data['mid_price']) / data['price_range'].replace(0, np.nan)
    data['efficiency_divergence'] = data['intraday_efficiency'] / data['opening_efficiency'].replace(0, np.nan)
    
    # Volume Microstructure Impact & Distribution
    # Volume Impact Asymmetry
    data['opening_volume_impact'] = data['volume'] * (abs(data['opening_impact']) > 0)
    data['intraday_volume_impact'] = data['volume'] * (abs(data['intraday_impact']) > 0)
    data['volume_impact_ratio'] = data['intraday_volume_impact'] / data['opening_volume_impact'].replace(0, np.nan)
    data['impact_volume_asymmetry'] = data['impact_asymmetry'] * data['volume_impact_ratio']
    
    # Trade Size Distribution Dynamics
    data['current_trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['prev_trade_size'] = data['amount'].shift(2) / data['volume'].shift(2).replace(0, np.nan)
    data['trade_size_momentum'] = data['current_trade_size'] / data['prev_trade_size'].replace(0, np.nan)
    
    # 5-day trade size volatility
    trade_size_series = data['amount'] / data['volume'].replace(0, np.nan)
    data['trade_size_volatility'] = trade_size_series.rolling(window=5, min_periods=3).std()
    data['trade_size_median'] = trade_size_series.rolling(window=3, min_periods=2).median()
    data['size_distribution_skew'] = (data['current_trade_size'] - data['trade_size_median']) / data['trade_size_volatility'].replace(0, np.nan)
    
    # Volume Concentration Patterns
    data['volume_clustering'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    data['volume_rolling_mean'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_concentration'] = data['volume'] / data['volume_rolling_mean'].replace(0, np.nan)
    data['concentration_momentum'] = data['volume_concentration'] * data['volume_clustering']
    
    # Volume persistence (3-day correlation)
    def volume_correlation(x):
        if len(x) < 3:
            return np.nan
        return pd.Series(x).corr(pd.Series(x).shift(1))
    
    data['volume_persistence'] = data['volume'].rolling(window=3, min_periods=3).apply(volume_correlation, raw=False)
    
    # Microstructure Regime Detection & Classification
    # Volatility Structure Analysis
    data['current_volatility'] = data['high'] - data['low']
    data['prev_volatility'] = data['high'].shift(2) - data['low'].shift(2)
    data['volatility_momentum'] = data['current_volatility'] / data['prev_volatility'].replace(0, np.nan)
    
    # 8-day volatility persistence
    def volatility_correlation(x):
        if len(x) < 8:
            return np.nan
        return pd.Series(x).corr(pd.Series(x).shift(1))
    
    volatility_series = data['high'] - data['low']
    data['volatility_persistence'] = volatility_series.rolling(window=8, min_periods=8).apply(volatility_correlation, raw=False)
    data['volatility_regime'] = data['volatility_momentum'] * data['volatility_persistence']
    
    # Price-Microstructure Alignment
    data['price_change_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['volume_change_sign'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['price_volume_alignment'] = data['price_change_sign'] * data['volume_change_sign']
    data['impact_efficiency_alignment'] = data['impact_asymmetry'] * data['efficiency_divergence']
    data['microstructure_coherence'] = data['price_volume_alignment'] * data['impact_efficiency_alignment']
    data['alignment_strength'] = abs(data['microstructure_coherence'])
    
    # Regime Classification Framework
    high_momentum_mask = (data['volatility_regime'] > 1.2) & (data['alignment_strength'] > 0.8)
    low_momentum_mask = (data['volatility_regime'] < 0.8) & (data['alignment_strength'] < 0.3)
    transition_mask = ~high_momentum_mask & ~low_momentum_mask
    
    # Multi-Scale Signal Generation & Integration
    # Momentum-Impact Signal Components
    data['ultra_short_signal'] = data['ultra_short_momentum'] * data['impact_asymmetry']
    data['short_term_signal'] = data['short_term_momentum'] * data['volume_impact_ratio']
    data['medium_term_signal'] = data['medium_term_momentum'] * data['size_distribution_skew']
    data['signal_convergence'] = data['ultra_short_signal'] * data['short_term_signal'] * data['medium_term_signal']
    
    # Volume-Microstructure Enhancement
    data['volume_momentum_enhancement'] = data['volume_clustering'] * data['volume_persistence']
    data['impact_concentration'] = data['impact_volume_asymmetry'] * data['volume_concentration']
    data['microstructure_quality'] = data['efficiency_divergence'] * data['alignment_strength']
    data['volume_microstructure_factor'] = data['volume_momentum_enhancement'] * data['impact_concentration'] * data['microstructure_quality']
    
    # Regime-Adaptive Signal Weighting
    high_momentum_signal = data['signal_convergence'] * data['volume_microstructure_factor']
    low_momentum_signal = data['medium_term_signal'] * data['volume_concentration']
    transition_signal = (data['ultra_short_signal'] + data['short_term_signal']) * data['impact_concentration']
    
    # Final alpha factor with regime weighting
    alpha_factor = pd.Series(index=data.index, dtype=float)
    alpha_factor[high_momentum_mask] = high_momentum_signal[high_momentum_mask]
    alpha_factor[low_momentum_mask] = low_momentum_signal[low_momentum_mask]
    alpha_factor[transition_mask] = transition_signal[transition_mask]
    
    # Pattern Recognition & Signal Validation
    # Microstructure Pattern Detection
    data['momentum_acceleration'] = data['momentum_gradient'] * data['impact_asymmetry']
    data['volume_confirmation'] = data['volume_impact_ratio'] * data['volume_concentration']
    data['efficiency_validation'] = data['efficiency_divergence'] * data['microstructure_coherence']
    data['pattern_strength'] = data['momentum_acceleration'] * data['volume_confirmation'] * data['efficiency_validation']
    
    # Signal Quality Assessment
    ultra_short_rolling = data['ultra_short_signal'].rolling(window=5, min_periods=3)
    short_term_rolling = data['short_term_signal'].rolling(window=5, min_periods=3)
    
    def rolling_corr(x, y):
        if len(x) < 3:
            return np.nan
        return pd.Series(x).corr(pd.Series(y))
    
    data['signal_consistency'] = data['ultra_short_signal'].rolling(window=5, min_periods=3).apply(
        lambda x: rolling_corr(x, data['short_term_signal'].loc[x.index]), raw=False
    )
    
    data['volume_support'] = data['volume_microstructure_factor'] * data['pattern_strength']
    data['microstructure_alignment_score'] = data['alignment_strength'] * data['pattern_strength']
    data['quality_score'] = data['signal_consistency'] * data['volume_support'] * data['microstructure_alignment_score']
    
    # Apply quality adjustment to final alpha
    quality_adjusted_alpha = alpha_factor * data['quality_score']
    
    # Clean and return the final alpha factor
    final_alpha = quality_adjusted_alpha.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return final_alpha
