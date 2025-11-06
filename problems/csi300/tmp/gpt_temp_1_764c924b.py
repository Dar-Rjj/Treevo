import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Momentum Asymmetry Factor that combines directional momentum strength
    with volume-weighted price efficiency and market microstructure signals.
    """
    data = df.copy()
    
    # 1. Directional Momentum Strength Analysis
    # Intraday Momentum Patterns
    data['bullish_intensity'] = np.where(data['close'] > data['open'],
                                        (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001),
                                        0)
    data['bearish_intensity'] = np.where(data['close'] < data['open'],
                                        (data['open'] - data['close']) / (data['high'] - data['low'] + 0.001),
                                        0)
    data['neutral_compression'] = (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 0.001)
    
    # Multi-period Momentum Persistence
    data['close_ret_1'] = data['close'].pct_change(1)
    data['close_ret_2'] = data['close'].pct_change(2)
    data['close_ret_3'] = data['close'].pct_change(3)
    data['close_ret_5'] = data['close'].pct_change(5)
    
    # 3-day Directional Consistency
    data['sign_1'] = np.sign(data['close'] - data['close'].shift(1))
    data['sign_2'] = np.sign(data['close'].shift(1) - data['close'].shift(2))
    data['sign_3'] = np.sign(data['close'].shift(2) - data['close'].shift(3))
    data['directional_consistency'] = data['sign_1'] * data['sign_2'] * data['sign_3']
    
    # 5-day Momentum Amplitude
    data['range_5day'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).sum()
    data['momentum_amplitude'] = (data['close'] - data['close'].shift(5)) / (data['range_5day'] + 0.001)
    
    # Momentum Regime Stability
    data['momentum_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['momentum_stability'] = data['momentum_sign'].rolling(window=5, min_periods=3).apply(
        lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else np.nan
    )
    
    # Momentum Asymmetry Detection
    data['bull_bear_ratio'] = data['bullish_intensity'] / (data['bearish_intensity'] + 0.001)
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['multi_period_momentum'] = data['close_ret_5']
    data['momentum_divergence'] = data['intraday_momentum'] - data['multi_period_momentum']
    
    # 2. Volume-Weighted Price Efficiency
    # Volume-Price Alignment Patterns
    data['volume_ratio'] = data['volume'] / (data['volume'].shift(1) + 0.001)
    data['confirmed_up_moves'] = np.where(data['close'] > data['open'],
                                         (data['close'] - data['open']) * data['volume_ratio'],
                                         0)
    data['confirmed_down_moves'] = np.where(data['close'] < data['open'],
                                           (data['open'] - data['close']) * data['volume_ratio'],
                                           0)
    data['volume_price_divergence'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Multi-day Volume Efficiency
    data['avg_volume_3day'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['range_efficiency'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)) * \
                              (data['volume'] / (data['avg_volume_3day'] + 0.001))
    
    # Volume Distribution Efficiency
    data['high_volume_efficiency'] = np.where(data['volume'] > data['volume'].shift(1),
                                             (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001),
                                             0)
    data['low_volume_inefficiency'] = np.where(data['volume'] < data['volume'].shift(1),
                                              (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 0.001),
                                              0)
    
    # 3. Market Microstructure Integration
    # Trade Flow Quality Assessment
    data['amount_efficiency'] = data['amount'] / (data['volume'] * data['close'] + 0.001)
    
    # Daily range threshold (median of past 20 days)
    data['daily_range'] = data['high'] - data['low']
    data['range_threshold'] = data['daily_range'].rolling(window=20, min_periods=10).median()
    data['large_trade_impact'] = np.where(data['daily_range'] > data['range_threshold'],
                                         data['amount'] / (data['amount'].rolling(window=5, min_periods=3).sum() + 0.001),
                                         0)
    
    # 4. Asymmetry Signal Generation
    # Base Asymmetry Score
    data['momentum_asymmetry'] = data['bull_bear_ratio'] * data['momentum_divergence']
    data['volume_efficiency'] = (data['high_volume_efficiency'] - data['low_volume_inefficiency']) * data['range_efficiency']
    
    # Pattern-based Multiplier Matrix
    # Pattern A: Strong Bullish Momentum + High Volume Confirmation
    pattern_a = np.where((data['bullish_intensity'] > data['bullish_intensity'].rolling(window=10).quantile(0.7)) & 
                        (data['volume_ratio'] > 1.2), 1.5, 1.0)
    
    # Pattern B: Strong Bearish Momentum + High Volume Confirmation
    pattern_b = np.where((data['bearish_intensity'] > data['bearish_intensity'].rolling(window=10).quantile(0.7)) & 
                        (data['volume_ratio'] > 1.2), -1.2, 1.0)
    
    # Pattern C: Weak Momentum + Volume Divergence
    pattern_c = np.where((abs(data['intraday_momentum']) < 0.1) & 
                        (data['volume_price_divergence'] < 0), -1.3, 1.0)
    
    # Pattern D: Neutral Compression + Low Volume
    pattern_d = np.where((data['neutral_compression'] > 2.0) & 
                        (data['volume_ratio'] < 0.8), 1.4, 1.0)
    
    # Combine patterns
    data['pattern_multiplier'] = pattern_a * pattern_b * pattern_c * pattern_d
    
    # Regime-dependent Weighting Scheme
    volatility_regime = data['daily_range'].rolling(window=20, min_periods=10).std()
    regime_weight = np.where(volatility_regime > volatility_regime.rolling(window=50).median(), 0.7, 1.3)
    
    # 5. Final Factor Construction
    data['base_asymmetry_score'] = data['momentum_asymmetry'] * data['volume_efficiency']
    data['adjusted_score'] = data['base_asymmetry_score'] * data['pattern_multiplier'] * regime_weight
    
    # Incorporate microstructure signals
    data['microstructure_adjustment'] = data['amount_efficiency'] * (1 + data['large_trade_impact'])
    
    # Final factor
    data['pvma_factor'] = data['adjusted_score'] * data['microstructure_adjustment']
    
    # Clean up intermediate columns
    factor = data['pvma_factor'].copy()
    
    return factor
