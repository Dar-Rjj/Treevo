import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Non-Linear Price Discovery & Information Asymmetry Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Basic price and volume calculations
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Information Flow Dynamics
    # Price Discovery Efficiency
    data['opening_efficiency'] = ((data['open'] - data['prev_close']) / 
                                 (data['high'] - data['low'] + 1e-8) * data['volume'])
    data['closing_efficiency'] = ((data['close'] - data['open']) / 
                                 (data['high'] - data['low'] + 1e-8) * data['amount'])
    data['intraday_discovery'] = ((data['high'] - data['low']) / 
                                 (data['open'] - data['prev_close'] + 1e-8) * data['volume'])
    
    # Information Asymmetry Patterns
    data['informed_trading'] = (np.abs(data['close'] - data['open']) / 
                               (data['high'] - data['low'] + 1e-8) * np.log(data['volume'] + 1))
    data['price_impact'] = ((data['close'] - data['open']) ** 2 / 
                           (data['high'] - data['low'] + 1e-8) * data['volume'])
    data['information_leakage'] = ((data['high'] - data['prev_high']) / 
                                  (data['low'] - data['prev_low'] + 1e-8) * 
                                  np.sign(data['close'] - data['open']))
    
    # Market Microstructure Noise
    data['noise_ratio'] = (np.abs((data['high'] + data['low']) / 2 - data['close']) / 
                          (np.abs(data['close'] - data['open']) + 1e-8))
    data['microstructure_friction'] = ((data['high'] - data['low']) / 
                                      (data['amount'] / (data['volume'] + 1e-8) + 1e-8))
    
    # Price staleness (5-day correlation)
    data['abs_close_open'] = np.abs(data['close'] - data['open'])
    data['price_staleness'] = data['abs_close_open'].rolling(window=5).corr(data['volume'])
    
    # Multi-Scale Information Processing
    # Short-term Information Flow
    data['opening_momentum'] = ((data['open'] - data['prev_close']) / 
                               (data['close'] - data['open'] + 1e-8))
    data['intraday_absorption'] = ((data['close'] - data['open']) / 
                                  (data['high'] - data['low'] + 1e-8) * np.log(data['amount'] + 1))
    data['eod_resolution'] = (np.abs(data['close'] - (data['high'] + data['low']) / 2) / 
                             (data['high'] - data['low'] + 1e-8))
    
    # Medium-term Information Persistence
    data['5d_return'] = data['close'].pct_change(5)
    data['10d_avg_abs_range'] = data['abs_close_open'].rolling(window=10).mean()
    data['information_momentum'] = data['5d_return'] / (data['10d_avg_abs_range'] + 1e-8)
    
    # Volume-information efficiency (10-day correlation)
    data['volume_info_efficiency'] = data['volume'].rolling(window=10).corr(data['abs_close_open'])
    
    # Price discovery trend (10-day slope of normalized range)
    data['normalized_range'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['price_discovery_trend'] = data['normalized_range'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan)
    
    # Long-term Structural Information
    data['information_capacity'] = (data['amount'] / (data['high'] - data['low'] + 1e-8)).rolling(window=20).mean()
    data['price_efficiency_stability'] = data['normalized_range'].rolling(window=20).var()
    data['market_depth'] = data['volume'] / (np.abs(data['close'] - data['open']) + 1e-8)
    data['market_depth_evolution'] = data['market_depth'].pct_change(20)
    
    # Cross-Dimensional Information Integration
    # Price-Volume Information Alignment
    data['volume_discovery'] = ((data['close'] - data['open']) * np.log(data['volume'] + 1) / 
                               (data['high'] - data['low'] + 1e-8))
    data['amount_information'] = data['amount'] / (np.abs(data['close'] - data['open']) * data['volume'] + 1e-8)
    
    # Information flow consistency (5-day correlation of changes)
    data['volume_change'] = data['volume'].pct_change()
    data['abs_range_change'] = data['abs_close_open'].pct_change()
    data['info_flow_consistency'] = data['volume_change'].rolling(window=5).corr(data['abs_range_change'])
    
    # Volatility context for regime-dependent weighting
    data['volatility'] = data['normalized_range'].rolling(window=10).std()
    data['volatility_rank'] = data['volatility'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan)
    
    # Composite Alpha Generation
    # Core Information Components
    data['price_discovery_composite'] = (
        data['opening_efficiency'].fillna(0) + 
        data['closing_efficiency'].fillna(0) + 
        data['intraday_discovery'].fillna(0)
    )
    
    data['information_asymmetry_composite'] = (
        data['informed_trading'].fillna(0) + 
        data['price_impact'].fillna(0) + 
        data['information_leakage'].fillna(0)
    )
    
    data['multi_scale_composite'] = (
        data['information_momentum'].fillna(0) + 
        data['volume_info_efficiency'].fillna(0) + 
        data['price_discovery_trend'].fillna(0)
    )
    
    # Regime-Adaptive Weighting
    # Volatility context weighting
    high_vol_weight = data['volatility_rank'].fillna(0.5)
    low_vol_weight = 1 - high_vol_weight
    
    # Apply regime weights
    data['regime_weighted_discovery'] = (
        high_vol_weight * data['information_leakage'].fillna(0) +
        low_vol_weight * data['price_discovery_composite'].fillna(0)
    )
    
    data['regime_weighted_asymmetry'] = (
        high_vol_weight * data['information_asymmetry_composite'].fillna(0) +
        low_vol_weight * data['microstructure_friction'].fillna(0)
    )
    
    data['regime_weighted_multi_scale'] = (
        high_vol_weight * data['information_momentum'].fillna(0) +
        low_vol_weight * data['multi_scale_composite'].fillna(0)
    )
    
    # Cross-dimensional confidence measures
    data['timeframe_alignment'] = (
        data['information_momentum'].fillna(0) * 
        data['price_discovery_trend'].fillna(0) * 
        data['market_depth_evolution'].fillna(0)
    )
    
    data['volume_confirmation'] = (
        data['volume_discovery'].fillna(0) * 
        data['amount_information'].fillna(0) * 
        data['info_flow_consistency'].fillna(0)
    )
    
    # Final Alpha Generation
    alpha = (
        data['regime_weighted_discovery'].fillna(0) * 
        np.sign(data['information_momentum'].fillna(0)) *
        data['regime_weighted_asymmetry'].fillna(0) *
        (1 + data['timeframe_alignment'].fillna(0)) *
        (1 + data['volume_confirmation'].fillna(0))
    )
    
    # Normalize and clean
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = (alpha - alpha.rolling(window=20, min_periods=10).mean()) / (
        alpha.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return alpha
