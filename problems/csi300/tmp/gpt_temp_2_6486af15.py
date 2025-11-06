import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume-Liquidity Alignment alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Momentum Components
    # Price Momentum
    data['price_momentum_st'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_mt'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_lt'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum
    data['volume_momentum_st'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_mt'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_lt'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Liquidity Momentum
    data['amount_momentum'] = data['amount'] / data['amount'].shift(5) - 1
    data['volume_to_amount_ratio'] = data['volume'] / data['amount']
    data['volume_to_amount_ratio_change'] = (
        data['volume_to_amount_ratio'] / data['volume_to_amount_ratio'].shift(5) - 1
    )
    
    # Volatility Scaling & Risk Adjustment
    # Price Volatility
    data['price_vol_st'] = data['close'].rolling(5).std()
    data['price_vol_mt'] = data['close'].rolling(10).std()
    data['price_vol_lt'] = data['close'].rolling(20).std()
    
    # Volume Volatility
    data['volume_vol_st'] = data['volume'].rolling(5).std()
    data['volume_vol_mt'] = data['volume'].rolling(10).std()
    data['volume_vol_lt'] = data['volume'].rolling(20).std()
    
    # Liquidity Volatility
    data['amount_vol'] = data['amount'].rolling(5).std()
    data['volume_to_amount_ratio_vol'] = data['volume_to_amount_ratio'].rolling(5).std()
    
    # Momentum-Volume Alignment Signals
    # Raw Alignment Components
    data['alignment_st'] = data['price_momentum_st'] * data['volume_momentum_st']
    data['alignment_mt'] = data['price_momentum_mt'] * data['volume_momentum_mt']
    data['alignment_lt'] = data['price_momentum_lt'] * data['volume_momentum_lt']
    
    # Volatility-Adjusted Alignment
    data['alignment_st_vol_adj'] = data['alignment_st'] / (
        data['price_vol_st'] * data['volume_vol_st'] + 1e-8
    )
    data['alignment_mt_vol_adj'] = data['alignment_mt'] / (
        data['price_vol_mt'] * data['volume_vol_mt'] + 1e-8
    )
    data['alignment_lt_vol_adj'] = data['alignment_lt'] / (
        data['price_vol_lt'] * data['volume_vol_lt'] + 1e-8
    )
    
    # Liquidity-Enhanced Alignment
    data['liquidity_momentum_alignment'] = (
        data['amount_momentum'] * data['volume_to_amount_ratio_change']
    )
    data['liquidity_vol_adj'] = data['liquidity_momentum_alignment'] / (
        data['amount_vol'] * data['volume_to_amount_ratio_vol'] + 1e-8
    )
    
    # Market Regime Detection
    # Volatility Regime
    data['stock_volatility'] = data['close'].rolling(20).std()
    vol_threshold_60 = data['stock_volatility'].rolling(20).quantile(0.6)
    vol_threshold_40 = data['stock_volatility'].rolling(20).quantile(0.4)
    
    data['volatility_regime'] = 'normal'
    data.loc[data['stock_volatility'] > vol_threshold_60, 'volatility_regime'] = 'high'
    data.loc[data['stock_volatility'] < vol_threshold_40, 'volatility_regime'] = 'low'
    
    # Volatility persistence
    data['volatility_persistence'] = data['stock_volatility'].rolling(10).corr(
        data['stock_volatility'].shift(10)
    )
    
    # Trend Regime
    data['price_trend_strength'] = data['close'] / data['close'].shift(20) - 1
    data['trend_direction'] = np.sign(data['price_trend_strength'])
    
    # Trend consistency using correlation with linear trend
    def trend_consistency(series):
        if len(series) < 10:
            return np.nan
        x = np.arange(1, 11)
        y = series.values[-10:]
        if np.std(y) == 0:
            return 0
        return np.corrcoef(x, y)[0, 1]
    
    data['trend_consistency'] = data['close'].rolling(10).apply(
        trend_consistency, raw=False
    )
    
    # Liquidity Regime
    amount_threshold_60 = data['amount'].rolling(20).quantile(0.6)
    data['liquidity_trend'] = data['amount'] / data['amount'].shift(10) - 1
    
    data['liquidity_regime'] = 'normal'
    high_liquidity_condition = (
        (data['amount'] > amount_threshold_60) & 
        (data['liquidity_trend'] > 0)
    )
    data.loc[high_liquidity_condition, 'liquidity_regime'] = 'high'
    
    # Regime-Adaptive Signal Combination
    # Initialize adapted signals
    data['alignment_st_adapted'] = data['alignment_st_vol_adj']
    data['alignment_mt_adapted'] = data['alignment_mt_vol_adj']
    data['alignment_lt_adapted'] = data['alignment_lt_vol_adj']
    data['liquidity_adapted'] = data['liquidity_vol_adj']
    
    # Volatility Regime Adaptation
    high_vol_mask = data['volatility_regime'] == 'high'
    low_vol_mask = data['volatility_regime'] == 'low'
    
    data.loc[high_vol_mask, 'alignment_st_adapted'] *= 0.7
    data.loc[high_vol_mask, 'alignment_mt_adapted'] *= 1.5
    data.loc[low_vol_mask, 'alignment_st_adapted'] *= 1.5
    data.loc[low_vol_mask, 'alignment_lt_adapted'] *= 0.7
    
    # Trend Regime Adaptation
    bull_trend_mask = (
        (data['price_trend_strength'] > 0.05) & 
        (data['trend_consistency'] > 0.5)
    )
    bear_trend_mask = (
        (data['price_trend_strength'] < -0.05) & 
        (data['trend_consistency'] < -0.5)
    )
    sideways_mask = abs(data['price_trend_strength']) < 0.02
    
    # Apply trend regime multipliers
    data.loc[bull_trend_mask, 'alignment_st_adapted'] *= 1.3
    data.loc[bull_trend_mask, 'alignment_mt_adapted'] *= 1.3
    data.loc[bull_trend_mask, 'alignment_lt_adapted'] *= 1.3
    data.loc[bull_trend_mask, 'liquidity_adapted'] *= 1.3
    
    data.loc[bear_trend_mask, 'alignment_st_adapted'] *= 0.8
    data.loc[bear_trend_mask, 'alignment_mt_adapted'] *= 0.8
    data.loc[bear_trend_mask, 'alignment_lt_adapted'] *= 0.8
    data.loc[bear_trend_mask, 'liquidity_adapted'] *= 1.2  # Increase liquidity emphasis
    
    data.loc[sideways_mask, 'alignment_st_adapted'] *= -1  # Reverse for mean reversion
    data.loc[sideways_mask, 'alignment_st_adapted'] *= 1.5  # Increase ST weight
    
    # Liquidity Regime Adaptation
    high_liq_mask = data['liquidity_regime'] == 'high'
    data.loc[high_liq_mask, 'liquidity_adapted'] *= 1.4
    data.loc[~high_liq_mask, 'liquidity_adapted'] *= 0.6
    
    # Signal Integration & Bounding
    # Multi-Timeframe Combination
    data['base_combined'] = (
        data['alignment_st_adapted'] * 
        data['alignment_mt_adapted'] * 
        data['alignment_lt_adapted']
    )
    data['liquidity_enhanced'] = data['base_combined'] * data['liquidity_adapted']
    
    # Signal Bounding & Stabilization
    data['bounded_signal'] = np.tanh(data['liquidity_enhanced'])
    data['bounded_signal'] = np.sign(data['bounded_signal']) * np.maximum(
        abs(data['bounded_signal']), 0.001
    )
    
    # Cross-Sectional Enhancement
    # Relative Strength
    signal_median = data['bounded_signal'].rolling(20).quantile(0.5)
    data['relative_strength'] = data['bounded_signal'] / (signal_median + 1e-8)
    
    # Momentum Persistence
    def momentum_persistence(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(1, 6)
        y = series.values[-5:]
        if np.std(y) == 0:
            return 0
        return np.corrcoef(x, y)[0, 1]
    
    data['momentum_persistence'] = data['bounded_signal'].rolling(5).apply(
        momentum_persistence, raw=False
    )
    
    # Signal Quality
    signal_std = data['bounded_signal'].rolling(5).std()
    signal_mean = data['bounded_signal'].rolling(5).mean()
    data['signal_quality'] = 1 - (signal_std / (abs(signal_mean) + 1e-8))
    data['signal_quality'] = np.clip(data['signal_quality'], 0, 1)
    
    # Final Alpha Factor
    data['confidence_score'] = data['signal_quality'] * data['momentum_persistence']
    data['final_alpha'] = data['bounded_signal'] * data['confidence_score']
    
    return data['final_alpha']
