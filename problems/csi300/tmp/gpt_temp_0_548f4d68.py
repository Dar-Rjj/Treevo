import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Adaptive Breakout Efficiency Alpha Factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Volatility Assessment
    # Short-term: 5-day High-Low range mean
    data['short_term_vol'] = (data['high'] - data['low']).rolling(window=5).mean()
    
    # Medium-term: 10-day return standard deviation
    data['returns'] = data['close'].pct_change()
    data['medium_term_vol'] = data['returns'].rolling(window=10).std()
    
    # Long-term: 20-day Open-Close spread mean
    data['long_term_vol'] = abs(data['close'] - data['open']).rolling(window=20).mean()
    
    # Volatility Regime Classification
    # Current vs Historical Volatility Comparison
    data['vol_regime'] = 1  # Medium by default
    current_vol = (data['high'] - data['low']).rolling(window=5).mean()
    hist_vol = (data['high'] - data['low']).rolling(window=20).mean()
    vol_ratio = current_vol / hist_vol
    
    # Regime classification
    data.loc[vol_ratio > 1.5, 'vol_regime'] = 2  # High volatility
    data.loc[vol_ratio < 0.7, 'vol_regime'] = 0  # Low volatility
    
    # Microstructure Noise Ratio
    intraday_vol = (data['high'] - data['low']) / data['close']
    overnight_gap = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['noise_ratio'] = intraday_vol / (overnight_gap + 1e-8)
    
    # Multi-timeframe Breakout Strength
    # Distance from 50-day High/Low
    data['dist_from_high'] = (data['close'] - data['high'].rolling(window=50).max()) / data['close']
    data['dist_from_low'] = (data['close'] - data['low'].rolling(window=50).min()) / data['close']
    data['breakout_strength'] = data['dist_from_high'] - data['dist_from_low']
    
    # Closing Position Efficiency
    data['close_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Momentum-Volume-Efficiency Convergence
    # 5/10/20-day Price Momentum Alignment
    mom_5 = data['close'].pct_change(5)
    mom_10 = data['close'].pct_change(10)
    mom_20 = data['close'].pct_change(20)
    data['momentum_alignment'] = np.sign(mom_5) + np.sign(mom_10) + np.sign(mom_20)
    
    # Volume clustering persistence
    vol_avg_20 = data['volume'].rolling(window=20).mean()
    data['high_volume_days'] = (data['volume'] > vol_avg_20).rolling(window=5).sum()
    
    # Volume-Pressure Surge Validation
    # Volume Spike Detection
    data['volume_spike'] = (data['volume'] > (2 * vol_avg_20)).astype(int)
    
    # Pressure Surge: Net Pressure Ratio
    data['pressure_ratio'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['pressure_surge'] = (data['pressure_ratio'].abs() > data['pressure_ratio'].rolling(window=10).std() * 1.5).astype(int)
    
    # Intraday Microstructure Efficiency Patterns
    # Morning-afternoon efficiency divergence (simplified)
    morning_return = (data['high'] - data['open']) / data['open']
    afternoon_return = (data['close'] - data['low']) / data['low']
    data['session_divergence'] = (afternoon_return - morning_return) / (data['short_term_vol'] + 1e-8)
    
    # Overnight Gap Closure Efficiency
    data['gap_closure_eff'] = abs(data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 1e-8)
    
    # Microstructure Efficiency Assessment
    data['effective_spread_eff'] = (data['high'] - data['low']) / data['close'] * data['volume']
    data['price_impact_asym'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * data['volume']
    
    # Multi-Timeframe Efficiency Divergence
    # Short-term Efficiency Patterns
    # Intraday reversal efficiency
    high_condition = data['high'] > data['high'].shift(1)
    close_condition = data['close'] < data['open']
    data['intraday_reversal'] = (high_condition & close_condition).rolling(window=5).sum()
    
    # Medium-term Efficiency Momentum
    # Volatility-weighted efficiency
    data['vol_weighted_eff'] = data['returns'].rolling(window=10).mean() / (data['medium_term_vol'] + 1e-8) * np.sign(data['returns'].rolling(window=10).mean())
    
    # Liquidity momentum efficiency
    data['liquidity_ratio'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['liquidity_mom_eff'] = data['liquidity_ratio'].pct_change(5)
    
    # Long-term Structural Efficiency
    # Price-range efficiency trend
    range_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_trend'] = range_efficiency.rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Liquidity regime efficiency
    data['liquidity_regime_eff'] = (data['volume'] > (1.5 * data['volume'].rolling(window=20).mean())).rolling(window=20).sum()
    
    # Composite Factor Construction with Regime-Adaptive Weighting
    # Core Efficiency Components
    microstructure_eff = (
        data['close_efficiency'] + 
        data['effective_spread_eff'].rank(pct=True) + 
        data['price_impact_asym'].rank(pct=True)
    ) / 3
    
    breakout_momentum = (
        data['breakout_strength'] + 
        data['momentum_alignment'] + 
        (data['volume_spike'] * data['pressure_surge'])
    ) / 3
    
    multi_timeframe_divergence = (
        data['session_divergence'].rank(pct=True) + 
        data['vol_weighted_eff'].rank(pct=True) + 
        data['efficiency_trend'].rank(pct=True)
    ) / 3
    
    # Regime-Adaptive Weighting
    regime_weights = {
        0: [0.3, 0.4, 0.3],  # Low volatility: focus on convergence patterns
        1: [0.4, 0.3, 0.3],  # Medium volatility: balanced approach
        2: [0.2, 0.5, 0.3]   # High volatility: amplify breakout momentum
    }
    
    # Apply regime-specific weights
    alpha_values = []
    for idx, row in data.iterrows():
        regime = row['vol_regime']
        weights = regime_weights[regime]
        alpha = (
            weights[0] * microstructure_eff.loc[idx] +
            weights[1] * breakout_momentum.loc[idx] +
            weights[2] * multi_timeframe_divergence.loc[idx]
        )
        alpha_values.append(alpha)
    
    data['alpha'] = alpha_values
    
    # Final confidence adjustment based on cross-dimensional alignment
    confidence_components = (
        np.sign(microstructure_eff) == np.sign(breakout_momentum)
    ) & (
        np.sign(breakout_momentum) == np.sign(multi_timeframe_divergence)
    )
    
    data['confidence_multiplier'] = 1 + (0.5 * confidence_components)
    data['final_alpha'] = data['alpha'] * data['confidence_multiplier']
    
    return data['final_alpha']
