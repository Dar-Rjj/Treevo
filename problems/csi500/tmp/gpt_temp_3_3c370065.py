import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Liquidity Momentum Divergence Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['price_range'] = df['high'] - df['low']
    df['price_return'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # 1. Bidirectional Liquidity Momentum
    # Short-Term Liquidity Flow
    df['liquidity_efficiency'] = df['amount'] / (df['price_range'] + 1e-8)
    df['short_term_liquidity'] = df['liquidity_efficiency'].rolling(window=3, min_periods=3).sum()
    df['liquidity_momentum_5d'] = df['short_term_liquidity'].pct_change(periods=5)
    
    # Medium-Term Volume-Price Divergence
    df['volume_price_corr_10d'] = df['volume_change'].rolling(window=10, min_periods=10).corr(df['price_return'])
    
    # Volume vs Price trend divergence
    df['volume_trend_15d'] = df['volume'].rolling(window=15, min_periods=15).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    df['price_trend_15d'] = df['close'].rolling(window=15, min_periods=15).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    df['volume_price_divergence'] = df['volume_trend_15d'] - df['price_trend_15d']
    
    # 2. Market Microstructure Regimes
    # Order Flow Imbalance Detection
    df['amount_rank'] = df['amount'].rolling(window=5, min_periods=5).rank(pct=True)
    df['amount_concentration'] = df['amount_rank'].rolling(window=5, min_periods=5).apply(
        lambda x: np.percentile(x, 70) - np.percentile(x, 30), raw=True
    )
    df['amount_variance_5d'] = df['amount'].rolling(window=5, min_periods=5).var()
    
    # Price Impact Sensitivity
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['shadow_asymmetry'] = (df['upper_shadow'] - df['lower_shadow']) / (df['price_range'] + 1e-8)
    df['shadow_efficiency_ratio'] = (df['upper_shadow'].rolling(window=10, min_periods=10).mean() + 1e-8) / \
                                   (df['lower_shadow'].rolling(window=10, min_periods=10).mean() + 1e-8)
    
    # 3. Cross-Timeframe Momentum Fractals
    # Multi-Scale Momentum Alignment
    df['momentum_3d'] = df['close'].pct_change(periods=3)
    df['momentum_10d'] = df['close'].pct_change(periods=10)
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    df['momentum_15d'] = df['close'].pct_change(periods=15)
    
    df['momentum_direction_consistency'] = np.sign(df['momentum_3d']) * np.sign(df['momentum_10d'])
    df['momentum_magnitude_ratio'] = (abs(df['momentum_5d']) + 1e-8) / (abs(df['momentum_15d']) + 1e-8)
    
    # Fractal Pattern Recognition
    for window in [3, 7, 15]:
        df[f'range_efficiency_{window}d'] = (df['close'].diff(window).abs() + 1e-8) / \
                                           (df['high'].rolling(window).max() - df['low'].rolling(window).min() + 1e-8)
    
    df['fractal_range_efficiency'] = (df['range_efficiency_3d'] * df['range_efficiency_7d'] * 
                                     df['range_efficiency_15d']) ** (1/3)
    
    # Volume clustering patterns
    df['volume_zscore_5d'] = (df['volume'] - df['volume'].rolling(window=5, min_periods=5).mean()) / \
                            (df['volume'].rolling(window=5, min_periods=5).std() + 1e-8)
    df['volume_clustering'] = df['volume_zscore_5d'].rolling(window=5, min_periods=5).apply(
        lambda x: np.sum(np.abs(x) > 1) / len(x), raw=True
    )
    
    # 4. Dynamic Factor Integration
    # Liquidity-Momentum Interaction
    df['liquidity_modulated_momentum'] = df['liquidity_momentum_5d'] * df['short_term_liquidity']
    df['inverse_liquidity_efficiency'] = 1 / (df['liquidity_efficiency'] + 1e-8)
    
    # Microstructure Regime Multiplier
    df['order_flow_impact'] = df['amount_concentration'] * df['amount_variance_5d']
    df['price_impact_adjustment'] = df['shadow_asymmetry'] * df['shadow_efficiency_ratio']
    
    # Fractal Convergence Enhancement
    df['multi_scale_alignment'] = (df['momentum_direction_consistency'] * 
                                  df['momentum_magnitude_ratio'] * 
                                  df['fractal_range_efficiency'])
    
    # Final factor calculation with regime-dependent weighting
    df['regime_weight'] = 1 + df['order_flow_impact'] + df['price_impact_adjustment']
    
    # Combine all components
    liquidity_component = (df['liquidity_modulated_momentum'] * df['volume_price_corr_10d'] * 
                          df['volume_price_divergence'])
    
    microstructure_component = (df['inverse_liquidity_efficiency'] * df['regime_weight'] * 
                               df['volume_clustering'])
    
    fractal_component = df['multi_scale_alignment'] * df['fractal_range_efficiency']
    
    # Final factor with normalization
    result = (liquidity_component + microstructure_component + fractal_component) * df['regime_weight']
    
    # Remove any potential lookahead bias and ensure proper indexing
    result = result.fillna(0)
    
    return result
