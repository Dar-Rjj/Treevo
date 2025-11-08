import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Volatility Persistence with Liquidity Shock Absorption factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Volatility Component
    # Directional Volatility Calculation
    data['upside_vol'] = (data['high'] - data['close']) / data['close']
    data['downside_vol'] = (data['close'] - data['low']) / data['close']
    
    # Volatility Persistence Analysis
    data['upside_persistence'] = data['upside_vol'] - data['upside_vol'].shift(2)
    data['downside_persistence'] = data['downside_vol'] - data['downside_vol'].shift(2)
    
    # Net Volatility Bias
    vol_diff = abs(data['upside_vol'] - data['downside_vol'])
    data['net_vol_bias'] = (data['upside_persistence'] - data['downside_persistence']) * vol_diff
    
    # Regime Change Detection
    data['vol_regime'] = data['upside_vol'].rolling(window=5).std() / data['upside_vol'].rolling(window=20).std()
    
    # Liquidity Shock Component
    # Effective Spread
    data['effective_spread'] = 2 * (data['high'] - data['low']) / (data['high'] + data['low'])
    
    # Volume Concentration
    data['volume_percentile'] = data['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
    )
    
    # Price Resilience (intraday retracement)
    data['price_resilience'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Liquidity Depth
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['liquidity_depth'] = data['avg_trade_size'].rolling(window=5).std() / data['avg_trade_size'].rolling(window=20).std()
    
    # Shock Propagation Analysis
    data['volume_shock_persistence'] = data['volume'].pct_change().rolling(window=3).std()
    
    # Microstructure Efficiency Component
    # Price Discovery Efficiency
    data['price_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Noise-to-Signal Ratio
    intraday_range = data['high'] - data['low']
    data['noise_signal_ratio'] = intraday_range.rolling(window=5).std() / intraday_range.rolling(window=20).std()
    
    # Order Flow Imbalance
    price_change = data['close'].pct_change()
    data['order_flow_pressure'] = price_change.rolling(window=3).sum() * data['volume'].rolling(window=3).mean()
    
    # Flow Reversal Potential
    data['reversal_probability'] = -data['order_flow_pressure'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0
    )
    
    # Adaptive Signal Integration
    # Regime-Weighted Combination
    high_vol_regime = data['vol_regime'] > data['vol_regime'].rolling(window=20).quantile(0.7)
    low_vol_regime = data['vol_regime'] < data['vol_regime'].rolling(window=20).quantile(0.3)
    
    # Liquidity condition filtering
    stressed_liquidity = data['effective_spread'] > data['effective_spread'].rolling(window=20).quantile(0.7)
    normal_liquidity = ~stressed_liquidity
    
    # Component weights based on regimes
    shock_component_weight = np.where(high_vol_regime | stressed_liquidity, 0.6, 0.3)
    micro_component_weight = np.where(low_vol_regime & normal_liquidity, 0.6, 0.3)
    vol_component_weight = 1.0 - shock_component_weight - micro_component_weight
    
    # Component signals
    vol_component = data['net_vol_bias'].fillna(0)
    
    shock_component = (
        -data['effective_spread'].fillna(0) + 
        data['price_resilience'].fillna(0) - 
        data['liquidity_depth'].fillna(0) - 
        data['volume_shock_persistence'].fillna(0)
    )
    
    micro_component = (
        data['price_range_efficiency'].fillna(0) - 
        data['noise_signal_ratio'].fillna(0) + 
        data['reversal_probability'].fillna(0)
    )
    
    # Dynamic Horizon Selection
    # Short-term signal (1-2 days)
    short_term_signal = (
        0.7 * micro_component.rolling(window=2).mean() + 
        0.3 * shock_component.rolling(window=2).mean()
    )
    
    # Medium-term signal (3-5 days)
    medium_term_signal = (
        0.6 * vol_component.rolling(window=5).mean() + 
        0.4 * shock_component.rolling(window=5).mean()
    )
    
    # Combined signal with regime weighting
    combined_signal = (
        vol_component_weight * vol_component +
        shock_component_weight * shock_component +
        micro_component_weight * micro_component
    )
    
    # Final signal with horizon blending
    final_signal = (
        0.4 * short_term_signal + 
        0.6 * medium_term_signal + 
        0.2 * combined_signal
    )
    
    # Risk-Adjusted Signal Refinement
    # Volatility Scaling
    recent_vol = data['close'].pct_change().rolling(window=10).std()
    vol_scaling = 1.0 / (1.0 + recent_vol.fillna(0))
    
    # Liquidity Buffer
    liquidity_buffer = 1.0 - data['effective_spread'].rolling(window=10).rank(pct=True)
    
    # Final risk-adjusted factor
    risk_adjusted_factor = final_signal * vol_scaling * liquidity_buffer
    
    return risk_adjusted_factor
