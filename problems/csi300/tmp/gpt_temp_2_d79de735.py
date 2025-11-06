import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining temporal asymmetry, price discovery quality, 
    volatility clustering, and liquidity dynamics with cross-temporal integration.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # 1. Multi-Horizon Momentum Divergence
    df['momentum_2d'] = df['close'].pct_change(2)
    df['momentum_7d'] = df['close'].pct_change(7)
    df['momentum_divergence'] = df['momentum_2d'] / (df['momentum_7d'] + 1e-8)
    
    # Momentum regime transition probability (using rolling correlation)
    df['momentum_corr_5d'] = df['momentum_2d'].rolling(window=5).corr(df['momentum_7d'])
    df['momentum_regime_prob'] = 1 - abs(df['momentum_corr_5d'])
    
    # 2. Volume-Time Asymmetry Dynamics
    # Early vs late session volume concentration (assuming first/last 30% of volume)
    df['volume_cumsum'] = df['volume'].cumsum()
    df['volume_total_30pct'] = df['volume_cumsum'] * 0.3
    df['early_volume_ratio'] = df['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[:int(len(x)*0.3)]) / (np.sum(x) + 1e-8)
    )
    
    # Volume acceleration
    df['volume_ma_3'] = df['volume'].rolling(window=3).mean()
    df['volume_ma_8'] = df['volume'].rolling(window=8).mean()
    df['volume_acceleration'] = df['volume_ma_3'] / (df['volume_ma_8'] + 1e-8)
    
    # 3. Price Discovery Quality - Opening vs Closing Efficiency
    df['opening_efficiency'] = (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['closing_efficiency'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['efficiency_divergence'] = df['opening_efficiency'] - df['closing_efficiency']
    
    # 4. Bid-Ask Imbalance Proxy
    df['price_range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['upper_volume_concentration'] = df['volume'] * df['price_range_position']
    df['volume_pressure_asymmetry'] = (df['upper_volume_concentration'].rolling(window=5).mean() - 
                                      df['volume'].rolling(window=5).mean() * 0.5)
    
    # 5. Multi-Scale Volatility Asymmetry
    df['volatility_3d'] = df['close'].pct_change().rolling(window=3).std()
    df['volatility_8d'] = df['close'].pct_change().rolling(window=8).std()
    df['volatility_ratio'] = df['volatility_3d'] / (df['volatility_8d'] + 1e-8)
    
    # Volatility clustering intensity
    df['volatility_persistence'] = df['volatility_3d'].rolling(window=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0
    )
    
    # 6. Session-Based Liquidity Analysis
    df['liquidity_timing'] = df['early_volume_ratio'] * df['volume_acceleration']
    
    # 7. Range Compression & Expansion Timing
    df['daily_range'] = (df['high'] - df['low']) / (df['close'].shift(1) + 1e-8)
    df['range_efficiency'] = df['daily_range'] * df['closing_efficiency']
    
    # 8. Cross-Temporal Integration
    # Temporal regime classification
    df['morning_dominance'] = df['early_volume_ratio'] * df['opening_efficiency']
    df['afternoon_acceleration'] = (1 - df['early_volume_ratio']) * df['closing_efficiency']
    
    # Session transition signals
    df['session_transition'] = abs(df['morning_dominance'] - df['afternoon_acceleration'])
    
    # 9. Dynamic Temporal Enhancement
    # Multi-session consistency validation
    df['momentum_consistency'] = df['momentum_2d'].rolling(window=3).std()
    df['volume_consistency'] = df['volume'].rolling(window=3).std()
    
    # Final factor integration with temporal weighting
    df['temporal_asymmetry_factor'] = (
        df['momentum_divergence'] * 0.15 +
        df['momentum_regime_prob'] * 0.12 +
        df['volume_acceleration'] * 0.13 +
        df['efficiency_divergence'] * 0.14 +
        df['volume_pressure_asymmetry'] * 0.11 +
        df['volatility_ratio'] * 0.10 +
        df['liquidity_timing'] * 0.08 +
        df['range_efficiency'] * 0.07 +
        df['session_transition'] * 0.06 +
        df['momentum_consistency'] * 0.04
    )
    
    # Apply volatility-based filtering
    volatility_filter = df['volatility_8d'].rolling(window=10).rank(pct=True)
    df['final_factor'] = df['temporal_asymmetry_factor'] * volatility_filter
    
    # Clean up intermediate columns
    result = df['final_factor'].copy()
    
    return result
