import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Volatility-Regime Alpha Factor
    Combines price-volume divergence across multiple timeframes with volatility normalization,
    regime detection, and amount flow integration to generate predictive signals.
    """
    df = data.copy()
    
    # Price-Volume Divergence Engine
    # Short-Term Component (5-day)
    df['price_momentum_st'] = df['close'] / df['close'].shift(5) - 1
    df['volume_momentum_st'] = df['volume'] / df['volume'].shift(5) - 1
    df['divergence_st'] = df['price_momentum_st'] - df['volume_momentum_st']
    
    # Medium-Term Component (10-day)
    df['price_momentum_mt'] = df['close'] / df['close'].shift(10) - 1
    df['volume_momentum_mt'] = df['volume'] / df['volume'].shift(10) - 1
    df['divergence_mt'] = df['price_momentum_mt'] - df['volume_momentum_mt']
    
    # Long-Term Component (20-day)
    df['price_momentum_lt'] = df['close'] / df['close'].shift(20) - 1
    df['volume_momentum_lt'] = df['volume'] / df['volume'].shift(20) - 1
    df['divergence_lt'] = df['price_momentum_lt'] - df['volume_momentum_lt']
    
    # Multi-Timeframe Combination
    df['composite_divergence'] = 0.5 * df['divergence_st'] + 0.3 * df['divergence_mt'] + 0.2 * df['divergence_lt']
    
    # Volatility Normalization Layer
    df['daily_returns'] = df['close'] / df['close'].shift(1) - 1
    df['volatility_st'] = df['daily_returns'].rolling(window=5).std()
    df['volatility_mt'] = df['daily_returns'].rolling(window=20).std()
    
    df['volatility_floor'] = np.maximum(df['volatility_mt'], 0.001)
    df['volatility_ratio'] = df['volatility_st'] / df['volatility_floor']
    df['normalized_divergence'] = df['composite_divergence'] / df['volatility_floor']
    
    # Regime Detection & Filtering
    # Volume Regime Analysis
    df['volume_mean_20'] = df['volume'].rolling(window=20).mean()
    df['volume_std_20'] = df['volume'].rolling(window=20).std()
    df['volume_zscore'] = (df['volume'] - df['volume_mean_20']) / df['volume_std_20']
    df['high_volume_regime'] = df['volume_zscore'] > 1
    df['low_volume_regime'] = df['volume_zscore'] < -1
    
    # Volatility Regime Analysis
    df['vol_mean_20'] = df['volatility_st'].rolling(window=20).mean()
    df['vol_std_20'] = df['volatility_st'].rolling(window=20).std()
    df['volatility_zscore'] = (df['volatility_st'] - df['vol_mean_20']) / df['vol_std_20']
    df['high_vol_regime'] = df['volatility_zscore'] > 1
    df['low_vol_regime'] = df['volatility_zscore'] < -1
    
    # Regime Persistence Scoring
    def calculate_persistence(series):
        persistence = pd.Series(index=series.index, dtype=float)
        current_count = 0
        for i, val in enumerate(series):
            if i == 0:
                persistence.iloc[i] = 0
                continue
            if val == series.iloc[i-1]:
                current_count += 1
            else:
                current_count = 0
            persistence.iloc[i] = current_count
        return persistence
    
    df['volume_persistence'] = calculate_persistence(df['high_volume_regime'] | df['low_volume_regime'])
    df['volatility_persistence'] = calculate_persistence(df['high_vol_regime'] | df['low_vol_regime'])
    df['combined_persistence'] = df['volume_persistence'] + df['volatility_persistence']
    
    # Amount Flow Integration
    # Daily Money Flow
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['raw_money_flow'] = df['typical_price'] * df['volume']
    df['directional_flow'] = np.sign(df['close'] - df['open']) * df['raw_money_flow']
    
    # Flow Momentum
    df['flow_change_5d'] = df['directional_flow'] / df['directional_flow'].shift(5) - 1
    df['flow_price_alignment'] = np.sign(df['flow_change_5d']) * np.sign(df['price_momentum_st'])
    
    # Flow-Regime Integration
    df['flow_mean_20'] = df['directional_flow'].abs().rolling(window=20).mean()
    df['flow_std_20'] = df['directional_flow'].abs().rolling(window=20).std()
    df['high_flow_regime'] = df['directional_flow'].abs() > (df['flow_mean_20'] + df['flow_std_20'])
    df['flow_persistence'] = calculate_persistence(df['high_flow_regime'])
    
    # Alpha Signal Generation
    # Regime Weight Assignment
    conditions = [
        df['combined_persistence'] >= 8,
        (df['combined_persistence'] >= 4) & (df['combined_persistence'] < 8),
        (df['combined_persistence'] >= 1) & (df['combined_persistence'] < 4),
        df['combined_persistence'] == 0
    ]
    choices = [2.0, 1.5, 1.0, 0.5]
    df['regime_weight'] = np.select(conditions, choices, default=1.0)
    
    # Flow Enhancement Factor
    df['flow_alignment_bonus'] = np.where(df['flow_price_alignment'] > 0, 0.3, 0)
    df['flow_persistence_bonus'] = 0.1 * np.minimum(df['flow_persistence'], 5)
    df['flow_enhancement'] = df['flow_alignment_bonus'] + df['flow_persistence_bonus']
    
    # Final Alpha Factor
    df['base_alpha'] = df['normalized_divergence'] * df['regime_weight']
    df['enhanced_alpha'] = df['base_alpha'] * (1 + df['flow_enhancement'])
    df['final_signal'] = df['enhanced_alpha'] * np.sign(df['composite_divergence'])
    
    return df['final_signal']
