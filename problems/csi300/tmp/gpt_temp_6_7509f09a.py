import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Momentum Divergence
    df = df.copy()
    
    # Momentum Components
    df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['volume_momentum'] = df['volume'] / df['volume'].shift(5)
    df['amount_momentum'] = df['amount'] / df['amount'].shift(5)
    
    # Divergence Patterns
    df['price_volume_divergence'] = df['price_momentum'] / df['volume_momentum']
    df['price_amount_divergence'] = df['price_momentum'] / df['amount_momentum']
    df['triple_divergence'] = df['price_momentum'] / (df['volume_momentum'] * df['amount_momentum'])
    
    # Momentum Persistence
    df['price_momentum_sign'] = np.sign(df['price_momentum'])
    df['momentum_consistency'] = df['price_momentum_sign'].rolling(window=5).apply(
        lambda x: sum(x.iloc[i] == x.iloc[i-1] for i in range(1, len(x)) if not pd.isna(x.iloc[i]) and not pd.isna(x.iloc[i-1])), 
        raw=False
    )
    df['volume_confirmation'] = df['volume_momentum'] * df['momentum_consistency']
    df['enhanced_divergence'] = df['price_volume_divergence'] * df['momentum_consistency']
    
    # Multi-Day Range Efficiency
    # Efficiency Components
    df['daily_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['cumulative_movement'] = abs(df['close'] - df['close'].shift(1)).rolling(window=3).sum()
    df['cumulative_range'] = (df['high'] - df['low']).rolling(window=3).sum()
    
    # Efficiency Persistence
    df['multi_day_efficiency'] = df['cumulative_movement'] / df['cumulative_range']
    df['efficiency_trend'] = df['multi_day_efficiency'] / df['daily_efficiency'].shift(1)
    df['efficiency_consistency'] = (df['daily_efficiency'] > 0.5).rolling(window=5).sum()
    
    # Volume-Weighted Efficiency
    df['volume_adjusted_efficiency'] = df['daily_efficiency'] * (df['volume'] / df['volume'].shift(1))
    df['persistent_volume_efficiency'] = df['multi_day_efficiency'] * df['efficiency_consistency']
    df['efficiency_momentum'] = df['efficiency_trend'] * df['persistent_volume_efficiency']
    
    # Directional Flow Quality
    # Flow Components
    df['flow_direction'] = np.sign(df['close'] - df['close'].shift(1))
    df['net_directional_flow'] = df['amount'] * df['flow_direction']
    df['flow_to_volume_ratio'] = df['net_directional_flow'] / df['volume']
    
    # Flow Persistence
    df['flow_momentum'] = df['net_directional_flow'].rolling(window=3).sum()
    df['flow_sign_consistency'] = df['net_directional_flow'].rolling(window=3).apply(
        lambda x: sum(np.sign(x.iloc[i]) == np.sign(x.iloc[-1]) for i in range(len(x)) if not pd.isna(x.iloc[i])), 
        raw=False
    )
    df['flow_acceleration'] = df['flow_momentum'] / df['flow_momentum'].shift(1)
    
    # Quality Assessment
    df['consistent_flow_quality'] = df['flow_momentum'] * df['flow_sign_consistency']
    df['volume_confirmed_flow'] = df['flow_to_volume_ratio'] * df['flow_acceleration']
    df['enhanced_flow_factor'] = df['consistent_flow_quality'] * df['volume_confirmed_flow']
    
    # Volatility-Volume Regime Dynamics
    # Volatility Structure
    df['short_term_volatility'] = df['close'].rolling(window=5).std()
    df['medium_term_volatility'] = df['close'].rolling(window=10).std()
    df['volatility_compression'] = df['short_term_volatility'] / df['medium_term_volatility']
    
    # Volume Regime
    df['volume_level'] = df['volume'] / df['volume'].rolling(window=10).mean()
    df['volume_persistence'] = (df['volume'] > df['volume'].rolling(window=10).mean()).rolling(window=5).sum()
    df['volume_clustering'] = df['volume_persistence'] / 5
    
    # Regime Interaction
    df['vol_volume_alignment'] = df['volatility_compression'] * df['volume_level']
    df['persistent_regime'] = df['vol_volume_alignment'] * df['volume_clustering']
    df['regime_momentum'] = df['persistent_regime'] / df['persistent_regime'].shift(1)
    
    # Final alpha factor combining all components
    alpha_factor = (
        df['enhanced_divergence'] * 0.25 +
        df['efficiency_momentum'] * 0.25 +
        df['enhanced_flow_factor'] * 0.25 +
        df['regime_momentum'] * 0.25
    )
    
    return alpha_factor
