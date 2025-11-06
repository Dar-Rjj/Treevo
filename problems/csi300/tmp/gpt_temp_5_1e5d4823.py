import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Momentum Breakout with Volume-Efficiency Regime Switching
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for momentum calculations
    df['return'] = df['close'] / df['close'].shift(1) - 1
    df['return_2d'] = df['close'] / df['close'].shift(2) - 1
    df['return_5d'] = df['close'] / df['close'].shift(5) - 1
    df['return_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Peer group momentum (simulated using rolling correlation with market proxy)
    df['market_return'] = df['return'].rolling(window=20, min_periods=10).mean()
    df['peer_correlation'] = df['return'].rolling(window=20).corr(df['market_return'])
    
    # Multi-timeframe relative momentum
    df['ultra_short_momentum'] = df['return_2d'] - df['market_return'].rolling(window=2).mean()
    df['short_term_momentum'] = df['return_5d'] - df['market_return'].rolling(window=5).median()
    df['medium_term_rank'] = df['return_20d'].rolling(window=20).rank(pct=True)
    
    # Momentum acceleration patterns
    df['momentum_acceleration'] = (df['ultra_short_momentum'] - df['short_term_momentum']) + \
                                 (df['short_term_momentum'] - df['medium_term_rank'])
    
    # Cross-asset consistency
    df['momentum_consistency'] = ((df['ultra_short_momentum'] > 0) & 
                                 (df['short_term_momentum'] > 0) & 
                                 (df['medium_term_rank'] > 0.5)).astype(int) - \
                                ((df['ultra_short_momentum'] < 0) & 
                                 (df['short_term_momentum'] < 0) & 
                                 (df['medium_term_rank'] < 0.5)).astype(int)
    
    # Peer divergence
    df['peer_divergence'] = ((df['ultra_short_momentum'] > 0) & 
                            (df['market_return'].rolling(window=2).mean() < 0)).astype(int)
    
    # Volume efficiency metrics
    df['volume_momentum'] = df['volume'] / df['volume'].rolling(window=5).mean() - 1
    df['amihud_ratio'] = abs(df['return']) / (df['amount'] + 1e-8)
    df['range_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Volatility patterns
    df['range_5d'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close']
    df['range_20d'] = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close']
    df['volatility_ratio'] = df['range_5d'] / (df['range_20d'] + 1e-8)
    
    # Regime classification
    df['high_efficiency'] = ((df['amihud_ratio'] < df['amihud_ratio'].rolling(window=20).median()) &
                            (df['range_efficiency'] > df['range_efficiency'].rolling(window=20).median()) &
                            (df['volume_momentum'] > 0)).astype(int)
    
    df['low_efficiency'] = ((df['amihud_ratio'] > df['amihud_ratio'].rolling(window=20).quantile(0.7)) &
                           (df['range_efficiency'] < df['range_efficiency'].rolling(window=20).quantile(0.3)) &
                           (df['volume_momentum'] < 0)).astype(int)
    
    df['compression_regime'] = (df['volatility_ratio'] < 0.7).astype(int)
    df['expansion_regime'] = (df['volatility_ratio'] > 1.3).astype(int)
    
    # Regime-adaptive breakout signals
    for i in range(len(df)):
        if i < 20:  # Ensure enough data for calculations
            result.iloc[i] = 0
            continue
            
        row = df.iloc[i]
        
        # Base momentum factor
        momentum_factor = (row['momentum_acceleration'] * 0.4 + 
                          row['momentum_consistency'] * 0.3 + 
                          row['peer_divergence'] * 0.3)
        
        # Regime adjustments
        if row['high_efficiency']:
            # Emphasize momentum divergence, reduce volume confirmation
            regime_factor = momentum_factor * 1.2 + row['peer_divergence'] * 0.5
            
        elif row['low_efficiency']:
            # Require volume acceleration, focus on momentum consistency
            regime_factor = momentum_factor * 0.8 + row['volume_momentum'] * 0.4 + row['momentum_consistency'] * 0.6
            
        elif row['compression_regime']:
            # Amplify breakout signals, use volatility ratio as multiplier
            volatility_multiplier = 1.0 + (0.7 - row['volatility_ratio']) * 2
            regime_factor = momentum_factor * volatility_multiplier
            
        elif row['expansion_regime']:
            # Blend efficiency and momentum, apply volatility filters
            efficiency_score = (1 - row['amihud_ratio'] / (row['amihud_ratio'].rolling(window=20).max() + 1e-8)) + \
                              row['range_efficiency']
            regime_factor = momentum_factor * 0.7 + efficiency_score * 0.3
            
        else:
            # Normal regime
            regime_factor = momentum_factor
        
        # Final factor with volume confirmation
        volume_confirmation = 1.0 + row['volume_momentum'] * 0.2
        final_factor = regime_factor * volume_confirmation
        
        result.iloc[i] = final_factor
    
    # Clean up intermediate columns
    cols_to_drop = ['return', 'return_2d', 'return_5d', 'return_20d', 'market_return', 
                   'peer_correlation', 'ultra_short_momentum', 'short_term_momentum', 
                   'medium_term_rank', 'momentum_acceleration', 'momentum_consistency',
                   'peer_divergence', 'volume_momentum', 'amihud_ratio', 'range_efficiency',
                   'range_5d', 'range_20d', 'volatility_ratio', 'high_efficiency',
                   'low_efficiency', 'compression_regime', 'expansion_regime']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return result
