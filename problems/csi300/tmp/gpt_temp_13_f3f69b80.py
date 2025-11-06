import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Multi-Scale Momentum Ratios
    df['short_term_momentum'] = (df['close'] / df['close'].shift(1)) / (df['close'].shift(1) / df['close'].shift(2)) - 1
    df['medium_term_momentum'] = (df['close'] / df['close'].shift(4)) / (df['close'].shift(4) / df['close'].shift(8)) - 1
    df['long_term_momentum'] = (df['close'] / df['close'].shift(7)) / (df['close'].shift(7) / df['close'].shift(21)) - 1
    
    # Cross-Scale Divergence
    df['short_medium_divergence'] = df['short_term_momentum'] - df['medium_term_momentum']
    df['medium_long_divergence'] = df['medium_term_momentum'] - df['long_term_momentum']
    
    # Volatility-Efficiency Components
    df['daily_range_efficiency'] = (df['high'] - df['low']) / np.maximum(
        df['high'] - df['low'], 
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Volatility Regime (4-day rolling std of 1-day returns)
    returns = df['close'] / df['close'].shift(1) - 1
    df['volatility_regime'] = (df['high'] - df['low']) / returns.rolling(window=4, min_periods=1).std()
    
    # Liquidity Dynamics
    df['volume_efficiency'] = df['volume'] / np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['amount_intensity'] = df['amount'] / df['volume']
    
    # Momentum-Efficiency Integration
    df['vol_weighted_short_momentum'] = df['short_term_momentum'] * df['volatility_regime']
    df['vol_weighted_medium_momentum'] = df['medium_term_momentum'] * df['volatility_regime']
    
    # Structural Break Detection
    df['min_low_4d'] = df['low'].rolling(window=4, min_periods=1).min()
    df['max_high_4d'] = df['high'].rolling(window=4, min_periods=1).max()
    df['penetration_strength'] = ((df['close'] - df['min_low_4d']) / 
                                 (df['max_high_4d'] - df['min_low_4d'])) * df['volume']
    
    df['break_persistence'] = ((df['close'] / df['close'].shift(5) - 1) * 
                              (df['close'] / df['close'].shift(20) - 1) * 
                              np.sign(df['close'] - df['close'].shift(1)))
    
    # Momentum-Break Divergence
    df['acceleration_divergence'] = df['short_medium_divergence'] * df['medium_long_divergence']
    df['regime_break_score'] = ((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))) * \
                              (np.abs(df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1)))
    
    # Cross-Scale Information Synthesis
    # Volume-Return Correlation Dynamics
    volume_returns_corr_short = []
    volume_returns_corr_medium = []
    
    for i in range(len(df)):
        if i >= 2:
            vol_window = df['volume'].iloc[i-2:i+1]
            ret_window = (df['close'].iloc[i-2:i+1] / df['close'].iloc[i-3:i] - 1).values
            if len(vol_window) == len(ret_window) and len(vol_window) >= 2:
                corr = np.corrcoef(vol_window, ret_window)[0, 1]
                volume_returns_corr_short.append(corr if not np.isnan(corr) else 0)
            else:
                volume_returns_corr_short.append(0)
        else:
            volume_returns_corr_short.append(0)
            
        if i >= 4:
            vol_window = df['volume'].iloc[i-4:i+1]
            ret_window = (df['close'].iloc[i-4:i+1] / df['close'].iloc[i-5:i] - 1).values
            if len(vol_window) == len(ret_window) and len(vol_window) >= 2:
                corr = np.corrcoef(vol_window, ret_window)[0, 1]
                volume_returns_corr_medium.append(corr if not np.isnan(corr) else 0)
            else:
                volume_returns_corr_medium.append(0)
        else:
            volume_returns_corr_medium.append(0)
    
    df['volume_return_corr_short'] = volume_returns_corr_short
    df['volume_return_corr_medium'] = volume_returns_corr_medium
    
    # Liquidity-Momentum Interaction
    df['open_close_efficiency'] = ((df['close'] - df['open']) / (df['high'] - df['low'])) * \
                                 np.sign(df['volume'] / df['volume'].shift(5) - 1)
    
    df['volume_price_divergence'] = ((df['volume'] / df['volume'].shift(5) - 1) / 
                                    (df['close'] / df['close'].shift(5) - 1))
    
    # Composite Factor Construction
    # Core Signal Generation
    df['break_enhanced_momentum'] = df['vol_weighted_short_momentum'] * (1 + df['penetration_strength'])
    
    # Efficiency-Weighted Divergence
    conditions = [
        df['volume_price_divergence'] > 0,
        df['volume_price_divergence'] < 0
    ]
    choices = [
        -df['short_term_momentum'] * df['volume_efficiency'],
        df['short_term_momentum'] * df['volume_efficiency']
    ]
    df['efficiency_weighted_divergence'] = np.select(conditions, choices, default=0)
    
    # Final Alpha Factor
    df['primary_component'] = df['break_enhanced_momentum'] * (1 + df['regime_break_score'] / (1 + np.abs(df['regime_break_score'])))
    df['secondary_component'] = df['efficiency_weighted_divergence'] * df['amount_intensity'] * df['volume_return_corr_short']
    
    # Final composite factor
    alpha_factor = df['primary_component'] + df['secondary_component']
    
    return alpha_factor
