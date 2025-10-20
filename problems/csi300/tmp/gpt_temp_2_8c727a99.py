import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate returns and volume metrics
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Multi-Timeframe Momentum-Efficiency
    # Calculate momentum for different timeframes
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_21d'] = data['close'] / data['close'].shift(21) - 1
    
    # Calculate price change per unit volume (efficiency)
    data['efficiency_3d'] = data['momentum_3d'] / (data['volume'].rolling(3).mean() + 1e-8)
    data['efficiency_8d'] = data['momentum_8d'] / (data['volume'].rolling(8).mean() + 1e-8)
    data['efficiency_21d'] = data['momentum_21d'] / (data['volume'].rolling(21).mean() + 1e-8)
    
    # Volume Structural Analysis
    # Volume momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    
    # Volume concentration (simulated - using rolling windows)
    # Assuming first 30 minutes has higher volume concentration
    data['volume_std_5d'] = data['volume'].rolling(5).std()
    data['volume_mean_5d'] = data['volume'].rolling(5).mean()
    data['volume_concentration'] = data['volume_std_5d'] / (data['volume_mean_5d'] + 1e-8)
    
    # Volatility Regime Classification
    # Calculate ATR (Average True Range)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_20d'] = data['tr'].rolling(20).mean()
    data['atr_60d'] = data['tr'].rolling(60).mean()
    data['atr_ratio'] = data['atr_20d'] / (data['atr_60d'] + 1e-8)
    
    # Classify volatility regimes
    data['vol_regime'] = 'normal'
    data.loc[data['atr_ratio'] > 1.2, 'vol_regime'] = 'high'
    data.loc[data['atr_ratio'] < 0.8, 'vol_regime'] = 'low'
    
    # Calculate divergence between momentum and efficiency
    data['momentum_divergence_3d'] = data['momentum_3d'] - data['efficiency_3d'].rolling(5).mean()
    data['momentum_divergence_8d'] = data['momentum_8d'] - data['efficiency_8d'].rolling(8).mean()
    data['momentum_divergence_21d'] = data['momentum_21d'] - data['efficiency_21d'].rolling(13).mean()
    
    # Combined momentum-efficiency score
    data['combined_momentum_eff'] = (
        data['momentum_3d'].rank(pct=True) * 0.3 +
        data['momentum_8d'].rank(pct=True) * 0.4 +
        data['momentum_21d'].rank(pct=True) * 0.3
    ) - (
        data['efficiency_3d'].rank(pct=True) * 0.3 +
        data['efficiency_8d'].rank(pct=True) * 0.4 +
        data['efficiency_21d'].rank(pct=True) * 0.3
    )
    
    # Volume structure score
    data['volume_structure'] = (
        data['volume_momentum_5d'].rank(pct=True) * 0.6 +
        data['volume_momentum_10d'].rank(pct=True) * 0.4 -
        data['volume_concentration'].rank(pct=True)
    )
    
    # Regime-Adaptive Integration
    factor_values = []
    
    for idx, row in data.iterrows():
        if pd.isna(row['vol_regime']) or pd.isna(row['combined_momentum_eff']) or pd.isna(row['volume_structure']):
            factor_values.append(np.nan)
            continue
            
        if row['vol_regime'] == 'high':
            # High volatility: emphasize divergence and concentration
            factor = (
                row['combined_momentum_eff'] * 0.4 +
                row['momentum_divergence_3d'] * 0.3 +
                row['momentum_divergence_8d'] * 0.2 +
                row['volume_concentration'] * 0.1
            )
        elif row['vol_regime'] == 'low':
            # Low volatility: focus on persistence
            factor = (
                row['combined_momentum_eff'] * 0.3 +
                row['momentum_21d'] * 0.4 +
                row['volume_momentum_10d'] * 0.3
            )
        else:
            # Normal volatility: balanced alignment
            factor = (
                row['combined_momentum_eff'] * 0.5 +
                row['volume_structure'] * 0.3 +
                (row['momentum_divergence_8d'] + row['momentum_divergence_21d']) * 0.2
            )
        
        factor_values.append(factor)
    
    # Create output series
    result = pd.Series(factor_values, index=data.index, name='factor')
    
    return result
