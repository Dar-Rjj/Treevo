import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Regime-Adaptive Price-Volume Momentum Divergence Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Calculation
    for period in [3, 5, 10, 20, 50]:
        data[f'price_mom_{period}d'] = data['close'] / data['close'].shift(period) - 1
    
    # Volume Momentum Calculation
    for period in [3, 5, 10, 20, 50]:
        data[f'volume_mom_{period}d'] = data['volume'] / data['volume'].shift(period) - 1
    
    # Regime Classification
    # Market Participation Regime
    data['amount_trend_ratio'] = data['amount'] / data['amount'].shift(20)
    data['amount_momentum'] = data['amount'] / data['amount'].shift(5) - 1
    
    # Participation regime classification
    conditions_participation = [
        (data['amount_trend_ratio'] > 1.15) & (data['amount_momentum'] > 0.05),
        (data['amount_trend_ratio'] >= 0.85) & (data['amount_trend_ratio'] <= 1.15),
        (data['amount_trend_ratio'] < 0.85) & (data['amount_momentum'] < -0.05)
    ]
    choices_participation = ['high', 'normal', 'low']
    data['participation_regime'] = np.select(conditions_participation, choices_participation, default='normal')
    
    # Price Volatility Regime
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    data['volatility_ratio'] = data['daily_range'] / data['range_5d_avg']
    
    # Volatility regime classification
    conditions_volatility = [
        data['volatility_ratio'] > 1.25,
        (data['volatility_ratio'] >= 0.75) & (data['volatility_ratio'] <= 1.25),
        data['volatility_ratio'] < 0.75
    ]
    choices_volatility = ['high', 'normal', 'low']
    data['volatility_regime'] = np.select(conditions_volatility, choices_volatility, default='normal')
    
    # Momentum Divergence Analysis
    divergence_columns = []
    for period in [3, 5, 10, 20, 50]:
        div_col = f'div_{period}d'
        data[div_col] = data[f'price_mom_{period}d'] - data[f'volume_mom_{period}d']
        divergence_columns.append(div_col)
    
    # Divergence Quality Assessment
    divergence_matrix = data[divergence_columns]
    data['direction_consistency'] = (divergence_matrix > 0).sum(axis=1)
    
    # Magnitude consistency calculation
    data['magnitude_consistency'] = 1 - (divergence_matrix.std(axis=1) / divergence_matrix.abs().mean(axis=1))
    data['magnitude_consistency'] = data['magnitude_consistency'].fillna(0).clip(0, 1)
    
    # Regime-Adaptive Weighting Scheme
    # Define base weights for each regime
    participation_weights = {
        'high': [0.4, 0.3, 0.15, 0.1, 0.05],
        'normal': [0.25, 0.25, 0.2, 0.15, 0.15],
        'low': [0.05, 0.15, 0.2, 0.3, 0.3]
    }
    
    volatility_weights = {
        'high': [0.35, 0.3, 0.2, 0.1, 0.05],
        'normal': [0.2, 0.2, 0.2, 0.2, 0.2],
        'low': [0.05, 0.1, 0.2, 0.3, 0.35]
    }
    
    # Calculate final weights for each row
    final_weights = []
    for idx, row in data.iterrows():
        if pd.isna(row['participation_regime']) or pd.isna(row['volatility_regime']):
            final_weights.append([0.2, 0.2, 0.2, 0.2, 0.2])
            continue
            
        part_weights = participation_weights[row['participation_regime']]
        vol_weights = volatility_weights[row['volatility_regime']]
        
        # Base weights as average
        base_weights = [(p + v) / 2 for p, v in zip(part_weights, vol_weights)]
        
        # Quality adjustment
        quality_adj = row['magnitude_consistency'] * 0.2
        
        # Final weights
        final_row_weights = [w * (1 + quality_adj) for w in base_weights]
        
        # Normalize to sum to 1
        weight_sum = sum(final_row_weights)
        if weight_sum > 0:
            final_row_weights = [w / weight_sum for w in final_row_weights]
        
        final_weights.append(final_row_weights)
    
    # Signal Construction
    signals = []
    for idx, row in data.iterrows():
        if pd.isna(row[divergence_columns]).any():
            signals.append(0)
            continue
        
        weights = final_weights[data.index.get_loc(idx)]
        
        # Core divergence score
        weighted_sum = sum(w * row[div_col] for w, div_col in zip(weights, divergence_columns))
        
        # Regime intensity
        regime_intensity = abs(row['amount_trend_ratio'] - 1) + abs(row['volatility_ratio'] - 1)
        
        # Signal enhancement
        regime_confidence = min(regime_intensity / 0.5, 1.0)
        enhanced_signal = weighted_sum * (1 + regime_confidence)
        
        # Quality boost
        quality_boost = row['direction_consistency'] * 0.1
        final_signal = enhanced_signal * (1 + quality_boost)
        
        signals.append(final_signal)
    
    # Return the factor values as a pandas Series
    return pd.Series(signals, index=data.index, name='regime_adaptive_divergence')
