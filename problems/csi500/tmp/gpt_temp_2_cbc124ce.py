import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate momentum components
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    
    # Calculate regime indicators
    df['current_volatility'] = (df['high'] - df['low']) / df['close']
    df['current_participation'] = df['amount'] / df['amount'].shift(1)
    
    # Determine regimes
    df['high_volatility_regime'] = df['current_volatility'] > 0.02
    df['high_participation_regime'] = df['current_participation'] > 1.5
    
    # Calculate divergence signals
    df['primary_signal'] = df['price_momentum_10d'] - df['volume_momentum_10d']
    df['confirmation_signal'] = df['price_momentum_5d'] - df['volume_momentum_5d']
    
    # Initialize final factor
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Apply regime-adaptive combination
    for idx in df.index:
        if df.loc[idx, 'high_volatility_regime']:
            # High Volatility Regime: 70% Primary + 30% Confirmation
            factor_values.loc[idx] = (0.7 * df.loc[idx, 'primary_signal'] + 
                                    0.3 * df.loc[idx, 'confirmation_signal'])
        elif df.loc[idx, 'high_participation_regime']:
            # High Participation Regime: 40% Primary + 60% Confirmation
            factor_values.loc[idx] = (0.4 * df.loc[idx, 'primary_signal'] + 
                                    0.6 * df.loc[idx, 'confirmation_signal'])
        else:
            # Normal Regime: 60% Primary + 40% Confirmation
            factor_values.loc[idx] = (0.6 * df.loc[idx, 'primary_signal'] + 
                                    0.4 * df.loc[idx, 'confirmation_signal'])
    
    return factor_values
