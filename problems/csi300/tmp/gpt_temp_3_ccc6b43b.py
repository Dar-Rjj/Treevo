import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Volatility Structure
    df['true_range'] = df['high'] - df['low']
    df['true_range_momentum'] = df['true_range'] / df['true_range'].shift(5)
    
    df['gap_adjusted_volatility'] = np.abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    df['volatility_persistence'] = df['true_range'] / ((df['true_range'].shift(1) + df['true_range'].shift(2)) / 2)
    
    # Entropy Analysis
    def calculate_price_path_entropy(window):
        returns = window['close'] / window['close'].shift(1) - 1
        entropy_components = (returns ** 2) * np.log(np.abs(returns) + 1)
        return -entropy_components.sum()
    
    df['price_path_entropy'] = df['close'].rolling(window=5).apply(
        lambda x: calculate_price_path_entropy(pd.DataFrame({'close': x})), raw=False
    )
    
    df['volume_price_product'] = df['volume'] * np.abs(df['close'] - df['close'].shift(1))
    df['volume_price_entropy'] = df['volume_price_product'] / df['volume_price_product'].rolling(window=5).sum()
    
    df['amount_ratio'] = df['amount'] / df['amount'].rolling(window=5).sum()
    df['amount_distribution_entropy'] = df['amount_ratio'] * np.log(df['amount_ratio'])
    
    # Fractal Microstructure
    df['multi_scale_range_ratio'] = df['true_range'] / ((df['true_range'].shift(2) + df['true_range'].shift(4)) / 2)
    
    def calculate_price_fractal_dimension(window):
        price_changes = np.abs(window['close'].diff().dropna())
        if len(price_changes) == 0 or (df.loc[window.index[-1], 'high'] - df.loc[window.index[-1], 'low']) <= 0:
            return np.nan
        return np.log(price_changes.sum()) / np.log(df.loc[window.index[-1], 'high'] - df.loc[window.index[-1], 'low'])
    
    df['price_fractal_dimension'] = df['close'].rolling(window=5).apply(
        lambda x: calculate_price_fractal_dimension(pd.DataFrame({'close': x, 'high': df.loc[x.index, 'high'], 'low': df.loc[x.index, 'low']})), raw=False
    )
    
    df['volume_fractality'] = df['volume'] / ((df['volume'].shift(2) + df['volume'].shift(4)) / 2)
    
    # Regime Adaptive Patterns
    df['volatility_entropy_coupling'] = df['true_range_momentum'] * df['price_path_entropy']
    df['gap_fractal_divergence'] = df['gap_adjusted_volatility'] * df['multi_scale_range_ratio']
    df['volume_entropy_asymmetry'] = df['volume_price_entropy'] * df['amount_distribution_entropy']
    
    # Market State Detection
    df['high_volatility_regime'] = ((df['true_range_momentum'] > 1.5) & (df['volatility_persistence'] > 1.2)).astype(float)
    df['entropy_expansion'] = (df['price_path_entropy'] > df['price_path_entropy'].rolling(window=5).mean()).astype(float)
    df['fractal_breakdown'] = ((df['price_fractal_dimension'] < 0.8) & (df['volume_fractality'] > 1.1)).astype(float)
    
    # Alpha Synthesis
    df['volatility_entropy_factor'] = df['volatility_entropy_coupling'] * df['high_volatility_regime'] * df['volume_fractality']
    df['gap_fractal_factor'] = df['gap_fractal_divergence'] * df['fractal_breakdown'] * df['amount_distribution_entropy']
    df['adaptive_entropy_factor'] = df['volume_entropy_asymmetry'] * df['entropy_expansion'] * df['price_fractal_dimension']
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['volatility_entropy_factor'] + df['gap_fractal_factor'] + df['adaptive_entropy_factor']) / 3
    
    return df['alpha_factor']
