import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper functions for entropy and fractal calculations
    def calculate_entropy(series, window=5):
        """Calculate entropy of a series using rolling window"""
        returns = series.pct_change()
        rolling_std = returns.rolling(window=window).std()
        rolling_mean = returns.rolling(window=window).mean()
        z_scores = (returns - rolling_mean) / (rolling_std + 1e-8)
        probabilities = np.exp(-0.5 * z_scores ** 2) / (np.sqrt(2 * np.pi) * (rolling_std + 1e-8))
        entropy = -probabilities * np.log(probabilities + 1e-8)
        return entropy
    
    def calculate_fractal_dimension(series, window=10):
        """Calculate fractal dimension using Hurst exponent approximation"""
        log_returns = np.log(series / series.shift(1))
        ranges = []
        for i in range(window, len(series)):
            window_data = log_returns.iloc[i-window:i]
            if len(window_data.dropna()) < 2:
                ranges.append(np.nan)
                continue
            cumulative = window_data.cumsum()
            R = cumulative.max() - cumulative.min()
            S = window_data.std()
            if S == 0:
                ranges.append(np.nan)
            else:
                ranges.append(R / S)
        
        fractal = pd.Series(ranges, index=series.index[window:])
        return fractal.reindex(series.index).fillna(method='ffill')
    
    # Calculate entropy and fractal dimensions for price and volume
    data['price_entropy'] = calculate_entropy(data['close'])
    data['volume_entropy'] = calculate_entropy(data['volume'])
    data['amount_entropy'] = calculate_entropy(data['amount'])
    
    data['price_fractal'] = calculate_fractal_dimension(data['close'])
    data['volume_fractal'] = calculate_fractal_dimension(data['volume'])
    data['amount_fractal'] = calculate_fractal_dimension(data['amount'])
    
    # Calculate cascade ratios
    data['volatility_cascade_ratio'] = (data['high'] - data['low']).rolling(window=5).std() / \
                                      (data['high'] - data['low']).rolling(window=20).std()
    data['volume_cascade_ratio'] = data['volume'].rolling(window=5).std() / \
                                  data['volume'].rolling(window=20).std()
    
    # Intraday Entropy-Fractal Divergence
    data['opening_entropy_fractal'] = ((data['open'] - data['close'].shift(1)) * data['volume'] / 
                                      (data['high'] - data['low'] + 1e-8) * 
                                      data['price_entropy'] * data['price_fractal'])
    
    data['midday_entropy_fractal'] = (((data['high'] + data['low'])/2 - (data['open'] + data['close'])/2) * 
                                     data['volume'] / (data['high'] - data['low'] + 1e-8) * 
                                     data['volatility_cascade_ratio'] * data['price_fractal'])
    
    data['closing_entropy_fractal'] = ((data['close'] - (data['high'] + data['low'])/2) * data['volume'] / 
                                      (data['high'] - data['low'] + 1e-8) * 
                                      data['price_entropy'] * data['volatility_cascade_ratio'] * data['price_fractal'])
    
    # Multi-Timeframe Entropy-Fractal Divergence
    data['intraday_entropy_fractal_divergence'] = (data['opening_entropy_fractal'] + 
                                                  data['midday_entropy_fractal'] + 
                                                  data['closing_entropy_fractal'])
    
    data['short_term_entropy_fractal'] = (data['intraday_entropy_fractal_divergence'] - 
                                         data['intraday_entropy_fractal_divergence'].shift(1))
    
    data['medium_term_entropy_fractal'] = data['intraday_entropy_fractal_divergence'].rolling(window=5).mean()
    data['entropy_fractal_acceleration'] = data['short_term_entropy_fractal'] - data['medium_term_entropy_fractal']
    
    # Entropy-Volume Dispersion Shifts
    data['entropy_volume_dispersion_breakout'] = (
        ((data['volume'] / data['volume'].shift(1) - 1) - 
         (data['volume'].shift(1) / data['volume'].shift(2) - 1)) * 
        data['volume_entropy'] * data['volume_fractal']
    )
    
    data['entropy_volume_dispersion_momentum'] = (
        ((data['volume'] / data['volume'].shift(1) - 1) - 
         (data['volume'].shift(1) / data['volume'].shift(2) - 1)) * 
        ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * 
        data['volume_entropy'] * data['volume_fractal']
    )
    
    # Entropy-Price Dispersion Transitions
    data['entropy_volatility_dispersion'] = (
        ((data['high'] - data['low']) / data['close'].shift(1) - 
         (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(2)) * 
        data['volatility_cascade_ratio'] * data['price_fractal']
    )
    
    data['entropy_range_dispersion_breakout'] = (
        ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8) - 
         (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(2) - data['low'].shift(2) + 1e-8)) * 
        data['price_entropy'] * data['price_fractal']
    )
    
    # Fractal-Cascade Integration
    data['early_entropy_fractal_signals'] = (data['intraday_entropy_fractal_divergence'] * 
                                            data['entropy_volume_dispersion_breakout'])
    
    data['entropy_fractal_momentum_interaction'] = (data['entropy_fractal_acceleration'] * 
                                                   data['entropy_volatility_dispersion'])
    
    data['quality_entropy_fractal_integration'] = (data['entropy_volume_dispersion_breakout'] * 
                                                  data['closing_entropy_fractal'])
    
    # Dispersion-Asymmetric Patterns
    data['entropy_fractal_bullish'] = np.where(
        (data['close'] > data['open']) & (data['volume'] > data['volume'].shift(1)),
        data['closing_entropy_fractal'] * data['price_entropy'] * data['price_fractal'],
        0
    )
    
    data['entropy_fractal_bearish'] = np.where(
        (data['close'] < data['open']) & (data['volume'] > data['volume'].shift(1)),
        data['closing_entropy_fractal'] * data['volatility_cascade_ratio'] * data['price_fractal'],
        0
    )
    
    data['entropy_volume_dispersion_asymmetry'] = (
        ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8) - 
         (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * 
        (data['volume'] / data['volume'].shift(1) - 1) * 
        data['volume_entropy'] * data['volume_fractal']
    )
    
    # Adaptive Fractal-Cascade Construction
    data['high_entropy_fractal_volatility'] = (data['entropy_volume_dispersion_breakout'] * 
                                              data['entropy_fractal_bullish'])
    
    data['low_entropy_fractal_volatility'] = (data['entropy_volume_dispersion_momentum'] * 
                                             data['intraday_entropy_fractal_divergence'].rolling(window=3).std())
    
    data['entropy_fractal_transition_phase'] = (data['early_entropy_fractal_signals'] * 
                                               data['entropy_fractal_momentum_interaction'])
    
    # Fractal-Cascade Regime Classification
    high_fractal_mask = (
        (data['volatility_cascade_ratio'] > data['volume_cascade_ratio'] * 2.0) & 
        (data['price_entropy'] > data['volume_entropy']) & 
        (data['price_fractal'] > data['volume_fractal'])
    )
    
    low_fractal_mask = (
        (data['volatility_cascade_ratio'] < data['volume_cascade_ratio'] * 0.5) & 
        (data['price_entropy'] < data['volume_entropy']) & 
        (data['price_fractal'] < data['volume_fractal'])
    )
    
    transition_fractal_mask = (
        (abs(data['volatility_cascade_ratio'] - data['volume_cascade_ratio']) > 
         abs(data['amount'] / data['amount'].shift(2) - 1)) & 
        (abs(data['price_entropy'] - data['volume_entropy']) > 0.1) & 
        (abs(data['price_fractal'] - data['volume_fractal']) > 0.1)
    )
    
    # Dispersion-Flow Dynamics
    data['fractal_dispersion_flow'] = (
        (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
        (data['amount'] / data['amount'].shift(1) - 1) * 
        data['price_fractal'] * data['volume'] / data['volume'].shift(1) * 
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    data['amount_dispersion_compression'] = (
        ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8) - 
         (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * 
        (data['amount'] / data['amount'].shift(1) - 1) * 
        data['amount_fractal'] * abs(data['volume'] - data['volume'].shift(1)) / data['volume']
    )
    
    data['dispersion_flow_divergence'] = data['fractal_dispersion_flow'] - data['amount_dispersion_compression']
    
    # Dispersion-Amount Regime Dynamics
    data['volume_dispersion_momentum'] = (
        ((data['amount'] / data['amount'].shift(1) - 1) - 
         (data['amount'].shift(1) / data['amount'].shift(2) - 1)) * 
        ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * 
        data['amount_fractal'] * 
        (data['volume'] / data['volume'].shift(1) - data['volume'].shift(1) / data['volume'].shift(2))
    )
    
    data['dispersion_amount_price_divergence'] = (
        data['volume_dispersion_momentum'] - 
        data['entropy_fractal_acceleration'] * np.sign(data['entropy_fractal_acceleration']) * 
        (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 1e-8)
    )
    
    data['dispersion_amount_convergence'] = (
        data['volume_dispersion_momentum'] * data['entropy_fractal_acceleration'] * 
        (data['close'] - data['open'])
    )
    
    # Final Fractal Entropy-Cascade Alpha Synthesis
    data['core_entropy_fractal_factor'] = (data['entropy_volume_dispersion_asymmetry'] * 
                                          data['quality_entropy_fractal_integration'])
    
    data['dispersion_momentum_component'] = (data['fractal_dispersion_flow'] * 
                                           data['volume_dispersion_momentum'] * 
                                           data['dispersion_flow_divergence'])
    
    # Regime-Adaptive Fractal Synthesis
    alpha = pd.Series(index=data.index, dtype=float)
    
    # High Fractal-Cascade
    alpha[high_fractal_mask] = (
        data['core_entropy_fractal_factor'] * 
        data['high_entropy_fractal_volatility'] * 
        data['dispersion_amount_convergence']
    )[high_fractal_mask]
    
    # Low Fractal-Cascade
    alpha[low_fractal_mask] = (
        data['core_entropy_fractal_factor'] * 
        data['low_entropy_fractal_volatility'] * 
        data['dispersion_amount_price_divergence']
    )[low_fractal_mask]
    
    # Transition Fractal-Cascade
    alpha[transition_fractal_mask] = (
        data['core_entropy_fractal_factor'] * 
        data['entropy_fractal_transition_phase'] * 
        data['dispersion_flow_divergence']
    )[transition_fractal_mask]
    
    # Fill remaining values with weighted average
    remaining_mask = ~(high_fractal_mask | low_fractal_mask | transition_fractal_mask)
    alpha[remaining_mask] = (
        data['core_entropy_fractal_factor'] * 
        data['dispersion_momentum_component']
    )[remaining_mask]
    
    # Clean and normalize
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(method='ffill').fillna(0)
    
    return alpha
