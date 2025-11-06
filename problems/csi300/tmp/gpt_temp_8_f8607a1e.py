import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Horizon Range Momentum Analysis
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) * (data['high'] - data['low'])
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(10)) * (data['high'] - data['low'])
    data['long_term_momentum'] = (data['close'] - data['close'].shift(30)) * (data['high'] - data['low'])
    
    # Momentum consistency assessment
    data['momentum_divergence'] = (
        np.sign(data['short_term_momentum']) + 
        np.sign(data['medium_term_momentum']) + 
        np.sign(data['long_term_momentum'])
    )
    
    # Intraday Reversal-Momentum Divergence Analysis
    data['intraday_reversal'] = -1 * (data['close'] - data['open']) / data['open']
    data['intraday_momentum'] = (data['high'] - data['low']) / data['open']
    data['long_term_price_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_divergence_intraday'] = data['intraday_momentum'] - data['long_term_price_momentum']
    data['intraday_signal'] = 0.6 * data['intraday_reversal'] + 0.4 * data['momentum_divergence_intraday']
    
    # Multi-Timeframe Volatility Adjustment
    data['returns'] = data['close'].pct_change()
    data['historical_volatility'] = data['returns'].rolling(window=10, min_periods=5).std()
    data['intraday_volatility'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['combined_volatility'] = data['historical_volatility'] * data['intraday_volatility']
    
    # Volume-Pressure Accumulation System
    data['volume_ma'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_surprise'] = data['volume'] - data['volume_ma']
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    
    data['reversal_pressure'] = np.sign(data['close'] - data['open']) * (data['high'] - data['low'])
    data['pressure_intensity'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volume-weighted pressure accumulation with time decay
    decay_weights = np.array([0.5, 0.25, 0.15, 0.07, 0.03])  # 5-day decay
    volume_pressure = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            pressure_sum = 0
            for j, weight in enumerate(decay_weights):
                idx = i - (4 - j)
                if not pd.isna(window_data.iloc[j]['reversal_pressure']):
                    pressure_sum += weight * window_data.iloc[j]['reversal_pressure'] * window_data.iloc[j]['volume_ratio']
            volume_pressure.iloc[i] = pressure_sum
        else:
            volume_pressure.iloc[i] = 0
    
    data['volume_pressure_accumulation'] = volume_pressure
    
    # Multi-Timeframe Reversal Efficiency
    data['short_term_momentum_5d'] = data['close'].shift(1) - data['close'].shift(5)
    data['medium_term_momentum_20d'] = data['close'].shift(1) - data['close'].shift(20)
    
    data['opening_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_reversal_efficiency'] = np.sign(data['close'] - data['open']) * np.sign(data['high'] - data['low'])
    data['price_amplitude'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Efficiency-weighted reversal signals
    data['efficiency_weight'] = data['opening_efficiency'] * data['pressure_intensity']
    
    # Volume-Volatility Confirmation Layer
    data['volume_weighted_volatility'] = data['volume'] * (data['high'] - data['low']) / data['open']
    data['volatility_adjusted_volume'] = data['volume'] / ((data['high'] - data['low']) / data['open'])
    
    # Volume-volatility signal enhancement
    data['volume_volatility_alignment'] = np.corrcoef(
        data['volume'].rolling(window=5).mean().fillna(0),
        data['intraday_volatility'].rolling(window=5).mean().fillna(0)
    )[0, 1] if len(data) > 5 else 0
    
    # Composite Alpha Generation
    # Volatility-weighted multi-scale signal
    volatility_weighted_momentum = (
        0.4 * data['short_term_momentum'] / (data['combined_volatility'] + 1e-8) +
        0.35 * data['medium_term_momentum'] / (data['combined_volatility'] + 1e-8) +
        0.25 * data['long_term_momentum'] / (data['combined_volatility'] + 1e-8)
    )
    
    # Incorporate intraday signals with volume efficiency
    intraday_component = data['intraday_signal'] * data['volume_ratio'] * data['efficiency_weight']
    
    # Volume-pressure adjustment
    pressure_component = data['volume_pressure_accumulation'] * np.sign(data['intraday_reversal'])
    
    # Volume-volatility confirmation
    volatility_confirmation = data['volume_volatility_alignment'] * data['volume_weighted_volatility']
    
    # Final composite alpha
    alpha = (
        0.35 * volatility_weighted_momentum +
        0.25 * intraday_component +
        0.20 * pressure_component +
        0.20 * volatility_confirmation
    )
    
    # Apply momentum consistency filter
    alpha = alpha * (1 - 0.2 * np.abs(data['momentum_divergence']))
    
    # Remove any potential future-looking data and return
    alpha_clean = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_clean
