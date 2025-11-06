import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Analysis
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_decay'] = (data['close'] / data['close'].shift(3) - 1) / (data['close'].shift(3) / data['close'].shift(6) - 1)
    data['price_acceleration'] = ((data['close'] - data['close'].shift(5)) / data['close'].shift(5)) - ((data['close'].shift(5) - data['close'].shift(10)) / data['close'].shift(10))
    
    # Volume & Microstructure Analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3)
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(3)) / (data['volume'].shift(3) / data['volume'].shift(6))
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['effective_volume'] = data['volume'] * (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volatility & Market Depth
    data['realized_volatility'] = (data['high'] - data['low']) / data['close']
    data['bid_ask_proxy'] = (data['high'] - data['low']) / ((data['open'] + data['close']) / 2).replace(0, np.nan)
    data['depth_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['intraday_noise_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Divergence Detection
    data['price_change_3d'] = data['close'].pct_change(3)
    data['volume_change_3d'] = data['volume'].pct_change(3)
    data['correlation_divergence'] = 1 - data['price_change_3d'].rolling(window=3, min_periods=1).corr(data['volume_change_3d'])
    data['direction_divergence'] = data['momentum_decay'] * data['volume_acceleration']
    data['microstructure_divergence'] = data['price_acceleration'] * data['volume_acceleration'] * data['realized_volatility']
    data['depth_momentum_divergence'] = data['depth_pressure'] * data['volume_concentration'] * data['bid_ask_proxy']
    
    # Noise Filtering & Quality Assessment
    data['microstructure_noise_score'] = data['bid_ask_proxy'] * data['intraday_noise_ratio']
    data['execution_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Flow consistency calculation
    data['price_change_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['same_sign_count'] = data['price_change_sign'].rolling(window=5, min_periods=1).apply(lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0)
    data['volume_autocorr'] = data['volume'].rolling(window=5, min_periods=1).apply(lambda x: x.corr(x.shift(1)) if len(x.dropna()) > 1 else 0)
    data['flow_consistency'] = data['same_sign_count'] * data['volume_autocorr']
    
    data['noise_adjusted_momentum'] = ((data['close'] - data['close'].shift(1)) / data['microstructure_noise_score'].replace(0, np.nan)) * np.sign(data['close'] - data['close'].shift(1))
    
    # Adaptive Signal Construction
    data['volatility_regime_score'] = data['realized_volatility'] * (data['realized_volatility'] / data['realized_volatility'].shift(1)).replace([np.inf, -np.inf], np.nan)
    data['volume_confirmed_momentum'] = (data['close'] - data['close'].shift(1)) * data['volume'] * np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['filtered_momentum_score'] = data['noise_adjusted_momentum'] * data['volume_confirmed_momentum']
    data['adaptive_divergence'] = data['microstructure_divergence'] * (1 - data['volatility_regime_score']) + data['depth_momentum_divergence'] * data['volatility_regime_score']
    
    # Composite Alpha Generation
    data['core_divergence'] = data['correlation_divergence'] * data['direction_divergence'] * data['adaptive_divergence']
    data['quality_enhancement'] = data['core_divergence'] * data['execution_efficiency'] * data['flow_consistency']
    data['volume_confirmation'] = data['quality_enhancement'] * data['effective_volume']
    data['amount_ratio'] = data['amount'] / data['amount'].rolling(window=10, min_periods=1).mean()
    data['final_alpha'] = data['volume_confirmation'] * data['filtered_momentum_score'] * data['amount_ratio']
    
    # Return the final alpha factor
    return data['final_alpha']
