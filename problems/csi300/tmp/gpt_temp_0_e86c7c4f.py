import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Multi-timeframe momentum divergence
    data['short_momentum'] = data['close'].pct_change(periods=3)
    data['medium_momentum'] = data['close'].pct_change(periods=10)
    data['momentum_divergence'] = data['short_momentum'] - data['medium_momentum']
    
    # Volatility regime calculation
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['volatility_regime'] = data['volatility_20d'] / data['volatility_20d'].rolling(window=60).mean()
    
    # Volume-weighted price acceleration
    data['price_acceleration'] = data['close'].pct_change().diff()
    data['volume_acceleration'] = data['volume'].pct_change().diff()
    data['current_liquidity'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['avg_liquidity_10d'] = data['current_liquidity'].rolling(window=10).mean()
    data['liquidity_deviation'] = data['current_liquidity'] / data['avg_liquidity_10d']
    
    # Dynamic support/resistance levels
    data['resistance_20d'] = data['high'].rolling(window=20).max()
    data['support_20d'] = data['low'].rolling(window=20).min()
    data['break_efficiency'] = np.where(
        data['close'] > data['resistance_20d'].shift(1),
        (data['close'] - data['resistance_20d'].shift(1)) / (data['high'] - data['low']),
        np.where(
            data['close'] < data['support_20d'].shift(1),
            (data['support_20d'].shift(1) - data['close']) / (data['high'] - data['low']),
            0
        )
    )
    
    # Asymmetric volatility
    data['returns'] = data['close'].pct_change()
    data['upside_vol'] = data['returns'].where(data['returns'] > 0, 0).rolling(window=20).std()
    data['downside_vol'] = data['returns'].where(data['returns'] < 0, 0).rolling(window=20).std()
    data['vol_asymmetry'] = data['upside_vol'] / data['downside_vol'].replace(0, np.nan)
    data['return_skewness'] = data['returns'].rolling(window=20).skew()
    
    # Cumulative order imbalance
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['money_flow'] = data['typical_price'] * data['volume']
    data['daily_imbalance'] = np.where(
        data['close'] > data['close'].shift(1),
        data['money_flow'],
        np.where(data['close'] < data['close'].shift(1), -data['money_flow'], 0)
    )
    data['cumulative_imbalance'] = data['daily_imbalance'].rolling(window=5).sum()
    
    # Price-volume convergence
    data['price_trend'] = data['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['volume_trend'] = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['price_volume_convergence'] = np.sign(data['price_trend']) * np.sign(data['volume_trend']) * \
                                      (abs(data['price_trend']) * abs(data['volume_trend'])) ** 0.5
    
    # Combine all components with regime-adaptive weights
    # Regime-adaptive momentum divergence
    momentum_factor = data['momentum_divergence'] * np.where(
        data['volatility_regime'] > 1.2, 
        data['volatility_regime'], 
        np.where(data['volatility_regime'] < 0.8, 0.5, 1.0)
    )
    
    # Volume-weighted acceleration with liquidity adjustment
    acceleration_factor = (data['price_acceleration'] * 0.7 + data['volume_acceleration'] * 0.3) * \
                         data['liquidity_deviation']
    
    # Break efficiency with volume confirmation
    volume_surge = data['volume'] / data['volume'].rolling(window=10).mean()
    break_factor = data['break_efficiency'] * volume_surge
    
    # Asymmetric volatility response with skewness adjustment
    volatility_factor = data['vol_asymmetry'] * (1 + data['return_skewness'] * 0.1)
    
    # Cumulative imbalance with persistence
    sign_persistence = data['daily_imbalance'].rolling(window=3).apply(
        lambda x: 1 if all(np.sign(x) == np.sign(x.iloc[0])) and len(x) == 3 else 0
    )
    imbalance_factor = data['cumulative_imbalance'] * (1 + sign_persistence * 0.2) * data['volatility_regime']
    
    # Final factor combination
    factor = (
        momentum_factor * 0.25 +
        acceleration_factor * 0.20 +
        break_factor * 0.15 +
        volatility_factor * 0.15 +
        imbalance_factor * 0.15 +
        data['price_volume_convergence'] * 0.10
    )
    
    return factor
