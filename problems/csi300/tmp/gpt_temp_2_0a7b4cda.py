import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Regime-Adaptive Entropy-Volume Momentum factor
    """
    data = df.copy()
    
    # 1. Regime Classification
    # Volatility regime
    data['volatility'] = (data['high'] - data['low']) / data['close']
    
    # Trend regime
    data['trend_short'] = np.sign(data['close'] - data['close'].shift(5))
    data['trend_medium'] = np.sign(data['close'] - data['close'].shift(10))
    data['trend'] = data['trend_short'] * data['trend_medium']
    
    # Liquidity regime
    data['volume_ma20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['liquidity'] = data['volume'] / data['volume_ma20']
    
    # 2. Entropy-Volume Dynamics
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Price entropy using True Range percentile ranks
    def calculate_entropy(series, window=10):
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
            else:
                window_data = series.iloc[i-window:i]
                # Calculate percentile ranks
                ranks = (window_data.rank(pct=True) * 100).fillna(50)
                # Create histogram bins for entropy calculation
                hist, _ = np.histogram(ranks, bins=5, range=(0, 100))
                hist = hist[hist > 0]  # Remove zero bins for entropy calculation
                if len(hist) > 1:
                    entropy_val = entropy(hist)
                else:
                    entropy_val = 0
                entropy_values.append(entropy_val)
        return pd.Series(entropy_values, index=series.index)
    
    data['price_entropy'] = calculate_entropy(data['true_range'], window=10)
    
    # Volume-Price Asymmetry (simplified buy/sell pressure)
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['price_change'] = data['typical_price'] - data['typical_price'].shift(1)
    data['buy_pressure'] = np.where(data['price_change'] > 0, data['volume'], 0)
    data['sell_pressure'] = np.where(data['price_change'] < 0, data['volume'], 0)
    
    # 5-day rolling sums for stability
    data['buy_pressure_ma'] = data['buy_pressure'].rolling(window=5, min_periods=3).sum()
    data['sell_pressure_ma'] = data['sell_pressure'].rolling(window=5, min_periods=3).sum()
    data['volume_asymmetry'] = (data['buy_pressure_ma'] - data['sell_pressure_ma']) / (data['buy_pressure_ma'] + data['sell_pressure_ma'] + 1e-8)
    
    # 3. Adaptive Momentum
    data['momentum_short'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_medium'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    
    # 4. Signal Synthesis
    # Regime classification thresholds
    high_vol_threshold = data['volatility'].rolling(window=20, min_periods=10).quantile(0.7)
    low_vol_threshold = data['volatility'].rolling(window=20, min_periods=10).quantile(0.3)
    high_liq_threshold = data['liquidity'].rolling(window=20, min_periods=10).quantile(0.7)
    low_liq_threshold = data['liquidity'].rolling(window=20, min_periods=10).quantile(0.3)
    
    # Regime identification
    data['vol_regime'] = np.where(data['volatility'] > high_vol_threshold, 2, 
                                 np.where(data['volatility'] < low_vol_threshold, 0, 1))
    data['trend_regime'] = np.where(data['trend'] > 0, 1, 
                                   np.where(data['trend'] < 0, -1, 0))
    data['liq_regime'] = np.where(data['liquidity'] > high_liq_threshold, 2, 
                                 np.where(data['liquidity'] < low_liq_threshold, 0, 1))
    
    # Regime-adaptive weights
    # High volatility favors entropy, low volatility favors momentum
    volatility_weight = data['vol_regime'] / 2.0  # 0, 0.5, or 1
    
    # Strong trends favor momentum, weak trends favor volume asymmetry
    trend_strength_weight = abs(data['trend_regime'])  # 0 or 1
    
    # High liquidity favors all signals, low liquidity reduces weights
    liquidity_weight = data['liq_regime'] / 2.0  # 0, 0.5, or 1
    
    # Component combination with regime adaptation
    entropy_component = data['price_entropy'] * data['volume_asymmetry']
    momentum_component = data['momentum_short'] * 0.6 + data['momentum_medium'] * 0.4
    
    # Regime-weighted signal
    regime_weight = (volatility_weight + trend_strength_weight + liquidity_weight) / 3.0
    
    # Final factor calculation
    base_signal = (entropy_component * volatility_weight + 
                   momentum_component * (1 - volatility_weight))
    
    factor = base_signal * regime_weight
    
    # Cross-regime validation filter
    # Only take signals when at least 2 regimes are favorable
    favorable_regimes = ((data['vol_regime'] >= 1).astype(int) + 
                        (data['trend_regime'] != 0).astype(int) + 
                        (data['liq_regime'] >= 1).astype(int))
    
    factor = np.where(favorable_regimes >= 2, factor, 0)
    
    return factor
