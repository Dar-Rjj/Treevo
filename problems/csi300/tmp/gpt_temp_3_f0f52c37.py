import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical signals:
    1. Momentum Decay-Adjusted Price Reversal
    2. Volatility-Regime Adjusted Range Breakout
    3. Liquidity-Weighted Price Impact Factor
    4. Smart Money Flow Divergence
    5. Order Imbalance Acceleration
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Decay-Adjusted Price Reversal
    # Calculate 5-day momentum
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    
    # Apply exponential decay with λ = 0.9
    decay_weights = [0.9**k for k in range(5)]
    decayed_momentum = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 5:
            recent_momentum = []
            for j in range(5):
                if i - j >= 0:
                    mom = data['close'].iloc[i-j] / data['close'].iloc[i-j-5] - 1
                    recent_momentum.append(mom)
            if len(recent_momentum) == 5:
                decayed_momentum.iloc[i] = np.average(recent_momentum, weights=decay_weights)
    
    # Volume confirmation
    avg_volume_10d = data['volume'].shift(1).rolling(window=10, min_periods=5).mean()
    volume_ratio = data['volume'] / avg_volume_10d
    momentum_factor = decayed_momentum * volume_ratio
    
    # 2. Volatility-Regime Adjusted Range Breakout
    # Calculate daily range
    daily_range = data['high'] - data['low']
    
    # Calculate returns and volatility
    returns = data['close'].pct_change()
    vol_20d = returns.rolling(window=20, min_periods=10).std()
    median_vol = vol_20d.rolling(window=60, min_periods=30).median()
    
    # Volatility regime (1 for high, -1 for low)
    vol_regime = np.where(vol_20d > median_vol, 1, -1)
    
    # Range breakout signal
    overnight_gap = abs(data['close'].shift(1) - data['open'])
    breakout_signal = overnight_gap / daily_range.shift(1)
    volatility_factor = breakout_signal * vol_regime
    
    # 3. Liquidity-Weighted Price Impact Factor
    # Calculate dollar volume and price impact
    dollar_volume = data['close'] * data['volume']
    price_change = data['close'] - data['close'].shift(1)
    price_impact = price_change / dollar_volume.replace(0, np.nan)
    
    # Liquidity assessment
    avg_dollar_volume_10d = dollar_volume.shift(1).rolling(window=10, min_periods=5).mean()
    liquidity_score = dollar_volume / avg_dollar_volume_10d
    
    # Combine signals
    liquidity_factor = price_impact * liquidity_score * np.sign(price_change)
    
    # 4. Smart Money Flow Divergence
    # Traditional money flow
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    
    # Smart money detection (large trades)
    volume_90th = data['volume'].rolling(window=20, min_periods=10).quantile(0.9)
    is_large_trade = data['volume'] > volume_90th
    is_price_moving = abs(data['close'].pct_change()) > returns.rolling(window=20, min_periods=10).std()
    
    smart_money_periods = is_large_trade & is_price_moving
    retail_money_periods = ~smart_money_periods
    
    # Calculate flows
    smart_money_flow = money_flow.where(smart_money_periods, 0)
    retail_money_flow = money_flow.where(retail_money_periods, 0)
    
    # Rolling averages for comparison
    smart_flow_avg = smart_money_flow.rolling(window=5, min_periods=3).mean()
    retail_flow_avg = retail_money_flow.rolling(window=5, min_periods=3).mean()
    
    # Divergence signal
    flow_ratio = smart_flow_avg / retail_flow_avg.replace(0, np.nan)
    price_momentum_3d = data['close'] / data['close'].shift(3) - 1
    smart_money_factor = flow_ratio * price_momentum_3d
    
    # 5. Order Imbalance Acceleration
    # Calculate order imbalance
    mid_price = (data['high'] + data['low']) / 2
    buy_pressure = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, 1)
    sell_pressure = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, 1)
    imbalance_ratio = buy_pressure - sell_pressure
    
    # Acceleration (second derivative)
    imbalance_change = imbalance_ratio.diff()
    acceleration = imbalance_change.diff()
    
    # Volume validation
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / data['volume'].rolling(window=20, min_periods=10).std()
    volume_confirmation = np.where(abs(volume_zscore) > 1, 1, 0.5)
    
    imbalance_factor = acceleration * volume_confirmation
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'momentum': momentum_factor,
        'volatility': volatility_factor,
        'liquidity': liquidity_factor,
        'smart_money': smart_money_factor,
        'imbalance': imbalance_factor
    })
    
    # Normalize each factor
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=60, min_periods=30).mean()) / x.rolling(window=60, min_periods=30).std())
    
    # Final combined factor (equal weight)
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
