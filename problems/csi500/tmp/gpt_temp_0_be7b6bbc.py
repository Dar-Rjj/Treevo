import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multiple market microstructure signals
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volume-Weighted Price Momentum Factor
    # Calculate momentum components
    data['mom_short'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['mom_medium'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['mom_ratio'] = data['mom_short'] / data['mom_medium']
    
    # Volume weighting scheme
    data['vol_trend'] = data['volume'] / data['volume'].shift(1)
    vol_ma_20 = data['volume'].rolling(window=20).mean()
    data['vol_persistence'] = (data['volume'] > vol_ma_20).astype(int)
    data['vol_persistence'] = data['vol_persistence'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    data['vol_momentum'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    
    # Volatility smoothing
    returns = data['close'].pct_change()
    data['price_vol'] = returns.rolling(window=20).std()
    vol_median_60 = data['price_vol'].rolling(window=60).median()
    data['vol_regime'] = data['price_vol'] / vol_median_60
    
    momentum_factor = data['mom_ratio'] * data['vol_persistence'] / data['vol_regime']
    
    # Liquidity Regime Switching Indicator
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    vol_ma_5 = data['volume'].rolling(window=5).mean()
    data['vol_concentration'] = data['volume'] / vol_ma_5
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    data['price_impact'] = abs(data['close'] - data['close'].shift(1)) / typical_price
    
    data['liquidity_score'] = data['spread_proxy'] * data['vol_concentration']
    
    # Detect regime changes
    liquidity_threshold = data['liquidity_score'].rolling(window=20).median()
    regime = (data['liquidity_score'] > liquidity_threshold).astype(int)
    data['regime_persistence'] = regime.groupby((regime != regime.shift()).cumsum()).cumcount() + 1
    
    # Calculate transition probability (simplified)
    regime_changes = (regime != regime.shift()).astype(int)
    data['transition_prob'] = regime_changes.rolling(window=60).mean()
    
    liquidity_factor = data['liquidity_score'] * data['regime_persistence']
    
    # Intraday Range Efficiency Factor
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Range persistence
    range_expanding = (data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))
    data['range_persistence'] = range_expanding.astype(int)
    data['range_persistence'] = data['range_persistence'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # Volume-range correlation
    data['daily_range'] = data['high'] - data['low']
    data['vol_range_corr'] = data['volume'].rolling(window=10).corr(data['daily_range'])
    data['vol_efficiency'] = data['volume'] / data['daily_range'].replace(0, np.nan)
    
    range_factor = data['daily_efficiency'] * data['vol_range_corr'] * data['range_persistence']
    
    # Order Flow Imbalance Factor
    data['directional_amount'] = data['amount'] * np.sign(data['close'] - data['open'])
    
    # Amount persistence
    amount_direction = np.sign(data['directional_amount'])
    data['amount_persistence'] = amount_direction.astype(int)
    data['amount_persistence'] = data['amount_persistence'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    data['amount_acceleration'] = (data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1)
    
    # Price-amount correlation
    price_change_magnitude = abs(data['close'] - data['open'])
    data['price_amount_corr'] = data['amount'].rolling(window=10).corr(price_change_magnitude)
    data['amount_efficiency'] = price_change_magnitude / data['amount'].replace(0, np.nan)
    
    # Flow momentum
    data['flow_momentum'] = data['directional_amount'].rolling(window=5).sum()
    
    flow_factor = data['flow_momentum'] * data['amount_efficiency'] * data['amount_persistence']
    
    # Multi-Timeframe Convergence Factor
    # 5-day vs 20-day momentum alignment
    mom_5 = data['close'].pct_change(5)
    mom_20 = data['close'].pct_change(20)
    momentum_alignment = (np.sign(mom_5) == np.sign(mom_20)).astype(int)
    
    # Volume trend consistency
    vol_trend_5 = (data['volume'] > data['volume'].rolling(window=5).mean()).astype(int)
    vol_trend_20 = (data['volume'] > data['volume'].rolling(window=20).mean()).astype(int)
    volume_consistency = (vol_trend_5 == vol_trend_20).astype(int)
    
    # Range expansion confirmation
    range_expansion_5 = (data['daily_range'] > data['daily_range'].rolling(window=5).mean()).astype(int)
    range_expansion_20 = (data['daily_range'] > data['daily_range'].rolling(window=20).mean()).astype(int)
    range_confirmation = (range_expansion_5 == range_expansion_20).astype(int)
    
    # Convergence score
    data['convergence_score'] = momentum_alignment + volume_consistency + range_confirmation
    
    # Signal magnitude (average of normalized factors)
    factors = pd.DataFrame({
        'momentum': momentum_factor,
        'liquidity': liquidity_factor,
        'range': range_factor,
        'flow': flow_factor
    })
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    data['signal_magnitude'] = normalized_factors.mean(axis=1)
    
    # Persistence score
    convergence_persistence = (data['convergence_score'] >= 2).astype(int)
    data['persistence_score'] = convergence_persistence.groupby(
        (convergence_persistence != convergence_persistence.shift()).cumsum()
    ).cumcount() + 1
    
    # Final composite factor
    composite_factor = (
        data['convergence_score'] * 
        data['signal_magnitude'] * 
        data['persistence_score']
    )
    
    # Clean and return the final factor
    final_factor = composite_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
