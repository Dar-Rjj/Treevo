import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Divergence with Volume-Efficiency Filtering
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Divergence Analysis
    # Short-term Momentum Components
    data['ret_1d'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['ret_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Medium-term Momentum Components
    data['ret_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['ret_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Momentum Divergence Calculation
    data['div_short'] = data['ret_5d'] - data['ret_1d']
    data['div_medium'] = data['ret_10d'] - data['ret_5d']
    data['div_accel'] = data['ret_20d'] - data['ret_10d']
    
    # Volume Pattern Confirmation & Weighting
    # Volume Momentum Analysis
    data['vol_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['vol_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['vol_ma_20d'] = data['volume'].rolling(window=20).mean()
    
    data['vol_ratio'] = data['volume'] / data['vol_ma_5d']
    data['vol_surge'] = (data['volume'] > 1.5 * data['vol_ma_10d']).astype(float)
    
    vol_accel_denom = data['vol_ma_5d'] / data['vol_ma_20d']
    data['vol_accel'] = data['vol_ratio'] / vol_accel_denom.replace(0, np.nan)
    
    # Volume-Price Relationship
    data['vol_trend'] = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    
    data['vol_consistency'] = data['volume'].rolling(window=5).std() / data['vol_ma_5d']
    data['vol_volatility_ratio'] = data['volume'].rolling(window=10).std() / data['volume'].rolling(window=20).std()
    
    # Market Regime Adaptation
    # Volatility Regime Detection
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_5d'] = data['tr'].rolling(window=5).mean()
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    vol_median = data['intraday_vol'].rolling(window=20).median()
    data['high_vol_regime'] = (data['intraday_vol'] > vol_median).astype(float)
    
    # Oscillation Pattern Analysis
    def count_direction_changes(series):
        if len(series) < 2:
            return 0
        changes = 0
        for i in range(1, len(series)):
            if (series.iloc[i] > series.iloc[i-1] and series.iloc[i-1] < series.iloc[i-2] if i >= 2 else False) or \
               (series.iloc[i] < series.iloc[i-1] and series.iloc[i-1] > series.iloc[i-2] if i >= 2 else False):
                changes += 1
        return changes
    
    data['price_oscillation'] = data['close'].rolling(window=5).apply(count_direction_changes, raw=False)
    data['oscillation_rate'] = data['price_oscillation'] / (data['intraday_vol'] + 1e-6)
    data['trend_stability'] = 1 / (abs(data['div_short']) + abs(data['div_medium']) + 1e-6)
    
    # Efficiency & Liquidity Filtering
    # Trading Efficiency Assessment
    data['liquidity_proxy'] = (data['high'] - data['low']) / data['close']
    data['volume_efficiency'] = data['amount'] / (data['volume'] * data['close']).replace(0, np.nan)
    data['vol_amount_ratio'] = data['volume'] / data['amount'].replace(0, np.nan)
    
    # Price Impact Evaluation
    data['market_depth'] = 1 / (data['liquidity_proxy'] + 1e-6)
    data['price_impact'] = (data['high'] - data['low']) / (data['volume'] + 1e-6)
    data['slippage_est'] = (data['high'] - data['low']) / data['close']
    
    # Composite Factor Construction
    # Combine divergence components with regime weighting
    base_divergence = (data['div_short'] + data['div_medium'] + data['div_accel']) / 3
    
    # High volatility: Scale divergence signals by volatility
    # Low volatility: Enhance divergence persistence
    regime_weighted_div = np.where(
        data['high_vol_regime'] == 1,
        base_divergence * (1 + data['intraday_vol']),
        base_divergence * (1 + data['trend_stability'])
    )
    
    # Apply volume confirmation weighting
    vol_confirmation = (data['vol_ratio'] * data['vol_accel'] * (1 + data['vol_trend']))
    divergence_with_vol = regime_weighted_div * vol_confirmation
    
    # Filter through efficiency constraints
    efficiency_score = (data['volume_efficiency'] * data['market_depth']) / (data['price_impact'] + 1e-6)
    liquidity_threshold = (data['liquidity_proxy'] < data['liquidity_proxy'].rolling(window=20).quantile(0.8)).astype(float)
    
    # Final adaptive alpha factor
    final_factor = (divergence_with_vol * efficiency_score * liquidity_threshold * 
                   (1 + data['intraday_pressure']) * data['oscillation_rate'])
    
    return final_factor
