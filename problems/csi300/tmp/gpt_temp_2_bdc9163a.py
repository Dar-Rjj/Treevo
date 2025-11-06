import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Clustering with Price Path Efficiency and Liquidity Asymmetry alpha factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Timeframe Volatility Clustering Patterns
    # Calculate rolling volatilities
    returns = data['close'].pct_change()
    vol_5d = returns.rolling(window=5).std()
    vol_15d = returns.rolling(window=15).std()
    vol_30d = returns.rolling(window=30).std()
    
    # Volatility autocorrelation measures
    vol_autocorr_5d = vol_5d.rolling(window=10).apply(lambda x: x.autocorr(lag=1), raw=False)
    vol_autocorr_15d = vol_15d.rolling(window=20).apply(lambda x: x.autocorr(lag=1), raw=False)
    vol_autocorr_30d = vol_30d.rolling(window=30).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Volatility regime classification
    vol_regime = pd.Series(np.where(vol_5d > vol_5d.rolling(window=20).median(), 1, 0), index=data.index)
    
    # Regime transition dynamics
    regime_changes = vol_regime.diff().abs()
    low_to_high_prob = regime_changes.rolling(window=20).mean()
    
    # Volatility clustering score
    vol_clustering_score = (
        vol_autocorr_5d.fillna(0) * 0.4 + 
        vol_autocorr_15d.fillna(0) * 0.35 + 
        vol_autocorr_30d.fillna(0) * 0.25
    ) * (1 + low_to_high_prob.fillna(0))
    
    # 2. Price Path Efficiency with Microstructure Noise
    # True range efficiency
    true_range = data['high'] - data['low']
    price_change = data['close'] - data['close'].shift(1)
    range_efficiency = np.abs(price_change) / (true_range + 1e-8)
    
    # Opening jump absorption
    opening_jump = np.abs(data['open'] - data['close'].shift(1))
    jump_absorption = opening_jump / (true_range + 1e-8)
    
    # Intraday path tortuosity approximation
    intraday_movement = (data['high'] - data['low']) + np.abs(data['close'] - data['open'])
    path_tortuosity = intraday_movement / (true_range + 1e-8)
    
    # Microstructure noise estimation via price reversal
    price_reversal = (returns * returns.shift(1) < 0).astype(float)
    reversal_freq = price_reversal.rolling(window=10).mean()
    
    # Volume impact on microstructure noise
    volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    microstructure_noise = reversal_freq * (1 / (volume_ratio + 1e-8))
    
    # Price efficiency score
    efficiency_score = (
        (1 - range_efficiency.fillna(0)) * 0.4 +
        (1 - jump_absorption.fillna(0)) * 0.3 +
        (2 - path_tortuosity.fillna(0)) * 0.3
    ) * (1 - microstructure_noise.fillna(0))
    
    # 3. Liquidity Asymmetry with Order Flow Imbalance
    # Order flow imbalance approximation using amount and volume
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    trade_size_skew = (avg_trade_size - avg_trade_size.rolling(window=20).mean()) / avg_trade_size.rolling(window=20).std()
    
    # Volume-based imbalance
    volume_trend = data['volume'].pct_change(periods=3)
    price_volume_corr = data['close'].rolling(window=10).corr(data['volume'])
    
    # Liquidity asymmetry signals
    liquidity_asymmetry = (
        trade_size_skew.fillna(0) * 0.5 +
        volume_trend.fillna(0) * 0.3 +
        price_volume_corr.fillna(0) * 0.2
    )
    
    # Order flow persistence
    flow_persistence = liquidity_asymmetry.rolling(window=5).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Final liquidity score
    liquidity_score = liquidity_asymmetry * (1 + flow_persistence.fillna(0))
    
    # 4. Volatility-Price Efficiency Correlation with Liquidity Feedback
    # Cross-correlation measures
    vol_eff_corr = vol_5d.rolling(window=15).corr(efficiency_score)
    liq_vol_corr = liquidity_score.rolling(window=15).corr(vol_5d)
    flow_noise_corr = liquidity_score.rolling(window=15).corr(microstructure_noise)
    
    # Interaction strength
    interaction_strength = (
        np.abs(vol_eff_corr.fillna(0)) * 0.4 +
        np.abs(liq_vol_corr.fillna(0)) * 0.35 +
        np.abs(flow_noise_corr.fillna(0)) * 0.25
    )
    
    # Feedback direction
    feedback_direction = np.sign(vol_eff_corr.fillna(0) * liq_vol_corr.fillna(0))
    
    # 5. Synthesize Multi-Factor Volatility-Liquidity Alpha
    # Combine components with interaction weighting
    volatility_component = vol_clustering_score * efficiency_score
    liquidity_component = liquidity_score * (1 + flow_persistence.fillna(0))
    
    # Final alpha synthesis
    alpha = (
        volatility_component.fillna(0) * 0.4 +
        liquidity_component.fillna(0) * 0.35 +
        (volatility_component * liquidity_component).fillna(0) * 0.25
    ) * (1 + interaction_strength.fillna(0) * feedback_direction.fillna(0))
    
    # Normalize and return
    alpha_normalized = (alpha - alpha.rolling(window=60).mean()) / (alpha.rolling(window=60).std() + 1e-8)
    
    return alpha_normalized
