def heuristics_v2(df):
    weighted_avg_price = (df['high'] * 0.6 + df['low'] * 0.4)
    roc_weighted_avg_price = weighted_avg_price.pct_change()
    log_amount_to_volume_ratio = np.log(df['amount'] / df['volume'])
    ema_roc_w_avg = roc_weighted_avg_price.ewm(span=5, adjust=False).mean()
    ema_log_amt_to_vol = log_amount_to_volume_ratio.ewm(span=5, adjust=False).mean()
    weight_roc_w_avg = ema_roc_w_avg / (ema_roc_w_avg + ema_log_amt_to_vol)
    weight_log_amt_to_vol = 1 - weight_roc_w_avg
    heuristics_matrix = (weight_roc_w_avg * roc_weighted_avg_price + weight_log_amt_to_vol * log_amount_to_volume_ratio).rename('heuristics_factor')
    return heuristics_matrix
