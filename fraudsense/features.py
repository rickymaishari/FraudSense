"""One shared, label-free feature builder. Same-hour transactions never see each other."""
import numpy as np
import pandas as pd

TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
BALANCE = ['opening_balance', 'log_balance', 'amount_balance_ratio', 'balance_missing']


def prior_count(d, keys):
    # Only repeated entities can have history; this greatly reduces PaySim memory.
    out = np.zeros(len(d), dtype=np.float32)
    repeated = d.duplicated(keys, keep=False)
    if not repeated.any():
        return out
    sub = d.loc[repeated, keys + ['step']].copy()
    sub['_row'] = np.flatnonzero(repeated)
    g = sub.groupby(keys + ['step'], sort=True, observed=True).size().rename('n').reset_index()
    g['prior'] = g.groupby(keys, observed=True)['n'].cumsum() - g.n
    merged = sub.merge(g[keys + ['step', 'prior']], on=keys + ['step'], how='left')
    out[merged._row.to_numpy()] = merged.prior.to_numpy()
    return out


def build_features(transactions, history=None):
    """Explicit history must precede all assessed hours. Batch has causal history.

    Previous amount means mean amount in the last observed hour: PaySim supplies
    no ordering within an hour. IDs are join keys only, never model predictors.
    """
    n_history = 0
    if history is not None and len(history):
        if history.step.max() >= transactions.step.min():
            raise ValueError('History must strictly precede every assessed simulation hour')
        n_history = len(history)
        d = pd.concat([history, transactions], ignore_index=True)
    else:
        d = transactions.reset_index(drop=True)
    a = d.amount.to_numpy(dtype=float)
    balance = pd.to_numeric(d.get('oldbalanceOrg', pd.Series(np.nan, index=d.index)), errors='coerce').to_numpy(float)
    x = pd.DataFrame({'amount': a, 'log_amount': np.log1p(a), 'hour': d.step.to_numpy() % 24,
                      'opening_balance': np.nan_to_num(balance), 'log_balance': np.log1p(np.nan_to_num(balance)),
                      'amount_balance_ratio': np.minimum(a / np.maximum(np.nan_to_num(balance), 1), 1e6),
                      'balance_missing': np.isnan(balance).astype(float)})
    for t in TYPES:
        x['type_' + t] = (d.type == t).to_numpy(dtype=float)
    for name in ['sender_count', 'sender_mean', 'sender_max', 'previous_hour_mean', 'sender_volume',
                 'sender_count_24h', 'sender_volume_24h']:
        x[name] = 0.
    x['sender_gap'] = -1.
    repeated = d.duplicated('nameOrig', keep=False)
    if repeated.any():
        s = d.loc[repeated, ['nameOrig', 'step', 'amount']].copy()
        s['_row'] = np.flatnonzero(repeated)
        g = s.groupby(['nameOrig', 'step'], sort=True, observed=True).amount.agg(['size', 'sum', 'max']).reset_index()
        groups = g.groupby('nameOrig', sort=False, observed=True)
        g['sender_count'] = groups['size'].cumsum() - g['size']
        g['sender_volume'] = groups['sum'].cumsum() - g['sum']
        g['sender_mean'] = g.sender_volume / g.sender_count.clip(lower=1)
        g['_max'] = groups['max'].cummax()
        g['sender_max'] = g.groupby('nameOrig', observed=True)._max.shift().fillna(0)
        g['_mean'] = g['sum'] / g['size']
        g['previous_hour_mean'] = g.groupby('nameOrig', observed=True)._mean.shift().fillna(0)
        g['sender_gap'] = g.step - groups.step.shift()
        g['sender_gap'] = g.sender_gap.fillna(-1)
        g['sender_count_24h'] = 0.
        g['sender_volume_24h'] = 0.
        # This loop runs only on repeated senders, not 6.36 million records.
        for _, part in g.groupby('nameOrig', sort=False, observed=True):
            h = part.step.to_numpy()
            left = np.searchsorted(h, h - 24, side='left')
            for src, dst in [('size', 'sender_count_24h'), ('sum', 'sender_volume_24h')]:
                cumulative = np.r_[0., np.cumsum(part[src].to_numpy())]
                g.loc[part.index, dst] = cumulative[:-1] - cumulative[left]
        cols = ['sender_count', 'sender_volume', 'sender_mean', 'sender_max', 'previous_hour_mean',
                'sender_gap', 'sender_count_24h', 'sender_volume_24h']
        m = s.merge(g[['nameOrig', 'step'] + cols], on=['nameOrig', 'step'], how='left')
        x.loc[m._row.to_numpy(), cols] = m[cols].to_numpy()
    x['recipient_count'] = prior_count(d, ['nameDest'])
    x['pair_count'] = prior_count(d, ['nameOrig', 'nameDest'])
    x['new_recipient'] = (x.pair_count == 0).astype(float)
    x['amount_to_history_mean'] = np.minimum(a / (x.sender_mean.to_numpy() + 1), 1e9)
    x['amount_deviation'] = np.log1p(a) - np.log1p(x.sender_mean)
    x['history_available'] = (x.sender_count > 0).astype(float)
    return x.astype('float32').iloc[n_history:].reset_index(drop=True)
