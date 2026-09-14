from dataclasses import dataclass, asdict
from typing import Iterable, Optional
import math
import numpy as np
import pandas as pd

SUPPORTED_INDEXES = {'NIFTY', 'BANKNIFTY', 'SENSEX'}

@dataclass(frozen=True)
class RiskConfig:
    capital: float = 100000.0
    risk_per_trade_pct: float = 0.25
    max_daily_loss_pct: float = 1.0
    max_positions: int = 2
    max_notional_pct: float = 20.0

@dataclass(frozen=True)
class Opportunity:
    index: str
    expiry: str
    strike: float
    option_type: str
    score: float
    probability: float
    expectancy_r: float
    status: str
    liquidity_score: float
    flow_score: float
    volatility_score: float
    reasons: tuple

@dataclass(frozen=True)
class Trade:
    timestamp: object
    index: str
    expiry: str
    strike: float
    option_type: str
    entry: float
    exit: float
    qty: int
    pnl: float
    reason: str

class DataContract:
    REQUIRED = {'timestamp','index','expiry','strike','option_type','spot','ltp','bid','ask','volume','oi'}
    @classmethod
    def validate(cls, df):
        missing = cls.REQUIRED - set(df.columns)
        if missing: raise ValueError('Missing required columns: ' + str(sorted(missing)))
        bad = set(df['index'].dropna().unique()) - SUPPORTED_INDEXES
        if bad: raise ValueError('Unsupported indexes: ' + str(sorted(bad)))
        if not set(df['option_type'].dropna().unique()) <= {'CE','PE'}: raise ValueError('option_type must be CE or PE')

def normalize_chain(df):
    DataContract.validate(df)
    x=df.copy(); x['timestamp']=pd.to_datetime(x.timestamp,errors='coerce')
    for c in ['spot','strike','ltp','bid','ask','volume','oi']: x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'change_oi' not in x: x['change_oi']=0.0
    if 'iv' not in x: x['iv']=np.nan
    x['mid']=np.where((x.bid>0)&(x.ask>0),(x.bid+x.ask)/2,x.ltp)
    x['spread_pct']=np.where(x.mid>0,(x.ask-x.bid).clip(lower=0)/x.mid*100,np.inf)
    x['moneyness']=(x.strike-x.spot)/x.spot.replace(0,np.nan)
    x['intrinsic']=np.where(x.option_type.eq('CE'),(x.spot-x.strike).clip(lower=0),(x.strike-x.spot).clip(lower=0))
    x['time_value']=(x.ltp-x.intrinsic).clip(lower=0)
    return x.sort_values(['index','expiry','option_type','strike','timestamp']).reset_index(drop=True)

def add_chain_features(df):
    x=normalize_chain(df); g=x.groupby(['index','expiry'],group_keys=False)
    for col in ['oi','volume','change_oi','iv']:
        mean=g[col].transform('mean'); std=g[col].transform('std').replace(0,np.nan)
        x[col+'_z']=((x[col]-mean)/std).replace([np.inf,-np.inf],np.nan).fillna(0)
    totals=x.pivot_table(index=['index','expiry'],columns='option_type',values=['oi','volume','change_oi'],aggfunc='sum',fill_value=0)
    totals.columns=['_'.join(c).lower() for c in totals.columns]; totals=totals.reset_index()
    for c in ['oi_ce','oi_pe','volume_ce','volume_pe','change_oi_ce','change_oi_pe']:
        if c not in totals: totals[c]=0.0
    totals['pcr_oi']=totals.oi_pe/totals.oi_ce.replace(0,np.nan)
    totals['pcr_volume']=totals.volume_pe/totals.volume_ce.replace(0,np.nan)
    totals[['pcr_oi','pcr_volume']]=totals[['pcr_oi','pcr_volume']].replace([np.inf,-np.inf],np.nan).fillna(0)
    x=x.merge(totals[['index','expiry','pcr_oi','pcr_volume','change_oi_ce','change_oi_pe']],on=['index','expiry'],how='left')
    x['atm_distance']=(x.strike-x.spot).abs(); x['atm_distance_pct']=x.atm_distance/x.spot.replace(0,np.nan)*100
    return x

def classify_regime(underlying, lookback=20):
    r=underlying.close.pct_change().fillna(0); mom=r.rolling(10).sum(); vol=r.rolling(lookback).std(); med=vol.expanding(min_periods=lookback).median()
    out=pd.Series('RANGE',index=underlying.index)
    out[(mom>0)&(vol<=med)]='BULL'; out[(mom<0)&(vol<=med)]='BEAR'; out[(mom>0)&(vol>med)]='BULL_HIGH_VOL'; out[(mom<0)&(vol>med)]='BEAR_HIGH_VOL'
    return out

class HistoricalSimilarity:
    def __init__(self,k=100): self.k=k; self.mean=None; self.std=None; self.X=None; self.y=None
    def fit(self,X,y):
        a=np.asarray(X,dtype=float); self.mean=np.nanmean(a,axis=0); self.std=np.nanstd(a,axis=0); self.std[self.std<1e-12]=1; self.X=np.nan_to_num((a-self.mean)/self.std); self.y=np.asarray(list(y),dtype=float); return self
    def predict(self,state):
        if self.X is None: raise RuntimeError('fit first')
        z=np.nan_to_num((np.asarray(state,dtype=float)-self.mean)/self.std); d=np.sqrt(((self.X-z)**2).sum(axis=1)); idx=np.argsort(d)[:min(self.k,len(d))]; w=1/(d[idx]+1e-6); vals=self.y[idx]
        return {'p_up':float(np.average(vals>0,weights=w)),'expected_move':float(np.average(vals,weights=w)),'neighbors':int(len(idx))}

class OpportunityScorer:
    def __init__(self,min_score=75.0,min_expectancy_r=.15,max_spread_pct=3.0): self.min_score=min_score; self.min_expectancy_r=min_expectancy_r; self.max_spread_pct=max_spread_pct
    def score(self,row,probability,rr=1.5):
        liq=float(np.clip(100-row.spread_pct*20,0,100)) if np.isfinite(row.spread_pct) else 0
        flow=float(np.clip(50+8*row.oi_z+6*row.volume_z+5*(row.change_oi/max(abs(row.oi),1))*100,0,100))
        iv=float(row.iv_z) if np.isfinite(row.iv_z) else 0; vol=float(np.clip(100-30*abs(iv),0,100)); hist=float(np.clip(probability*100,0,100)); rr_score=float(np.clip(rr/3*100,0,100))
        score=.30*hist+.25*flow+.20*liq+.15*vol+.10*rr_score; exp_r=probability*rr-(1-probability)
        status='TRADE' if score>=self.min_score and exp_r>=self.min_expectancy_r and liq>=60 and row.spread_pct<=self.max_spread_pct else ('WATCH' if score>=60 else 'NO_TRADE')
        return Opportunity(str(row['index']),str(row['expiry']),float(row['strike']),str(row['option_type']),round(score,2),round(probability,4),round(exp_r,4),status,round(liq,2),round(flow,2),round(vol,2),(f'hist={hist:.1f}',f'flow={flow:.1f}',f'liq={liq:.1f}',f'vol_value={vol:.1f}',f'rr={rr:.2f}'))

def scan_chain(chain,probabilities:Optional[dict]=None,rr=1.5,topn=20):
    f=add_chain_features(chain); probabilities=probabilities or {}; scorer=OpportunityScorer(); out=[]
    for _,r in f.iterrows(): out.append(scorer.score(r,float(probabilities.get((r['index'],r['expiry'],float(r['strike']),r['option_type']),.5)),rr))
    return sorted(out,key=lambda z:z.score,reverse=True)[:topn]

def risk_size(entry,stop,lot_size,cfg):
    risk_per_lot=abs(entry-stop)*lot_size
    if risk_per_lot<=0: return 0
    allowed=cfg.capital*cfg.risk_per_trade_pct/100; cap=cfg.capital*cfg.max_notional_pct/100
    return max(0,min(int(allowed//risk_per_lot),int(cap//(entry*lot_size)) if entry>0 else 0))

def performance_metrics(pnls:Iterable[float]):
    a=np.asarray(list(pnls),dtype=float)
    if not len(a): return {'trades':0,'net':0.0,'win_rate':0.0,'profit_factor':0.0,'expectancy':0.0,'max_drawdown':0.0,'sharpe_like':0.0}
    wins=a[a>0].sum(); losses=-a[a<0].sum(); eq=a.cumsum(); peak=np.maximum.accumulate(np.r_[0,eq])[1:]; dd=peak-eq
    return {'trades':int(len(a)),'net':float(a.sum()),'win_rate':float((a>0).mean()),'profit_factor':float(wins/losses) if losses else float('inf'),'expectancy':float(a.mean()),'max_drawdown':float(dd.max()),'sharpe_like':float(a.mean()/a.std(ddof=1)*math.sqrt(len(a))) if len(a)>1 and a.std(ddof=1)>0 else 0.0}

def passes_validation_gate(m,min_trades=100,min_pf=1.20): return m['trades']>=min_trades and m['profit_factor']>min_pf and m['expectancy']>0

def walk_forward_splits(df,train_days=504,validation_days=126,step_days=126):
    day=pd.to_datetime(df.timestamp).dt.normalize(); d=day.drop_duplicates().sort_values().tolist(); i=0
    while i+train_days+validation_days<=len(d):
        train_end=d[i+train_days-1]; val_end=d[i+train_days+validation_days-1]; yield df[day<=train_end],df[(day>train_end)&(day<=val_end)]; i+=step_days

def monte_carlo(pnls,iterations=5000,seed=42):
    a=np.asarray(list(pnls),dtype=float)
    if not len(a): return {'iterations':0}
    rng=np.random.default_rng(seed); totals=np.empty(iterations); dds=np.empty(iterations)
    for i in range(iterations):
        s=rng.choice(a,size=len(a),replace=True); eq=s.cumsum(); peak=np.maximum.accumulate(np.r_[0,eq])[1:]; totals[i]=eq[-1]; dds[i]=(peak-eq).max()
    return {'iterations':iterations,'p5_total':float(np.percentile(totals,5)),'p50_total':float(np.percentile(totals,50)),'p95_total':float(np.percentile(totals,95)),'p95_max_drawdown':float(np.percentile(dds,95)),'prob_positive':float((totals>0).mean())}

class PaperBroker:
    def __init__(self,capital=100000): self.cash=float(capital); self.positions=[]; self.orders=[]
    def submit(self,opp,entry,stop,lot_size,cfg):
        if opp.status!='TRADE': return {'accepted':False,'reason':'signal_not_trade'}
        if len(self.positions)>=cfg.max_positions: return {'accepted':False,'reason':'max_positions'}
        lots=risk_size(entry,stop,lot_size,cfg)
        if lots<=0: return {'accepted':False,'reason':'risk_size_zero'}
        order={'status':'PAPER_ACCEPTED','index':opp.index,'expiry':opp.expiry,'strike':opp.strike,'option_type':opp.option_type,'entry':entry,'lots':lots,'qty':lots*lot_size}; self.orders.append(order); self.positions.append(order); return order
