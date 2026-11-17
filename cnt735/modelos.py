"""Núcleo científico sem Streamlit. Convenção RMSE=sqrt(SSE/n)."""
import numpy as np
import pandas as pd
from scipy import stats

TERMOS = {
 '11':['1','P','T','PT'],
 '12_linear':['1','C','T','CT'],
 '12_curva':['1','C','T','CT','Q'],
 '13':['1','A','S','P','A2','S2','P2','AS','AP','SP'],
 '14':['1','D','C','E','D2','E2','DC','DE','CE'],
 '16':['1','A','B','C','AB','AC','BC','Bloco'],
}

def matriz(df, tipo):
 d=df.copy();out=pd.DataFrame(index=d.index);out['1']=1.
 for c in ('P','T','C','A','B','S','D','E','Bloco'):
  if c in d:out[c]=pd.to_numeric(d[c])
 for a,b in [('P','T'),('C','T'),('A','S'),('A','P'),('S','P'),('D','C'),('D','E'),('C','E'),('A','B'),('A','C'),('B','C')]:
  if a in d and b in d:out[a+b]=pd.to_numeric(d[a])*pd.to_numeric(d[b])
 for a in ('A','S','P','D','E'):
  if a in d:out[a+'2']=pd.to_numeric(d[a])**2
 if 'C' in d and 'T' in d:out['Q']=(pd.to_numeric(d['C'])**2+pd.to_numeric(d['T'])**2)/2
 return out[TERMOS[tipo]].astype(float)

def ajustar(df,tipo):
 X=matriz(df,tipo);xx=X.to_numpy();y=pd.to_numeric(df.Y).to_numpy(dtype=float)
 n,p=xx.shape;rank=np.linalg.matrix_rank(xx)
 if rank<p:raise ValueError(f'Matriz não identificável: posto {rank} < {p}.')
 b=np.linalg.lstsq(xx,y,rcond=None)[0]
 yh=xx@b;res=y-yh;sse=float(res@res);sst=float(((y-y.mean())**2).sum());gl=n-p
 r2=1-sse/sst if sst>1e-15 else np.nan
 r2a=1-(1-r2)*(n-1)/gl if gl>0 else np.nan
 mse=sse/gl if gl>0 else np.nan
 cov=mse*np.linalg.inv(xx.T@xx) if gl>0 else np.full((p,p),np.nan)
 se=np.sqrt(np.diag(cov)) if gl>0 else np.full(p,np.nan)
 tcrit=stats.t.ppf(.975,gl) if gl>0 else np.nan
 coef=pd.DataFrame({'Termo':X.columns,'Coeficiente':b,'EP':se,'IC95_inf':b-tcrit*se,'IC95_sup':b+tcrit*se})
 previsto=df.copy();previsto['Predito']=yh;previsto['Residuo']=res;previsto['Residuo_abs']=np.abs(res)
 return {'tipo':tipo,'X':X,'Xt':X.T,'XtX':X.T@X,'XtX_inv':pd.DataFrame(np.linalg.inv(xx.T@xx),index=X.columns,columns=X.columns),'Xty':pd.Series(xx.T@y,index=X.columns,name='Xty'),'coef':coef,'beta':b,'pred':previsto,'n':n,'p':p,'gl':gl,'sse':sse,'sst':sst,'r2':r2,'r2aj':r2a,'rmse':np.sqrt(sse/n),'rse':np.sqrt(mse) if gl>0 else np.nan,'cov':cov,'mse':mse}

def prever(modelo,df):
 return matriz(df,modelo['tipo']).to_numpy()@modelo['beta']

def anova_ortogonal(df,modelo):
 """Para 2³ balanceado, contrastes mutuamente ortogonais."""
 X=modelo['X'];y=np.asarray(df.Y,dtype=float);n=len(y);mse=modelo['mse'];out=[]
 for nome in X.columns[1:]:
  x=X[nome].to_numpy();contrast=float(x@y);sq=contrast**2/float(x@x)
  F=sq/mse if mse>0 else np.nan
  out.append({'Termo':nome,'Contraste':contrast,'Efeito':2*contrast/n,'Coeficiente':contrast/n,'SQ':sq,'gl':1,'QM':sq,'F':F,'p':stats.f.sf(F,1,modelo['gl']) if np.isfinite(F) else np.nan})
 out.append({'Termo':'Erro','Contraste':np.nan,'Efeito':np.nan,'Coeficiente':np.nan,'SQ':modelo['sse'],'gl':modelo['gl'],'QM':mse,'F':np.nan,'p':np.nan})
 out.append({'Termo':'Total corrigido','SQ':modelo['sst'],'gl':n-1})
 return pd.DataFrame(out)

def extremos(modelo, nomes, fixos=None, resolucao=23):
 from itertools import product
 fixos=fixos or {}
 levels=np.linspace(-1,1,resolucao)
 d=pd.DataFrame(list(product(*([levels]*len(nomes)))),columns=nomes)
 for k,v in fixos.items():d[k]=v
 d['Predito']=prever(modelo,d)
 return d
