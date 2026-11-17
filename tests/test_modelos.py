import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cnt735.modelos import ajustar,matriz,anova_ortogonal
BASE=Path(__file__).resolve().parents[1]/'data'
def data(k):return pd.DataFrame(json.loads((BASE/f'assunto{k}.json').read_text()))
def test_assunto11():
 m=ajustar(data('11'),'11')
 assert np.allclose(m['beta'],[73.5,-15.5,-40.5,-1.5])
 assert m['gl']==0 and m['sse']<1e-20
 assert np.allclose(m['XtX'].values,4*np.eye(4))
def test_assunto12_nao_identificavel_6():
 d=data('12');x=matriz(d,'12_curva');assert np.linalg.matrix_rank(x)==5
 assert np.allclose(d['C']**2,d['T']**2)
 assert ajustar(d,'12_linear')['p']==4
 assert ajustar(d,'12_curva')['p']==5
def test_assunto13():
 m=ajustar(data('13'),'13');assert m['n']==27 and m['p']==10 and m['gl']==17
 assert m["coef"].IC95_inf.isna().all()  # resultados históricos não certificam independência
 assert np.isfinite(m["coef_nominal"].IC95_inf_hipotetico).all()
def test_assunto14():
 d=data('14');m=ajustar(d,'14');assert m['n']==18 and m['p']==9
 assert 'C2' not in m['X'].columns
def test_assunto15():
 d=data('13');bb=[22,24,4,6,16,18,10,12,8,26,2,20,14]
 sub=d.set_index('ID').loc[bb].reset_index();m=ajustar(sub,'13')
 assert m['n']==13 and m['p']==10 and m['gl']==3
def test_assunto16():
 d=data('16');m=ajustar(d,'16');assert m['n']==24 and m['gl']==16
 assert np.allclose(d.Bloco,d.A*d.B*d.C)
 an=anova_ortogonal(d,m);assert np.isclose(an.iloc[:-2].SQ.sum()+m['sse'],m['sst'])
