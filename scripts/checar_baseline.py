"""Autoverificação CLI reprodutível sem Streamlit. Execução: python -m scripts.checar_baseline."""
from __future__ import annotations

import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from cnt735.dados import origem, box_behnken
from cnt735.modelos import ajustar, prever, anova_ortogonal


def finite_ou_none(valor):
    return float(valor) if np.isfinite(valor) else None


def executar():
    resultados = {}
    for a, t in [('11','11'),('12','12_linear'),('12','12_curva'),
                 ('13','13'),('14','14'),('16','16')]:
        df, meta = origem(a)
        fit = ajustar(df,t,meta)
        resultados[t] = {'n': fit['n'], 'p': fit['p'], 'gl': fit['gl'],
                         'r2': finite_ou_none(fit['r2']), 'r2aj': finite_ou_none(fit['r2aj']),
                         'rmse': finite_ou_none(fit['rmse']),
                         'inferencial': fit['inferencial'], 'tipo_origem': meta.origem,
                         'hash_fonte': meta.hash_fonte,
                         'coeficientes': dict(zip(fit['X'].columns, [float(x) for x in fit['beta']]))}
    d, meta = origem('13')
    fit = ajustar(box_behnken(d), '13', meta)
    nao_usados = d.loc[~d.ID_Banco15.isin(box_behnken(d).ID_Banco15)]
    resultados['15_box_behnken'] = {
        'n': fit['n'], 'p': fit['p'], 'gl': fit['gl'], 'r2': fit['r2'],
        'rmse_treino': float(fit['rmse']),
        'n_nao_utilizados': len(nao_usados),
        'rmse_14_nao_utilizados': float(np.sqrt(np.mean((nao_usados.Y-prever(fit,nao_usados))**2)))}
    resultados['ambiente'] = {'python': platform.python_version(),
                             'numpy': np.__version__, 'pandas': pd.__version__,
                             'scipy': scipy.__version__}
    return resultados


if __name__ == '__main__':
    print(json.dumps(executar(), indent=2, ensure_ascii=False, allow_nan=False))
