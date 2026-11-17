"""Ferramentas de planejamento que não modificam o desenho original."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dados import validar


def ordem_aleatoria_em_blocos(df: pd.DataFrame, seed: int = 735) -> pd.DataFrame:
    """Propõe ordem aleatória restrita ao lote (Assunto 16).

    O sorteio é um PLANO futuro, não informação sobre ordem real das medições.
    Não altera fabricante nem variável de confundimento e não mistura lotes.
    """
    validar(df, "16")
    if "Lote" not in df.columns:
        raise ValueError("A coluna Lote é necessária para a aleatorização restrita.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed deve ser inteiro não negativo.")
    rng = np.random.default_rng(int(seed))
    lotes = df["Lote"].unique().tolist()
    partes = []
    for lote in lotes:
        parte = df.loc[df.Lote.eq(lote)].copy(deep=True)
        parte = parte.iloc[rng.permutation(len(parte))].reset_index(drop=True)
        parte["Ordem_no_lote_planejada"] = np.arange(1, len(parte)+1)
        partes.append(parte)
    resultado = pd.concat(partes, ignore_index=True)
    resultado["Ordem_global_planejada"] = np.arange(1, len(resultado)+1)
    resultado["Seed_ordem"] = int(seed)
    return resultado
