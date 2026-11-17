"""Dados de referência, chaves físicas, proveniência e validação (sem interface)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BB_IDS_BANCO15 = (22, 24, 4, 6, 16, 18, 10, 12, 8, 26, 2, 20, 14)
FATORES = {
    "11": ("P", "T"), "12": ("C", "T"), "13": ("A", "S", "P"),
    "14": ("D", "C", "E"), "16": ("A", "B", "C"),
}
NIVEIS = {
    "11": {"P": (-1, 1), "T": (-1, 1)},
    "12": {"C": (-1, 0, 1), "T": (-1, 0, 1)},
    "13": {"A": (-1, 0, 1), "S": (-1, 0, 1), "P": (-1, 0, 1)},
    "14": {"D": (-1, 0, 1), "C": (-1, 1), "E": (-1, 0, 1)},
    "16": {"A": (-1, 1), "B": (-1, 1), "C": (-1, 1)},
}

@dataclass(frozen=True)
class Proveniencia:
    assunto: str
    origem: Literal["documentado", "derivado", "simulado", "importado_nao_verificado", "medido_declarado"]
    fonte: str
    unidade: str
    independente: bool = False
    certificado: bool = False
    observacao: str = ""
    hash_fonte: str = ""
    versao: str = "2.2.0"

    def como_dict(self):
        return vars(self).copy()

METADADOS = {
    "11": Proveniencia("11", "documentado", "Aula 14 — matriz 2²", "massa: unidade não fornecida", observacao="Sem réplicas; modelo saturado."),
    "12": Proveniencia("12", "derivado", "Aula 15 — 5 respostas-base e dez pseudo-réplicas ±5%/−3%", "tamanho: unidade não informada", observacao="Não usar DP, EP e IC como incerteza experimental; 10 valores derivados."),
    "13": Proveniencia("13", "documentado", "Aula 18 — DADOS_27 (origem histórica declarada PENt2025)", "avanço de perfuração (%)", observacao="Aula 16 tem Y em branco e numeração própria, distinta de DADOS_27; origem física das respostas não certificada."),
    "14": Proveniencia("14", "simulado", "Aula 17 — respostas artificiais de CdS", "eficiência (%)", observacao="Resultados não são evidência de desempenho real de células solares."),
    "16": Proveniencia("16", "simulado", "Aula 19 — força magnética artificial", "força magnética: unidade não fornecida", observacao="Fabricante e ABC perfeitamente confundidos; único lote por fabricante."),
}

def origem(assunto: str) -> tuple[pd.DataFrame, Proveniencia]:
    key = "13" if assunto == "15" else assunto
    path = ROOT / "data" / f"assunto{key}.json"
    raw = path.read_bytes()
    rows = json.loads(raw)
    frame = enriquecer(pd.DataFrame(rows), key)
    return frame, replace(METADADOS[key], hash_fonte=hashlib.sha256(raw).hexdigest())

def chave_condicao(df: pd.DataFrame, assunto: str) -> pd.Series:
    key = "13" if assunto == "15" else assunto
    factors = FATORES[key]
    for factor in factors:
        if factor not in df.columns:
            raise ValueError(f"Fator obrigatório ausente: {factor}")
    vals = df[list(factors)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Fatores não podem conter NaN/Inf.")
    return pd.Series(["|".join(f"{v:+.0f}" for v in row) for row in vals], index=df.index, name="Chave_Tratamento")

def id_original_aula16(df: pd.DataFrame) -> pd.Series:
    """Ordem da Aula 16: A externo, S intermediário, P interno, todos -1/0/+1.
    Não reatribui o ID de respostas vindas do DADOS_27 da Aula 18.
    """
    a, s, p = [pd.to_numeric(df[c]) for c in ("A", "S", "P")]
    return (1 + 9 * (a + 1) + 3 * (s + 1) + (p + 1)).astype(int).rename("ID_Aula16")

def enriquecer(df: pd.DataFrame, assunto: str) -> pd.DataFrame:
    out = df.copy(deep=True)
    if assunto == "13":
        if "ID" not in out: raise ValueError("Banco 3³ sem ID da fonte DADOS_27.")
        out["ID_Banco15"] = out["ID"].astype(int)
        out["ID_Aula16"] = id_original_aula16(out)
    elif assunto == "12":
        out["Origem_Linha"] = np.where(out["Replica"].eq(1), "base_documentada", "derivada_deterministica")
        out["Tipo_Replica"] = np.where(out["Replica"].eq(1), "nao_confirmada", "derivada")
    elif assunto in {"14", "16"}:
        out["Origem_Linha"] = "simulada"
        if assunto == "16":
            out["Tipo_Replica"] = "simulada"
    else:
        out["Origem_Linha"] = "documentada_sem_confirmacao"
    out["Chave_Tratamento"] = chave_condicao(out, assunto)
    return out

def validar(df: pd.DataFrame, assunto: str, *, exige_estrutura: bool = True) -> None:
    key = "13" if assunto == "15" else assunto
    if not isinstance(df, pd.DataFrame) or df.empty: raise ValueError("Tabela de dados vazia.")
    cols = set(FATORES[key]) | {"Y"}
    faltantes = cols - set(df.columns)
    if faltantes: raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    for c in cols:
        try: v = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as e: raise ValueError(f"Coluna {c}: conteúdo não numérico.") from e
        if not np.isfinite(v).all(): raise ValueError(f"Coluna {c}: resposta/fator NaN ou infinito.")
    for c, permitidos in NIVEIS[key].items():
        bad = ~np.isin(df[c].to_numpy(dtype=float), permitidos)
        if bad.any(): raise ValueError(f"Níveis de {c} fora de {permitidos}.")
    if exige_estrutura and key == "16":
        bloco = df["A"].to_numpy() * df["B"].to_numpy() * df["C"].to_numpy()
        if "Bloco" in df and not np.allclose(df.Bloco, bloco):
            raise ValueError("Bloco deve ser exatamente A×B×C; edição inconsistente.")
    if "ID" in df and df["ID"].duplicated().any():
        raise ValueError("Há identificadores de corrida ID duplicados.")
    # Detecta adulteração acidental da identidade de um tratamento após a carga.
    if "Chave_Tratamento" in df and not df["Chave_Tratamento"].astype(str).equals(chave_condicao(df, key)):
        raise ValueError("Chave_Tratamento não corresponde mais aos fatores codificados.")
    if key == "13" and "ID_Aula16" in df and not df.ID_Aula16.astype(int).equals(id_original_aula16(df)):
        raise ValueError("ID_Aula16 não corresponde aos fatores do planejamento.")

def box_behnken(df: pd.DataFrame) -> pd.DataFrame:
    """Selecionar pelo ID da fonte das respostas, verificando a geometria experimental."""
    if "ID_Banco15" not in df:
        df = enriquecer(df, "13")
    if df.ID_Banco15.duplicated().any(): raise ValueError("ID_Banco15 duplicado.")
    if not set(BB_IDS_BANCO15).issubset(set(df.ID_Banco15)):
        raise ValueError("IDs de origem ausentes para Box–Behnken.")
    sub = df.set_index("ID_Banco15").loc[list(BB_IDS_BANCO15)].reset_index()
    cod = sub[["A", "S", "P"]].to_numpy(dtype=float)
    centros = np.sum(np.all(cod == 0, axis=1))
    arestas = np.sum(np.sum(cod == 0, axis=1) == 1)
    if centros != 1 or arestas != 12 or sub.Chave_Tratamento.duplicated().any():
        raise ValueError("Subconjunto não possui 12 arestas + 1 centro Box–Behnken.")
    return sub

def conferir_importacao(referencia: pd.DataFrame, importado: pd.DataFrame, assunto: str) -> pd.DataFrame:
    """Atualiza apenas respostas; identifica 13 por fatores e outros por corrida+fatores.
    Nunca usar ID isolado no 13, pois é ambíguo entre as duas planilhas.
    """
    key = "13" if assunto == "15" else assunto
    v = importado.copy()
    if "Y" not in v: raise ValueError("O arquivo precisa conter coluna Y.")
    if key == "13":
        for c in FATORES[key]:
            if c not in v:
                raise ValueError("Para Assunto 13/15, envie fatores A,S,P: IDs diferem entre planilhas.")
        keys = list(FATORES[key])
    else:
        keys = list(FATORES[key])
        if "ID" in referencia and "ID" in v: keys = ["ID"] + keys
        elif "Condicao" in referencia and "Condicao" in v and "Replica" in v:
            keys = ["Condicao", "Replica"] + keys
        else: raise ValueError("Importação exige ID da corrida ou Condicao+Replica além dos fatores.")
    if v.duplicated(keys).any(): raise ValueError("Chaves de importação duplicadas.")
    if referencia.duplicated(keys).any(): raise ValueError("Chaves ambíguas na referência.")
    if len(v) != len(referencia): raise ValueError("Quantidade de corridas divergente do planejamento.")
    r = referencia.copy(deep=True)
    resp = r[keys].merge(v[keys + ["Y"]], how="left", on=keys, validate="one_to_one", indicator=True)
    if not resp._merge.eq("both").all(): raise ValueError("Fatores/IDs incompatíveis com o planejamento distribuído.")
    y = pd.to_numeric(resp.Y, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all(): raise ValueError("Respostas importadas contêm valores inválidos.")
    r["Y"] = y
    return r
