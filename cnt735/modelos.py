"""Motor científico CNT735 v2.2: preserva matriz/ajustar/prever e valida pressupostos.

Métricas: RMSE de treino = sqrt(SSE/N); RSE = sqrt(SSE/(N-p)).
IC paramétrico só se `proveniencia` declarar medições independentes, certificadas.
A ausência de IC não impede a demonstração matemática de mínimos quadrados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from .dados import Proveniencia, METADADOS, validar, chave_condicao

TERMOS = {
    "11": ["1", "P", "T", "PT"],
    "12_linear": ["1", "C", "T", "CT"],
    "12_curva": ["1", "C", "T", "CT", "Q"],
    "13": ["1", "A", "S", "P", "A2", "S2", "P2", "AS", "AP", "SP"],
    "14": ["1", "D", "C", "E", "D2", "E2", "DC", "DE", "CE"],
    "16": ["1", "A", "B", "C", "AB", "AC", "BC", "Bloco"],
}
FACTORS_BY_TYPE = {"11": "11", "12_linear": "12", "12_curva": "12", "13": "13", "14": "14", "16": "16"}


def matriz(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    if tipo not in TERMOS: raise ValueError(f"Modelo desconhecido: {tipo}")
    d = df.copy()
    dummies = pd.DataFrame(index=d.index)
    dummies["1"] = 1.0
    for c in ("P", "T", "C", "A", "B", "S", "D", "E", "Bloco"):
        if c in d: dummies[c] = pd.to_numeric(d[c], errors="raise")
    for a, b in (("P", "T"), ("C", "T"), ("A", "S"), ("A", "P"), ("S", "P"),
                 ("D", "C"), ("D", "E"), ("C", "E"), ("A", "B"), ("A", "C"), ("B", "C")):
        if a in d and b in d: dummies[a + b] = dummies[a] * dummies[b]
    for a in ("A", "S", "P", "D", "E"):
        if a in d: dummies[a + "2"] = dummies[a] ** 2
    if "C" in d and "T" in d: dummies["Q"] = (dummies["C"] ** 2 + dummies["T"] ** 2) / 2
    m = dummies[TERMOS[tipo]].astype(float)
    if not np.isfinite(m.to_numpy()).all(): raise ValueError("Matriz X contém NaN/Inf.")
    return m


def ajuste_inferencia(proveniencia: Proveniencia, gl: int) -> tuple[bool, str]:
    if gl <= 0: return False, "Modelo saturado ou sem graus residuais: IC e testes indisponíveis."
    if proveniencia.origem == "simulado": return False, "Dados simulados: inferência numérica não é evidência física."
    if proveniencia.origem == "derivado": return False, "Pseudo-réplicas determinísticas: incerteza experimental não identificável."
    if not (proveniencia.certificado and proveniencia.independente):
        return False, "Independência e origem experimental não certificadas: EP/IC e p inferenciais ocultos."
    return True, "IC t condicionado a erros independentes, normais e homocedásticos; verificar diagnósticos."


def erro_puro(df: pd.DataFrame, assunto: str, *, validas: bool = False) -> dict:
    """Decomposição SSE quando existem replicatas independentes na mesma condição.

    Apenas 'validas=True' permite usar a decomposição para teste de falta de ajuste.
    """
    chave = chave_condicao(df, assunto)
    vals = pd.to_numeric(df.Y, errors="raise").to_numpy(dtype=float)
    grupo = pd.DataFrame({"cond": chave.to_numpy(), "y": vals}).groupby("cond").y
    gl = int(len(df) - grupo.ngroups)
    sq = float(sum(((g.to_numpy() - g.mean()) ** 2).sum() for _, g in grupo))
    return {"gl_erro_puro_algebrico": gl, "sq_erro_puro_algebrico": sq,
            "gl_erro_puro": gl if validas else 0,
            "sq_erro_puro": sq if validas else np.nan,
            "erro_puro_validado": bool(validas and gl > 0)}


def ajustar(df: pd.DataFrame, tipo: str, proveniencia: Proveniencia | None = None) -> dict:
    assunto = FACTORS_BY_TYPE.get(tipo)
    if assunto is None: raise ValueError(f"Modelo não implementado: {tipo}")
    validar(df, assunto)
    X = matriz(df, tipo)
    xx = X.to_numpy(dtype=float)
    y = pd.to_numeric(df.Y, errors="raise").to_numpy(dtype=float)
    n, p = xx.shape
    singular = np.linalg.svd(xx, compute_uv=False)
    rank = int(np.linalg.matrix_rank(xx))
    if rank < p: raise ValueError(f"Matriz não identificável: posto {rank} < {p}; verifique aliasing/níveis.")
    if n < p: raise ValueError("N menor que o número de parâmetros.")
    cond = float(singular[0] / singular[-1])
    if cond > 1e10: raise ValueError(f"Matriz quase singular: condição={cond:.3g} (limite 1e10).")
    b = np.linalg.lstsq(xx, y, rcond=None)[0]
    yh = xx @ b
    res = y - yh
    sse = float(res @ res)
    sst = float(((y - y.mean()) ** 2).sum())
    gl = n - p
    r2 = 1 - sse / sst if sst > 1e-15 else np.nan
    r2a = 1 - (1 - r2) * (n - 1) / gl if gl > 0 else np.nan
    mse = sse / gl if gl > 0 else np.nan
    # Utilizar fatoração numericamente mais estável que a inversão de XtX para estimar.
    inv = np.linalg.inv(xx.T @ xx)  # exclusivamente demonstração do enunciado
    cov_formal = mse * inv if gl > 0 else np.full((p, p), np.nan)
    se_formal = np.sqrt(np.maximum(0.0, np.diag(cov_formal))) if gl > 0 else np.full(p, np.nan)
    tcrit = stats.t.ppf(0.975, gl) if gl > 0 else np.nan
    meta = proveniencia or METADADOS[assunto]
    can_infer, motivo = ajuste_inferencia(meta, gl)
    # Contrato rigoroso: EP/IC inferenciais são N/D sem declaração de independência.
    # A tabela nominal abaixo é APENAS uma demonstração contrafactual de OLS.
    se = se_formal if can_infer else np.full(p, np.nan)
    coef = pd.DataFrame({"Termo": X.columns, "Coeficiente": b, "EP": se,
                         "IC95_inf": b - tcrit * se, "IC95_sup": b + tcrit * se})
    coef_nominal = pd.DataFrame({"Termo": X.columns, "Coeficiente": b,
                                  "EP_formal_hipotetico": se_formal,
                                  "IC95_inf_hipotetico": b - tcrit * se_formal,
                                  "IC95_sup_hipotetico": b + tcrit * se_formal})
    previsto = df.copy()
    previsto["Predito"] = yh
    previsto["Residuo"] = res
    previsto["Residuo_abs"] = np.abs(res)
    # Alavancagem e Cook: diagnósticos geométricos descritivos, não testes.
    leverage = np.einsum("ij,jk,ik->i", xx, inv, xx)
    leverage = np.clip(leverage, 0, 1)
    den = 1 - leverage
    if gl > 0 and mse > 1e-20:
        student = np.divide(res, np.sqrt(mse * den), out=np.full(n, np.nan), where=den > 1e-12)
        cook = np.divide((res**2 / (p * mse)) * leverage, den**2,
                         out=np.full(n, np.nan), where=den > 1e-12)
    else:
        student = np.full(n, np.nan)
        cook = np.full(n, np.nan)
    previsto["Leverage"] = leverage
    previsto["Residuo_studentizado_interno"] = student
    previsto["Cook"] = cook
    previsto["Predito_LOO"] = np.divide(yh - leverage * y, den,
                                       out=np.full(n, np.nan), where=den > 1e-12)
    previsto["Residuo_LOO"] = np.divide(res, den, out=np.full(n, np.nan), where=den > 1e-12)
    loo = float(np.sqrt(np.nanmean(previsto.Residuo_LOO**2))) if np.isfinite(previsto.Residuo_LOO).all() else np.nan
    ep = erro_puro(df, assunto, validas=can_infer)
    gl_lof = gl - ep["gl_erro_puro"] if ep["erro_puro_validado"] else 0
    sq_lof = max(0.0, sse - ep["sq_erro_puro"]) if ep["erro_puro_validado"] else np.nan
    if gl_lof > 0 and ep["gl_erro_puro"] > 0 and ep["sq_erro_puro"] > 0:
        f_lof = (sq_lof / gl_lof) / (ep["sq_erro_puro"] / ep["gl_erro_puro"])
        p_lof = float(stats.f.sf(f_lof, gl_lof, ep["gl_erro_puro"]))
    else: f_lof = p_lof = np.nan
    return {"tipo": tipo, "X": X, "Xt": X.T, "XtX": X.T @ X,
            "XtX_inv": pd.DataFrame(inv, index=X.columns, columns=X.columns),
            "Xty": pd.Series(xx.T @ y, index=X.columns, name="Xty"),
            "coef": coef, "coef_nominal": coef_nominal, "beta": b, "pred": previsto, "n": n, "p": p, "gl": gl,
            "posto": rank, "num_condicao": cond, "sse": sse, "sst": sst,
            "r2": r2, "r2aj": r2a, "rmse": np.sqrt(sse / n), "rmse_loo": loo,
            "rse": np.sqrt(mse) if gl > 0 else np.nan,
            "cov": cov_formal if can_infer else np.full((p, p), np.nan),
            "mse": mse, "inferencial": can_infer, "status_inferencia": motivo,
            "proveniencia": meta, "gl_lof": gl_lof, "sq_lof": sq_lof,
            "F_lof": f_lof, "p_lof": p_lof, **ep}


def prever(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    X = matriz(df, modelo["tipo"]).to_numpy()
    return X @ modelo["beta"]


def anova_ortogonal(df: pd.DataFrame, modelo: dict) -> pd.DataFrame:
    """ANOVA válida numericamente se colunas ±1 são ortogonais e balanceadas."""
    X = modelo["X"]
    xx = X.to_numpy()
    gram = xx.T @ xx
    if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-8):
        raise ValueError("O planejamento não é ortogonal: contrastes não são independentes.")
    y = np.asarray(df.Y, dtype=float)
    n, mse = len(y), modelo["mse"]
    out = []
    for nome in X.columns[1:]:
        x = X[nome].to_numpy()
        contraste = float(x @ y)
        sq = contraste**2 / float(x @ x)
        f = sq / mse if mse > 0 else np.nan
        # F pode ser mostrado como aritmética; p só se proveniência sustentar inferência.
        out.append({"Termo": nome, "Contraste": contraste, "Efeito": 2 * contraste / n,
                    "Coeficiente": contraste / float(x @ x), "SQ": sq,
                    "gl": 1, "QM": sq, "F": f,
                    "p_inferencial": stats.f.sf(f, 1, modelo["gl"]) if modelo["inferencial"] and np.isfinite(f) else np.nan})
    out.append({"Termo": "Erro", "Contraste": np.nan, "Efeito": np.nan, "Coeficiente": np.nan,
                "SQ": modelo["sse"], "gl": modelo["gl"], "QM": mse, "F": np.nan, "p_inferencial": np.nan})
    out.append({"Termo": "Total corrigido", "SQ": modelo["sst"], "gl": n - 1})
    return pd.DataFrame(out)


def extremos(modelo: dict, nomes: list[str], fixos: dict | None = None, resolucao: int = 23) -> pd.DataFrame:
    from itertools import product
    if not isinstance(resolucao, int) or not 2 <= resolucao <= 100:
        raise ValueError("Resolução da malha deve estar entre 2 e 100.")
    fixos = fixos or {}
    d = pd.DataFrame(list(product(*([np.linspace(-1, 1, resolucao)] * len(nomes)))), columns=nomes)
    for k, v in fixos.items(): d[k] = v
    d["Predito"] = prever(modelo, d)
    return d


def otimizar_quadratico(modelo: dict, nomes: list[str]) -> pd.DataFrame:
    """Multi-start L-BFGS-B determinístico; extremos candidatos, não confirmação global."""
    starts = [np.array(x) for x in __import__("itertools").product((-1., 0., 1.), repeat=len(nomes))]
    resultados = []
    for direcao, sentido in (("máximo", -1), ("mínimo", 1)):
        best = None
        for start in starts:
            def fun(x):
                d = pd.DataFrame([dict(zip(nomes, x))])
                return sentido * float(prever(modelo, d)[0])
            res = optimize.minimize(fun, start, method="L-BFGS-B", bounds=[(-1, 1)] * len(nomes),
                                    options={"maxiter": 400, "ftol": 1e-12})
            if best is None or res.fun < best.fun: best = res
        resultados.append({"Extremo": direcao, **dict(zip(nomes, best.x)),
                           "Predito": float(sentido * best.fun), "Convergiu": bool(best.success)})
    return pd.DataFrame(resultados)


def sugestao_ensaio(modelo_bb: dict, modelo27: dict, nomes=("A", "S", "P"), resolucao=11) -> dict:
    """Duas propostas: maior divergência e maior informação D incremental condicional."""
    grade = extremos(modelo_bb, list(nomes), resolucao=resolucao)
    grade["Delta_abs"] = np.abs(grade.Predito - prever(modelo27, grade))
    xx = matriz(grade, modelo_bb["tipo"]).to_numpy()
    grade["Ganho_D_condicional"] = np.einsum("ij,jk,ik->i", xx, modelo_bb["XtX_inv"].to_numpy(), xx)
    return {"divergencia": grade.loc[grade.Delta_abs.idxmax()].to_dict(),
            "informacao": grade.loc[grade.Ganho_D_condicional.idxmax()].to_dict()}


def sensibilidade_13(modelo: dict, resolucao: int = 9) -> pd.DataFrame:
    """Sensibilidade local do polinômio quadrático 3 fatores em [-1,1]^3.

    Gradientes são analíticos (unidades de Y por unidade codificada).
    Derivada pequena NÃO garante robustez contra ruído nem confirma ótimo.
    """
    if modelo["tipo"] != "13":
        raise ValueError("Sensibilidade analítica disponível para modelo quadrático de A,S,P.")
    if not isinstance(resolucao, int) or resolucao < 2 or resolucao > 40:
        raise ValueError("Resolução deve estar no intervalo [2, 40].")
    df = extremos(modelo, ["A", "S", "P"], resolucao=resolucao)
    b = dict(zip(modelo["X"].columns, modelo["beta"]))
    a, s, p = df.A.to_numpy(), df.S.to_numpy(), df.P.to_numpy()
    d_a = b["A"] + 2 * b["A2"] * a + b["AS"] * s + b["AP"] * p
    d_s = b["S"] + 2 * b["S2"] * s + b["AS"] * a + b["SP"] * p
    d_p = b["P"] + 2 * b["P2"] * p + b["AP"] * a + b["SP"] * s
    df["dY_dA"] = d_a
    df["dY_dS"] = d_s
    df["dY_dP"] = d_p
    df["Norma_gradiente"] = np.sqrt(d_a ** 2 + d_s ** 2 + d_p ** 2)
    return df
