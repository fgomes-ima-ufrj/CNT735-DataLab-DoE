"""Interoperabilidade com Calc: importação verificada por fatores e pacote ODS/CSV.

O pacote de saída contém tabelas editáveis; não pressupõe que os XLSX
históricos tenham sido recalculados sem erro no LibreOffice.
"""
from __future__ import annotations

import io
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .dados import Proveniencia, conferir_importacao

MAX_INPUT = 10 * 1024 * 1024


def ler_tabela(nome: str, raw: bytes, *, sheet: str | int = 0) -> pd.DataFrame:
    """CSV UTF-8/Latin-1; XLSX/ODS por python-calamine em instalação de usuário."""
    if not raw or len(raw) > MAX_INPUT: raise ValueError("Arquivo vazio ou maior que 10 MiB.")
    suffix = Path(nome).suffix.lower()
    source = io.BytesIO(raw)
    if suffix == ".csv":
        # Detecção simples e previsível do CSV exportado pelo Calc em PT-BR:
        # `;` costuma ser separador e `,` separador decimal.
        try:
            first = raw[:4096].decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            first = raw[:4096].decode("latin1")
            encoding = "latin1"
        first_line = next((line for line in first.splitlines() if line.strip()), "")
        sep = ";" if first_line.count(";") > first_line.count(",") else ","
        decimal = "," if sep == ";" else "."
        return pd.read_csv(source, encoding=encoding, sep=sep, decimal=decimal)
    if suffix == ".ods":
        return pd.read_excel(source, sheet_name=sheet, engine="odf")
    if suffix == ".xlsx":
        try:
            return pd.read_excel(source, sheet_name=sheet, engine="calamine")
        except ImportError as e:
            raise ValueError("Instale python-calamine conforme requirements.txt.") from e
    raise ValueError("Formato não aceito: use CSV, ODS ou XLSX.")


def importar_respostas(referencia: pd.DataFrame, nome: str, raw: bytes, assunto: str) -> pd.DataFrame:
    return conferir_importacao(referencia, ler_tabela(nome, raw), assunto)


def _json_safe(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (datetime,)): return obj.isoformat()
    if isinstance(obj, (float,)) and not np.isfinite(obj): return None
    raise TypeError(type(obj).__name__)


def tabelas_modelo(df: pd.DataFrame, model: dict, *, anova: pd.DataFrame | None = None,
                   extras: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    res = {
        "Dados": df,
        "Matriz_X": model["X"].reset_index(drop=True),
        "Transposta_Xt": model["Xt"].reset_index().rename(columns={"index": "Termo"}),
        "XtX": model["XtX"].reset_index().rename(columns={"index": "Termo"}),
        "Inversa_XtX": model["XtX_inv"].reset_index().rename(columns={"index": "Termo"}),
        "Xty": model["Xty"].reset_index().rename(columns={"index": "Termo", "Xty": "Valor"}),
        "Coeficientes": model["coef"],
        "IC_Nominal_Didatico": model["coef_nominal"],
        "Diagnosticos": model["pred"],
        "Resumo": pd.DataFrame([{
            "N": model["n"], "p": model["p"], "gl_resid": model["gl"],
            "Posto": model["posto"], "Condicao_X": model["num_condicao"],
            "SSE": model["sse"], "SST": model["sst"], "R2": model["r2"],
            "R2_aj": model["r2aj"], "RMSE_treino": model["rmse"],
            "RMSE_LOO_descritivo": model["rmse_loo"],
            "gl_erro_puro_validado": model["gl_erro_puro"],
            "SQ_erro_puro_validada": model["sq_erro_puro"],
            "Inferencia_habilitada": model["inferencial"],
            "Aviso": model["status_inferencia"],
            "IC_Nominal_NAO_evidencia_fisica": not model["inferencial"]}]),
        "Guia_Calc": pd.DataFrame({"Etapa": [
            "1. X", "2. Xt", "3. XtX", "4. XtX inversa", "5. Xty", "6. b", "7. Predição", "8. Resíduos", "9. RMSE",
            "10. IC95", "11. Posto / pseudorreplicação"],
            "Procedimento": [
                "Veja Matriz_X: intercepto, fatores e interações derivados (não independentes).",
                "=TRANSPOSE(matriz_X) com seleção de área correta; veja Transposta_Xt.",
                "=MMULT(transposta_X;matriz_X); confira a tabela XtX.",
                "=MINVERSE(XtX), desde que X tenha posto completo.",
                "=MMULT(transposta_X;y); confira Xty.",
                "=MMULT(MINVERSE(XtX);Xty) [didático]; QR/SVD para estabilidade numérica.",
                "Cada linha: SUMPRODUCT(linha_X;vetor_b).",
                "=Y - Predito. Verifique se Xᵀe≈0.",
                "=SQRT(SUMSQ(residuos)/N); RMSE de treino, não estimativa de σ.",
                "Somente com observações independentes, variância estimável e pressupostos; caso contrário N/D.",
                "Não usar pseudo-réplicas como unidades independentes; nunca interpretar modelo saturado como validado."],
        }),
    }
    if anova is not None: res["ANOVA"] = anova
    if extras:
        for nome, tab in extras.items(): res[nome[:31]] = tab
    return res


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash estável da tabela realmente ajustada (inclusive edições da sessão)."""
    buffer = df.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(buffer.encode("utf-8")).hexdigest()


def manifesto(assunto: str, meta: Proveniencia, model: dict) -> dict:
    return {
        "software": "CNT735 DataLab DoE", "version": "2.2.0", "assunto": assunto,
        "emissao_utc": datetime.now(timezone.utc).isoformat(),
        "proveniencia": meta.como_dict(), "tipo_modelo": model["tipo"],
        "hash_sha256_dados_ajustados": hash_dataframe(model["pred"].drop(
            columns=["Predito", "Residuo", "Residuo_abs", "Leverage",
                     "Residuo_studentizado_interno", "Cook", "Predito_LOO", "Residuo_LOO"], errors="ignore")),
        "termos": model["X"].columns.tolist(), "formula_rmse": "sqrt(SSE/N)",
        "inferencial": bool(model["inferencial"]), "restricao": model["status_inferencia"],
        "ic_nominal": "Apenas contrafactual matemático, não intervalo de confiança experimental certificado.",
        "observacao": "Tabelas editáveis. Dados de referência não reescritos. Verificar cálculo do Calc separadamente.",
    }


def _frame_csv_safe(frame: pd.DataFrame) -> bytes:
    tab = frame.copy()
    # Previne interpretação de textos importados como fórmula em planilha.
    for col in tab.select_dtypes(include=["object", "string"]).columns:
        tab[col] = tab[col].map(lambda v: "'" + str(v) if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t")) else v)
    return tab.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def exportar_csv_zip(tabelas: dict[str, pd.DataFrame], info: dict) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for nome, tab in tabelas.items():
            z.writestr(f"{nome}.csv", _frame_csv_safe(tab))
        z.writestr("MANIFESTO.json", json.dumps(info, ensure_ascii=False, indent=2, default=_json_safe))
        z.writestr("LEIA-ME.txt", "CNT735 DataLab DoE: importe CSV em UTF-8. Matriz X, XtX, Xty e beta são reproduzíveis no Calc. IC_Nominal_Didatico NÃO representa inferência experimental.\n")
    return out.getvalue()


def exportar_ods(tabelas: dict[str, pd.DataFrame], info: dict) -> bytes:
    """Pandas+odfpy exporta tabelas editáveis sem fórmulas XLSX incompatíveis.
    ODS não é uma planilha 'recalculada' até que o aluno insira fórmulas.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="odf") as writer:
        for nome, tab in tabelas.items():
            q = tab.replace([np.inf, -np.inf], np.nan).copy()
            q.columns = [str(c) for c in q.columns]
            q.to_excel(writer, sheet_name=nome[:31], index=False, na_rep="N/D")
        pd.DataFrame([info["proveniencia"]]).to_excel(writer, sheet_name="Proveniencia", index=False)
    return output.getvalue()
