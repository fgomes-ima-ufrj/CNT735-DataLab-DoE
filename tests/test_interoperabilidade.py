"""Testes de interoperabilidade CSV/ODS com LibreOffice Calc (sem recalcular no aplicativo Calc)."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from cnt735.dados import conferir_importacao, origem
from cnt735.modelos import ajustar, anova_ortogonal
from cnt735.interoperabilidade import (MAX_INPUT, exportar_csv_zip, exportar_ods,
                                         hash_dataframe, ler_tabela, manifesto,
                                         tabelas_modelo)


def test_importacao_csv_13_ordem_alterada_liga_y_por_condicao_e_nao_por_id():
    reference, _ = origem('13')
    imported = reference[['ID_Aula16', 'A', 'S', 'P', 'Y']].sample(frac=1, random_state=32)
    imported = imported.rename(columns={'ID_Aula16': 'ID'})
    imported['Y'] += 1
    recon = conferir_importacao(reference, imported, '13')
    assert np.allclose(recon.Y.to_numpy(), reference.Y.to_numpy() + 1)
    assert recon.ID_Banco15.equals(reference.ID_Banco15)


def test_13_id_sem_fatores_nao_aceito():
    reference, _ = origem('13')
    with pytest.raises(ValueError, match='A,S,P'):
        conferir_importacao(reference, reference[['ID', 'Y']], '13')


def test_13_condicao_duplicada_importacao_bloqueada():
    reference, _ = origem('13')
    imported = reference[['A', 'S', 'P', 'Y']].copy()
    imported.loc[0, ['A', 'S', 'P']] = imported.loc[1, ['A', 'S', 'P']].values
    with pytest.raises(ValueError, match='duplicadas'):
        conferir_importacao(reference, imported, '13')


def test_16_sem_id_ou_replica_bloqueia_importacao():
    reference, _ = origem('16')
    with pytest.raises(ValueError, match='ID|Replica'):
        conferir_importacao(reference, reference[['A','B','C','Y']], '16')


def test_importacao_12_por_condicao_e_replica():
    reference, _ = origem('12')
    imported = reference[['Condicao', 'Replica','C','T','Y']].sample(frac=1, random_state=23)
    imported['Y'] += 1.5
    result = conferir_importacao(reference, imported, '12')
    assert np.allclose(result.Y, reference.Y+1.5)


def test_importacao_16_por_id_e_fatores():
    reference, _ = origem('16')
    imported = reference[['ID','A','B','C','Y']].sample(frac=1, random_state=99)
    imported['Y'] *= 2
    result = conferir_importacao(reference, imported, '16')
    assert np.allclose(result.Y, reference.Y*2)


def test_csv_calc_separador_ponto_virgula_virgula_decimal():
    raw = 'A;S;P;Y\n-1;-1;-1;26,06\n0;0;0;33,50\n'.encode('utf-8')
    df = ler_tabela('saida.csv', raw)
    assert df.iloc[0].Y == pytest.approx(26.06)
    assert df.iloc[1].Y == pytest.approx(33.50)


def test_csv_virgula_e_ponto_decimal():
    df = ler_tabela('saida.csv', b'A,S,P,Y\n-1,-1,-1,26.06\n0,0,0,33.50\n')
    assert df.iloc[1].Y == pytest.approx(33.50)


@pytest.mark.parametrize('name', ['arquivo.docx', 'file.zip', 'ensaio.py'])
def test_extensao_nao_aceita(name):
    with pytest.raises(ValueError, match='Formato'):
        ler_tabela(name, b'qualquer')


def test_limite_tamanho_e_arquivo_vazio():
    with pytest.raises(ValueError, match='vazio|10 MiB'):
        ler_tabela('a.csv', b'')
    with pytest.raises(ValueError, match='vazio|10 MiB'):
        ler_tabela('a.csv', b' ' * (MAX_INPUT + 1))


def test_manifesto_muda_apos_edicao_y_e_preserva_fonte_original():
    data, meta = origem('11')
    m = ajustar(data,'11',meta)
    a = manifesto('11',meta,m)
    alterado = data.copy()
    alterado.loc[1,'Y'] += 5
    b = manifesto('11',meta,ajustar(alterado,'11',meta))
    assert a['proveniencia']['hash_fonte'] == b['proveniencia']['hash_fonte']
    assert a['hash_sha256_dados_ajustados'] != b['hash_sha256_dados_ajustados']
    assert a['hash_sha256_dados_ajustados'] == hash_dataframe(data)
    assert len(a['hash_sha256_dados_ajustados']) == 64


def test_tabelas_matrizes_exportadas_completo():
    df, meta = origem('16')
    m = ajustar(df,'16',meta)
    tables = tabelas_modelo(df,m,anova=anova_ortogonal(df,m))
    assert {'Dados', 'Matriz_X', 'Transposta_Xt', 'XtX', 'Inversa_XtX', 'Xty',
            'Coeficientes', 'IC_Nominal_Didatico', 'Diagnosticos', 'Resumo',
            'ANOVA', 'Guia_Calc'}.issubset(tables)
    assert len(tables['Diagnosticos']) == 24
    assert tables['Coeficientes']['IC95_inf'].isna().all()
    assert tables['ANOVA'].p_inferencial.isna().all()


def test_csv_zip_manifesto_e_tabela_editavel():
    df, meta = origem('13')
    m = ajustar(df,'13',meta)
    info = manifesto('13',meta,m)
    raw = exportar_csv_zip(tabelas_modelo(df,m), info)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert 'Matriz_X.csv' in z.namelist()
        assert 'IC_Nominal_Didatico.csv' in z.namelist()
        assert json.loads(z.read('MANIFESTO.json'))['inferencial'] is False
        tab = pd.read_csv(io.BytesIO(z.read('Matriz_X.csv')), encoding='utf-8-sig')
        assert np.allclose(tab['A'], df['A'])


def test_ods_pode_ser_lido_com_odfpy_sem_recalculo_em_calc():
    pytest.importorskip('odf')
    df, meta = origem('11')
    m = ajustar(df,'11',meta)
    info = manifesto('11', meta, m)
    raw = exportar_ods(tabelas_modelo(df,m), info)
    assert raw[:2] == b'PK'
    read = pd.read_excel(io.BytesIO(raw), engine='odf', sheet_name='Matriz_X')
    assert np.allclose(read.to_numpy(), m['X'].to_numpy())
    guia = pd.read_excel(io.BytesIO(raw), engine='odf', sheet_name='Guia_Calc')
    assert len(guia) >= 9


def test_csv_injecao_de_formula_neutralizada():
    df, meta = origem('11')
    m = ajustar(df,'11',meta)
    payload = tabelas_modelo(df,m,extras={'Teste_Texto': pd.DataFrame({'rotulo':['=HYPERLINK("x")', '+1+2', '@soma', 'seguro']})})
    with zipfile.ZipFile(io.BytesIO(exportar_csv_zip(payload, manifesto('11',meta,m)))) as z:
        txt = z.read('Teste_Texto.csv').decode('utf-8-sig')
        assert "'=HYPERLINK" in txt and "'+1+2" in txt and "'@soma" in txt
