"""Testes independentes de estatística e desenhos, sem a interface Streamlit.

Todos os dados são os exemplos HISTÓRICOS ou hipóteses artificiais de testes.
Um teste aprovado não transforma respostas simuladas em medições reais.
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from cnt735.dados import BB_IDS_BANCO15, METADADOS, box_behnken, chave_condicao, origem, validar
from cnt735.modelos import (ajustar, anova_ortogonal, erro_puro, extremos,
                            matriz, otimizar_quadratico, prever, sugestao_ensaio)


@pytest.mark.parametrize('k,n', [("11", 4), ("12", 15), ("13", 27), ("14", 18), ("16", 24)])
def test_quantidades_e_chaves_historicas(k, n):
    df, meta = origem(k)
    assert len(df) == n
    assert meta.hash_fonte and len(meta.hash_fonte) == 64
    assert df.Chave_Tratamento.equals(chave_condicao(df, k))
    assert df['Y'].notna().all()
    validar(df, k)


def test_11_coeficientes_efeitos_e_saturacao():
    df, meta = origem('11')
    m = ajustar(df, '11', meta)
    assert np.allclose(m['beta'], [73.5, -15.5, -40.5, -1.5], atol=1e-10)
    assert np.allclose(m['X'].T @ m['X'], 4 * np.eye(4))
    assert np.max(np.abs(m['pred'].Residuo)) < 1e-11
    assert m['gl'] == 0 and m['r2'] == pytest.approx(1)
    assert not m['inferencial'] and m['coef'].EP.isna().all()
    assert m['coef_nominal'].EP_formal_hipotetico.isna().all()


def test_12_quadraticos_confundidos_e_pseudorreplicas():
    df, meta = origem('12')
    m1 = ajustar(df, '12_linear', meta)
    m2 = ajustar(df, '12_curva', meta)
    assert np.array_equal((df.C ** 2).to_numpy(), (df["T"] ** 2).to_numpy())
    assert np.array_equal(m2['X'].Q, (df.C ** 2 + df["T"] ** 2) / 2)
    assert m2['posto'] == 5 and m2['gl'] == 10
    assert m2['rmse'] == pytest.approx(m1['rmse'], abs=1e-10)
    assert m2['r2aj'] < m1['r2aj']
    assert m2['coef'].EP.isna().all()
    assert m2['gl_erro_puro'] == 0 and m2['gl_erro_puro_algebrico'] == 10
    assert m2['proveniencia'].origem == 'derivado'
    assert (df.Tipo_Replica == 'derivada').sum() == 10


def test_13_ids_nao_equivalentes_mas_mesma_geometria():
    d, _ = origem('13')
    assert d.ID_Banco15.nunique() == d.ID_Aula16.nunique() == 27
    assert (d.ID_Banco15 != d.ID_Aula16).sum() == 18
    assert d.Chave_Tratamento.nunique() == 27
    assert set(d.Chave_Tratamento) == {
        '|'.join(f'{x:+.0f}' for x in pos) for pos in product((-1, 0, 1), repeat=3)
    }
    assert len(box_behnken(d)) == 13


def test_13_referencia_numérica_e_gl():
    d, meta = origem('13')
    m = ajustar(d, '13', meta)
    assert m['r2'] == pytest.approx(0.970555, abs=1e-6)
    assert m['r2aj'] == pytest.approx(0.954966, abs=1e-6)
    assert m['rmse'] == pytest.approx(0.535544, abs=1e-6)
    assert m['gl'] == 17
    assert not m['inferencial']
    assert m['coef'].IC95_inf.isna().all()
    assert np.isfinite(m['coef_nominal'].IC95_inf_hipotetico).all()


def test_14_sem_quadratico_c():
    d, meta = origem('14')
    m = ajustar(d, '14', meta)
    assert np.all(d.C * d.C == 1)
    assert 'C2' not in m['X'].columns
    assert m['r2'] == pytest.approx(0.996725, abs=1e-6)
    assert m['gl'] == 9
    assert not m['inferencial'] and m['coef'].EP.isna().all()
    a = pd.DataFrame([{'D': 0.7, 'E': -0.2, 'C': -1}, {'D': 0.7, 'E': -0.2, 'C': +1}])
    b = dict(zip(m['X'].columns, m['beta']))
    esperado = 2 * (b['C'] + 0.7 * b['DC'] - 0.2 * b['CE'])
    assert prever(m, a)[1] - prever(m, a)[0] == pytest.approx(esperado, abs=1e-10)


def test_15_bb_13_treino_14_validacao():
    d, meta = origem('13')
    bb = box_behnken(d)
    assert set(bb.ID_Banco15) == set(BB_IDS_BANCO15)
    assert (bb[['A', 'S', 'P']].eq(0).all(axis=1)).sum() == 1
    assert (bb[['A', 'S', 'P']].eq(0).sum(axis=1) == 1).sum() == 12
    m = ajustar(bb, '13', meta)
    assert m['gl'] == 3 and m['r2'] == pytest.approx(0.987918, abs=1e-6)
    excluded = d.loc[~d.ID_Banco15.isin(BB_IDS_BANCO15)]
    assert len(excluded) == 14
    rmse = np.sqrt(np.mean((excluded.Y - prever(m, excluded)) ** 2))
    assert rmse == pytest.approx(1.000063, abs=1e-6)
    assert rmse > m['rmse']


def test_16_ortogonal_aliasing_e_anova():
    d, meta = origem('16')
    m = ajustar(d, '16', meta)
    assert len(d) == 24 and d.Chave_Tratamento.nunique() == 8
    assert np.all(d.Bloco == d.A * d.B * d.C)
    assert d.groupby('Chave_Tratamento').size().eq(3).all()
    extra = m['X'].assign(ABC=d.A*d.B*d.C)
    assert np.linalg.matrix_rank(extra) == 8 < extra.shape[1]
    tab = anova_ortogonal(d, m)
    assert np.isclose(tab.iloc[:-2].SQ.sum() + m['sse'], m['sst'], atol=1e-9)
    assert m['gl'] == 16 and m['r2'] == pytest.approx(0.922285, abs=1e-6)
    assert tab.p_inferencial.isna().all()
    assert tab.set_index('Termo').loc['Bloco', 'Efeito'] == pytest.approx(4.359833, abs=1e-5)


@pytest.mark.parametrize('assunto,tipo', [
    ('11', '11'), ('12', '12_linear'), ('12', '12_curva'),
    ('13', '13'), ('14', '14'), ('16', '16')])
def test_coeficientes_contra_qr_e_pseudoinversa(assunto, tipo):
    d, meta = origem(assunto)
    m = ajustar(d, tipo, meta)
    x, y = m['X'].to_numpy(), d.Y.to_numpy()
    assert np.allclose(m['beta'], linalg.lstsq(x, y, lapack_driver='gelsy')[0], atol=1e-9)
    assert np.allclose(m['beta'], np.linalg.pinv(x) @ y, atol=1e-9)
    assert np.allclose(m['X'].to_numpy().T @ m['pred'].Residuo, 0, atol=1e-8)
    assert np.allclose(prever(m, d), m['pred'].Predito, atol=1e-10)


@pytest.mark.parametrize('assunto,tipo', [
    ('11', '11'), ('12', '12_linear'), ('12', '12_curva'),
    ('13', '13'), ('14', '14'), ('16', '16')])
def test_propriedade_permutacao_e_escala(assunto, tipo):
    d, meta = origem(assunto)
    m = ajustar(d, tipo, meta)
    perm = d.sample(frac=1, random_state=735).reset_index(drop=True)
    mp = ajustar(perm, tipo, meta)
    assert np.allclose(m['beta'], mp['beta'], atol=1e-9)
    esc = d.copy()
    esc['Y'] = 2*d.Y + 3
    ms = ajustar(esc, tipo, meta)
    assert np.allclose(ms['beta'], m['beta']*2 + np.r_[3, np.zeros(m['p']-1)], atol=1e-9)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_respostas_nao_finitas_bloqueadas(value):
    d, _ = origem('11')
    d.loc[0, 'Y'] = value
    with pytest.raises(ValueError, match='NaN|infinito'):
        ajustar(d, '11')


def test_fatores_fora_niveis_bloqueados():
    d, _ = origem('13')
    d.loc[0, 'S'] = .22
    with pytest.raises(ValueError, match='Níveis'):
        ajustar(d, '13')


def test_coluna_ausente_bloqueada():
    d, _ = origem('14')
    with pytest.raises(ValueError, match='Colunas obrigatórias'):
        ajustar(d.drop(columns='Y'), '14')


def test_nao_identificavel_e_rejeitado():
    d, _ = origem('11')
    d['T'] = -1
    d = d.drop(columns=['Chave_Tratamento'])  # força reavaliação do novo desenho
    with pytest.raises(ValueError, match='posto'):
        ajustar(d, '11')


def test_rejeita_id_duplicado():
    d, _ = origem('16')
    d.loc[1, 'ID'] = d.loc[0, 'ID']
    with pytest.raises(ValueError, match='duplicados'):
        ajustar(d, '16')


def test_rejeita_bloco_adulterado():
    d, _ = origem('16')
    d.loc[0, 'Bloco'] *= -1
    with pytest.raises(ValueError, match='Bloco'):
        ajustar(d, '16')


def test_erro_puro_so_com_replicas_independentes_declaradas():
    d, meta = origem('12')
    m = ajustar(d, '12_linear')
    assert m['gl_erro_puro'] == 0 and m['gl_erro_puro_algebrico'] == 10
    declaration = replace(meta, origem='medido_declarado', certificado=True, independente=True)
    m2 = ajustar(d, '12_linear', declaration)
    assert m2['gl_erro_puro'] == 10
    assert m2['inferencial'] and m2['coef'].IC95_inf.notna().all()
    assert m2['gl_lof'] == m2['gl'] - 10
    # O conjunto do teste ainda é artificial. Este teste valida APENAS o contrato de declaração.


def test_superficie_interacao_consistente_e_otimizacao_confinada():
    d, meta = origem('13')
    m = ajustar(d, '13', meta)
    tab = extremos(m, ['A','S'], fixos={'P':0}, resolucao=9)
    assert len(tab) == 81
    assert np.isclose(tab.Predito, prever(m, tab)).all()
    extremos_encontrados = otimizar_quadratico(m, ['A','S','P'])
    assert list(extremos_encontrados.Extremo) == ['máximo','mínimo']
    assert np.all(extremos_encontrados[['A','S','P']].abs().to_numpy() <= 1+1e-8)
    assert extremos_encontrados.iloc[0].Predito >= extremos_encontrados.iloc[1].Predito


def test_proposta_proximo_ensaio_dois_criterios():
    d, meta = origem('13')
    proposal = sugestao_ensaio(ajustar(box_behnken(d), '13', meta), ajustar(d, '13', meta), resolucao=5)
    assert set(proposal) == {'divergencia', 'informacao'}
    for value in proposal.values():
        assert all(abs(value[f]) <= 1 for f in ['A','S','P'])
        assert value['Ganho_D_condicional'] >= 0


def test_gradiente_analitico_concorda_com_diferencas_centrais():
    from cnt735.modelos import sensibilidade_13
    d, meta = origem('13')
    m = ajustar(d,'13',meta)
    grad = sensibilidade_13(m, resolucao=7)
    a = grad.iloc[80]
    base = {'A': a.A, 'S': a.S, 'P': a.P}
    for factor, deriv in [('A','dY_dA'),('S','dY_dS'),('P','dY_dP')]:
        up = base.copy(); dn = base.copy()
        up[factor] += 1e-5; dn[factor] -= 1e-5
        fd = (prever(m,pd.DataFrame([up]))[0] - prever(m,pd.DataFrame([dn]))[0]) / 2e-5
        assert a[deriv] == pytest.approx(fd, abs=1e-8)
    assert (grad.Norma_gradiente >= 0).all()


def test_chave_fisica_adulterada_rejeitada():
    d, _ = origem('13')
    d.loc[0, 'A'] = 0
    with pytest.raises(ValueError, match='Chave_Tratamento'):
        ajustar(d, '13')


def test_id_aula16_adulterado_rejeitado():
    d, _ = origem('13')
    d.loc[0, 'ID_Aula16'] = 22
    with pytest.raises(ValueError, match='ID_Aula16'):
        ajustar(d, '13')


def test_ordem_bloco_semente_e_restricao():
    from cnt735.desenhos import ordem_aleatoria_em_blocos
    d, _ = origem('16')
    a = ordem_aleatoria_em_blocos(d, seed=735)
    b = ordem_aleatoria_em_blocos(d, seed=735)
    c = ordem_aleatoria_em_blocos(d, seed=736)
    assert a.ID.to_list() == b.ID.to_list()
    assert a.ID.to_list() != c.ID.to_list()
    assert set(a.ID) == set(d.ID)
    assert list(a.groupby('Lote', sort=False).size()) == [12, 12]
    assert a.Ordem_global_planejada.to_list() == list(range(1,25))
    assert (a.Bloco == a.A*a.B*a.C).all()
    assert d.ID.to_list() == list(range(1,25)), 'ordem original não foi alterada'


def test_ordem_bloco_semente_invalida():
    from cnt735.desenhos import ordem_aleatoria_em_blocos
    d, _ = origem('16')
    with pytest.raises(ValueError, match='Seed'):
        ordem_aleatoria_em_blocos(d, -1)
