"""Smoke tests por stub: exercitam as rotas e chamadas Plotly, NÃO o navegador Streamlit.

A validação real de layout e widget estado exige `streamlit run app.py`.
"""
from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest


class FakeStreamlit(types.ModuleType):
    def __init__(self, route):
        super().__init__('streamlit')
        self.route = route
        self.session_state = {}
        self.sidebar = self
        self.column_config = self
        self.errors = []
        self.plot_count = 0
        self.tables_count = 0

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, trace): return False

    def __getattr__(self, attr):
        def noop(*args, **kwargs):
            return self
        return noop

    def error(self, *args, **kwargs): self.errors.append(args)
    def NumberColumn(self, *args, **kwargs): return None
    def tabs(self, titles): return [self for _ in titles]
    def columns(self, num): return [self for _ in range(num)]
    def expander(self, *args, **kwargs): return self
    def radio(self, title, options, **kwargs):
        if title == 'Assunto': return self.route
        if title == 'Profundidade pedagógica': return 'Aprofundar'
        return options[0]
    def selectbox(self, title, options, **kwargs): return options[0]
    def select_slider(self, *args, **kwargs): return kwargs.get('value', 0)
    def number_input(self, *args, **kwargs): return kwargs.get('value', 735)
    def checkbox(self, *args, **kwargs): return False
    def button(self, *args, **kwargs): return False
    def file_uploader(self, *args, **kwargs): return None
    def data_editor(self, df, *args, **kwargs): return df.copy()
    def text_area(self, *args, **kwargs): return ''
    def plotly_chart(self, figure, **kwargs):
        assert figure is not None
        # A serialização detecta erros de construção Plotly, mas NÃO testa o navegador.
        figure.to_json()
        self.plot_count += 1
    def dataframe(self, *args, **kwargs): self.tables_count += 1
    def stop(self): raise RuntimeError('Streamlit parou antes do previsto')
    def rerun(self): raise RuntimeError('Streamlit solicitou rerun inesperado')


@pytest.mark.parametrize('route', ['11','12','13','14','15','16'])
def test_rota_completa_por_stub(monkeypatch, route):
    st = FakeStreamlit(route)
    monkeypatch.setitem(sys.modules, 'streamlit', st)
    base = Path(__file__).resolve().parents[1]
    runpy.run_path(str(base/'app.py'), run_name=f'__cnt735_ui_stub_{route}__')
    assert not any(str(msg[0]).startswith('Não é possível analisar') for msg in st.errors), st.errors
    assert st.plot_count > 0
    assert st.tables_count > 0
