"""Regressão de paleta: o padrão histórico é Rainbow nos três tipos de visualização."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_paleta_padrao_e_arco_iris():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    color = next(node.value for node in tree.body
                 if isinstance(node, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "COLOR" for t in node.targets))
    names = [key.value for key in color.keys]
    values = [value.value for value in color.values]
    assert names[0] == "Arco-íris"
    assert values[0] == "Rainbow"
    assert "Viridis" in values and "Cividis" in values

def test_superficie_contorno_heatmap_usam_mesma_paleta():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "superficie")
    expected = {"Surface", "Contour", "Heatmap"}
    found = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in expected:
                setting = {kw.arg: kw.value for kw in node.keywords}
                assert "colorscale" in setting
                assert isinstance(setting["colorscale"], ast.Name)
                assert setting["colorscale"].id == "escala"
                found.add(node.func.attr)
    assert found == expected
