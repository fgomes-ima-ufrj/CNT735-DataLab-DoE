# Relatório de entrega — CNT735 DataLab DoE v2.2.0

**Data:** 17/11/2026  
**Escopo:** código-fonte integral do laboratório independente de DoE, Assuntos 11–16.

## Evidências de execução neste ambiente

- `python -m pytest -q` executado em 17/11/2026: **69 testes aprovados**.
- `python -m py_compile app.py cnt735/*.py scripts/*.py` concluído **sem erros**.
- `python -m scripts.checar_baseline` executado e salvo em `BASELINE_NUMERICO.json`.
- 6 fluxos de interface percorridos por stub (incluem construção de figuras e ODS): **6 aprovados**. **Não** equivalem a renderização Streamlit real.
- Exportação ODS foi gerada e lida novamente com `odfpy` e pandas, mas **não** houve recálculo no LibreOffice Calc.
- Importação CSV com `;`/vírgula decimal e CSV padrão verificada; importação XLSX requer `python-calamine`, não disponível neste ambiente.
- `streamlit` não estava instalado neste ambiente; portanto **não houve teste de interface real em navegador nem deploy**.
- GitHub Actions foi configurado no pacote, **ainda não executado no GitHub do usuário**.

## Preservação

Todos os cinco arquivos `data/assunto*.json` foram comparados por bytes com os arquivos equivalentes do ZIP original: **iguais**.
O ZIP final exclui `.git`, `__pycache__` e diretórios temporários de testes.

## Riscos que permanecem

1. A origem experimental dos valores 13/15 continua documental, não independentemente certificada.
2. Assunto 12 usa valores artificialmente derivados e não fornece IC físico de réplicas.
3. Assuntos 14 e 16 são demonstrações simuladas.
4. Sem teste visual em Streamlit real / Calc real (pendência de homologação antes da distribuição em aula).
5. Não foi realizada integração com a versão maior CNT735 DataLab v4 (0–16).

## Diretório

A pasta interna `CNT735-DataLab-DoE` é o projeto completo. Instalação e execução em `README.md`.
