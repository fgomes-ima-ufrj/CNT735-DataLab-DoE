# CNT735 DataLab DoE v2.2.0 — Validação e migração

**Baseline:** v2.1 do ZIP fornecido em 17/11/2026.
**Diretriz:** não sobrescrever dados-base e não reescrever o núcleo Python sem justificativa.

## Rastreamento das melhorias prioritárias da auditoria

| ID | Estado v2.2.0 | Evidência do projeto | Pendência de validação |
|---|---|---|---|
| M01 — IDs 13/15 | Implementado | `cnt735/dados.py`: `ID_Banco15`, `ID_Aula16`, `Chave_Tratamento`, `box_behnken`; testes de reconciliação | Conferir em Calc com a Aula 16 original |
| M02 — Pseudorreplicação | Implementado com bloqueio | `METADADOS["12"]`, `ajuste_inferencia`, `erro_puro`, `resumo_replica` | Substituir pseudo-réplicas por novas medidas reais se desejar inferência física |
| M03 — Tipos de dados | Implementado | `Proveniencia`, `Origem_Linha`, avisos na UI e manifesto | Confirmar cada fonte experimental antes de publicar artigo |
| M04 — NaN/Inf | Implementado | `validar` e testes parametrizados | Testar diretamente no editor Streamlit real |
| M05 — Estimabilidade | Implementado | Posto, condição, matriz e aviso de curvatura C²=T², C²=1, Bloco=ABC | Ampliar detecção automática de aliases simbólicos para desenhos futuros |
| M06 — Falta de ajuste | Implementado condicional | `erro_puro` com `gl_erro_puro`, `F_lof`, `p_lof` | Requer réplicas experimentais independentes verdadeiras |
| M07 — Validação externa | Implementado para BB histórico | RMSE nos 14 não usados e comparação 27/BB | Fazer confirmação prospectiva no laboratório |
| M08 — XLSX histórico | Não modificados | Foram preservadas as planilhas originais do usuário | Abrir e recalcular no LibreOffice Calc |
| M10 — Suíte de testes | Implementado | testes de regressão, propriedades, importações, exportações e UI stub | Testar navegador real, dependência calamine no deploy |
| M13 — Interoperabilidade | Implementado | ODS editável; ZIP com CSV, guia, manifesto e tabelas | Validar abertura/cálculo no Calc local e fórmulas do usuário |

## Critérios de aceitação para uso em aula

1. Executar `python -m pytest -q` e observar **zero falhas**.
2. Abrir `streamlit run app.py` no navegador e acessar os **seis** módulos.
3. Alterar uma resposta em 13 e confirmar que 15 usa **a mesma** base na mesma sessão.
4. Reconhecer os avisos **SIMULADO/DERIVADO** (12, 14 e 16) e **sem IC experimental**.
5. Conferir coeficientes 11: `73.5, -15.5, -40.5, -1.5`.
6. Conferir BB holdout 14 pontos: RMSE aprox. `1.000063`.
7. Exportar ODS de cada assunto; abrir no LibreOffice, comparar X, β, resíduos e os números de referência.
8. Importar CSV com A,S,P,Y trocados de ordem no Assunto 13: a associação deve ser **pela combinação**, não por ID.
9. Confirmar que importação de Y não finito, ID ou condição incoerente é recusada.
10. Registrar qual Python e versões de bibliotecas foram usadas no ambiente de implantação.

## O que testes automatizados NÃO asseguram

- que os dados distribuídos sejam experimentos físicos autênticos;
- que os erros sejam independentes, normalmente distribuídos ou homocedásticos;
- que respostas simuladas tenham validade de engenharia;
- que o modelo quadrático contenha o verdadeiro mecanismo físico;
- que o ótimo numérico seja o ótimo físico global;
- que o aplicativo se apresente corretamente em todas as telas;
- que abrir no Calc equivalha a recalcular fórmulas; os ODS exportados trazem **valores editáveis** e guia de cálculos, não um motor Calc embutido;
- que a versão DoE esteja integrada ao pacote maior CNT735 DataLab v4.

## Dependências e escopo de migração

- **Preservados:** `matriz`, `ajustar`, `prever`, `anova_ortogonal`, `extremos`,
  dados JSON originais, Streamlit e Plotly.
- **Adicionados:** `dados.py`, `interoperabilidade.py`, `pedagogia.py`, `desenhos.py`,
  testes e infraestrutura CI.
- **Intencional:** EP/IC condicionais passaram a ficar **N/D** na tabela principal
  dos bancos não verificados; a demonstração t antiga está em `IC_Nominal_Didatico`.
- **Próximas evoluções (não implementadas):** testes Streamlit AppTest ou Playwright
  em execução real, ODS com fórmulas locais automáticas, importação de novos desenhos
  arbitrários e modelos hierárquicos para múltiplos lotes por fabricante.
