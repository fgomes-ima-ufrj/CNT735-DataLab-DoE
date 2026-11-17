# CNT735 · Laboratório DoE — Assuntos 11 a 16

Aplicação Streamlit para os Assuntos 11–16, alinhada ao ciclo **dados → modelos → diagnóstico → interpretação → decisão**.

## Iniciar no Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Publicar na mesma conta GitHub

Envie **o conteúdo desta pasta**, não a pasta externa, ao repositório `fgomes-ima-ufrj/CNT735-Aula14-Fatorial-2x2`. Configure o Streamlit Community Cloud para `main` e `app.py`. O endereço público deve ser validado pelo usuário; não há deploy automático neste arquivo.

Se já houver uma aplicação publicada, o push na branch conectada normalmente provoca nova compilação/deploy.

## Origem e limites dos dados

| Assunto | Dados predefinidos | Qualificação |
|---|---|---|
| 11 | Planilha da Aula 14, quatro massas 128, 50, 100, 16 | Sem réplica; saturado |
| 12 | Planilha da Aula 15, 15 registros | Réplicas 2/3 geradas deterministamente (+5%/−3%) e não independentes; quadrático integral não identificável (C²=T²) |
| 13 | Planilha da Aula 18/Assunto 15, aba DADOS_27 | Planilha de Aula 16/Assunto 13 deixou as respostas em branco; preenchimento oriundo do banco 3³ presente na Aula 18. Origem histórica declarada pela planilha: PENt2025/Assunto13 |
| 14 | Planilha da Aula 17, 18 eficiências | Valores **simulados** para ensino; CdS/cádmio requer segurança laboratorial |
| 15 | Subconjunto IDs 22,24,4,6,16,18,10,12,8,26,2,20,14 da base do Assunto 13 | Os 14 pontos externos ao subconjunto permitem comparação, não validação prospectiva independente |
| 16 | Planilha da Aula 19, 24 observações | **Simulados**; Bloco=ABC; lote/ABC perfeitamente confundidos; apenas um lote por fabricante |

**Aviso:** todas as edições ficam na sessão do navegador e não modificam os arquivos JSON de origem. Dados editados podem ser exportados em CSV. Alterações no Assunto 13 e 15 são independentes por sessão; se quiser a mesma edição, importe a mesma base em ambos após aprimoramento futuro.

## Convenções matemáticas

- `RMSE treino = sqrt(SSE/N)`; `RSE = sqrt(SSE/(N-p))` quando `N>p`.
- IC95% de parâmetros: fórmula t baseada em MQO homocedástico e observações independentes. **Não interpretar como inferência empírica nos dados artificialmente replicados ou simulados.**
- Modelo saturado não oferece graus de liberdade residuais.
- As malhas são interpolações/avaliações de modelo, não medições ou comprovação de ótimo.
- Para modelos de 3 fatores quadráticos, todas as interações de dois fatores são produtos, nunca variáveis aleatórias independentes.

## Testes

```bash
python -m pytest -q
```

## Estrutura

`app.py` (interface), `cnt735/modelos.py` (motor independente), `data/assunto*.json` (dados de referência), `tests/` (verificação).
