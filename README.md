# CNT735 DataLab DoE — v2.2.0 (versão auditada)

**PENt/COPPE/UFRJ · Assuntos 11 a 16 · Python + Streamlit + LibreOffice Calc**

Aplicativo completo e editável para planejamento experimental, modelos por mínimos quadrados,
ANOVA, superfícies, validação, blocos e confundimento. Esta versão amplia o projeto **v2.1**
sem alterar os cinco bancos históricos JSON e sem migrar de Python ou Streamlit.

> **Atenção científica:** os dados do Assunto 12 incluem pseudorréplicas derivadas,
> os dos Assuntos 14 e 16 são simulados, e os Assuntos 11 e 13/15 possuem limitações
> de inferência. Resultados numericamente corretos não são validação física ou causal.

## 1. Instalação local (Ubuntu/Linux)

Recomendado Python 3.11 ou 3.12. Em uma pasta **fora do ZIP**:

```bash
unzip CNT735-DataLab-DoE-v2.2.0.zip
cd CNT735-DataLab-DoE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt  # testes opcionais
python -m pytest -q
streamlit run app.py
```

Abra o endereço informado pelo Streamlit (normalmente `http://localhost:8501`).
A interface permite editar **apenas as respostas Y**, por sessão, preservando os arquivos
originais. O Assunto **15 compartilha a base editada do Assunto 13**.

Sem Streamlit, o motor funciona por Python puro:

```python
from cnt735.dados import origem, box_behnken
from cnt735.modelos import ajustar

df, origem_dados = origem("13")
modelo27 = ajustar(df, "13", origem_dados)
modelo13 = ajustar(box_behnken(df), "13", origem_dados)
print(modelo27["coef"])
print(modelo13["r2"], modelo13["rmse"])
```

Verificação com todos os dados de referência: `python -m scripts.checar_baseline`.

## 2. GitHub / Streamlit Community Cloud

Use preferencialmente o repositório já existente **`fgomes-ima-ufrj/CNT735-DataLab-DoE`**,
verificando o nome exato na sua conta. **Não crie outro projeto por engano.**
Suba **os arquivos dentro da pasta extraída** com Git (sem a pasta `.venv`), usando
`main` e `app.py` no formulário do Streamlit Community Cloud. A URL pública
é definida pelo serviço, não está presumida nesta documentação.

```bash
git status
git add .
git commit -m "CNT735 DataLab DoE v2.2 - auditoria, testes e Calc"
git push origin main
```

**Antes de enviar:** compare as mudanças (`git diff`) e confirme a branch e o remoto.
O ZIP distribuído **não contém `.git`**; se estiver substituindo arquivos em um clone,
copie-os para dentro do clone sem apagar a pasta `.git` existente.
A pasta `.github/workflows/ci.yml` configura testes automatizados no GitHub Actions.

## 3. O que mudou na v2.2.0

| Prioridade | Implementação |
|---|---|
| P0 | Dupla identificação `ID_Banco15` ↔ `ID_Aula16` por `Chave_Tratamento` codificada, 27/27 condições reconciliáveis sem alterar respostas; seleção BB por IDs da fonte da resposta |
| P0 | Proveniência explícita: histórico/documentado, derivado e simulado; bloqueio de IC/EP inferenciais em dados não certificados; IC **nominais didáticos** em aba separada |
| P0 | Erro puro só é inferencial se existirem replicações independentes declaradas (teste de falta de ajuste quando viável); pseudo-replicações recebem apenas DP aritmético |
| P1 | Validação de fatores, Y finito, rank, número de condição e consistência de Bloco=ABC |
| P1 | Testes OLS contra solução QR/pseudoinversa, propriedades de permutação e escala, casos inválidos, aliases, BB holdout e testes por stub das seis rotas |
| P1 | Importação CSV/ODS/XLSX com chave de tratamento/identificadores, bloqueando união apenas por ID no 3³ |
| P1 | Exportação Calc: ODS editável e ZIP CSV com dados, X, Xt, XtX, inversa, Xty, β, resíduos, ANOVA/diagnósticos, guia e manifesto com SHA-256 |
| P1 | Box–Behnken: RMSE nos **14 não usados**, tabelas de erro e mapas ΔY; escala cromática compartilhada para comparar 27/13 |
| P2 | Sensibilidade do modelo quadrático por gradiente analítico; extremos multistart e critérios de ensaio por divergência e informação incremental, **sem alegar ótimo global confirmado** |
| P2 | Roteiros por assunto (prever–analisar–interpretar–decidir), três níveis pedagógicos e nova paleta `Viridis` por padrão (`Rainbow` legado opcional) |
| P2 | Proposta de ordem aleatorizada **dentro de cada lote** com seed explícita, sem mudar o confundimento do Assunto 16 |

## 4. Limitações científicas (por assunto)

| Assunto | Limite não eliminado pelo software |
|---|---|
| **11 — Fatorial 2²** | 4 observações e 4 parâmetros; saturado; nenhum grau de liberdade residual. Efeito = 2β, mas R²=1 não confirma predição. |
| **12 — Réplicas** | Dez linhas são derivadas deterministicamente das cinco respostas-base. A estrutura disponível torna C²=T²: os dois termos não são estimáveis separadamente. O termo `Q=(C²+T²)/2` é extensão restrita, não o modelo quadrático integral pedido. |
| **13 — 3³** | A planilha original da Aula 16 tem Y vazio; os 27 Y vêm do banco da Aula 18. Não utilizar o mesmo ID para unir as duas ordens (18 divergências). Sem réplicas independentes não há erro puro para separar falta de ajuste. |
| **14 — 3×2×3** | Eficiências iniciais são simuladas. C²=1 é confundido com intercepto. O modelo reduzido possui curvatura D²/E² comum a ambos os níveis C; C pode mudar a inclinação por DC/CE, não sua curvatura neste modelo. |
| **15 — Box–Behnken** | Um único ponto central e 13 observações; IC dependem do modelo/erro. Os 14 pontos não usados são validação *interna* do banco 27, não réplica prospectiva. `13/27` é fração de custo, não fração de informação. |
| **16 — Blocos** | Simulação de 24 corridas. Bloco = ABC, portanto fabricante/lote e a interação de terceira ordem não podem ser identificados separadamente. Um lote por fabricante impede generalizar diferenças ao fabricante como população. |

### Tabelas de intervalo de confiança

- Em `Coeficientes`, EP/IC ficam **N/D** quando não há origem/independência declarada.
- `IC_Nominal_Didatico` mostra os cálculos da fórmula t **condicionados a hipóteses não verificadas**;
  nunca são prova de variação experimental, especialmente no 12, 14 ou 16.
- Se importar novas medições e declarar unidades experimentais independentes,
  o software passa a permitir IC **condicionais à sua declaração**. A independência
  real, coleta e pressupostos **não são auditados automaticamente**.

## 5. Importação e exportação no Calc

Na aplicação, expanda **Importar respostas** e escolha `CSV`, `XLSX` ou `ODS`.
Para preservar comparabilidade com o exercício, só Y pode ser substituído; os fatores
codificados e o número de linhas precisam coincidir com a referência.

- **Assuntos 13/15:** obrigatoriamente colunas `A,S,P,Y`, associadas **pela condição**, nunca por `ID` isolado.
- **Assuntos 11/14/16:** colunas `ID`, fatores e `Y`.
- **Assunto 12:** `Condicao,Replica,C,T,Y`.
- Para declaração de independência, adicione coluna `Unidade_Experimental` única, sem vazios;
  trata-se de uma **declaração do usuário**, não certificação automática da coleta.
- CSV do Calc em PT-BR com `;` e vírgula decimal e CSV padrão com vírgulas são lidos.
- Importação é limitada a **10 MiB**; origens PDF/imagens não são suportadas.

**Exportar:** ODS com várias abas (tabelas de valores editáveis, sem fórmulas nativas
recalculadas) ou ZIP de CSV + `MANIFESTO.json` (SHA-256 da fonte e da tabela
ajustada). A aba `Guia_Calc` apresenta o roteiro de fórmulas; copie
as matrizes se desejar refazer a álgebra no Calc. Não assuma que os XLSX
históricos da disciplina estejam sem erros: algumas fórmulas cacheadas eram inválidas.

## 6. Validação e execução

```bash
python -m pytest -q                        # todos os testes, incluindo UI simulada
python -m pytest -q tests/test_modelos.py    # regressões históricas
python -m scripts.checar_baseline            # números de referência e holdout BB
python -m py_compile app.py cnt735/*.py
```

Os testes UI simulados via `FakeStreamlit` exercitam rotas e serialização Plotly,
**não comprovam renderização real, responsividade nem funcionamento do Streamlit
Community Cloud**. Para liberar à turma, testar em browser Firefox/Chrome no
Ubuntu e em dispositivo móvel, abrir/exportar ODS no **LibreOffice Calc** e comparar
recalculando as fórmulas. Ver `docs/VALIDACAO_E_MIGRACAO.md`.

## 7. Organização do projeto

```text
app.py                       interface Streamlit 11–16
cnt735/dados.py              metadados, origens, reconciliação e importação validada
cnt735/modelos.py            MQO, ANOVA, diagnósticos, superfícies e otimização
cnt735/desenhos.py           aleatorização reprodutível restrita aos blocos
cnt735/interoperabilidade.py  importação e exportação Calc, proveniência e hashes
cnt735/pedagogia.py          questões, previsões e interpretação
scripts/checar_baseline.py   referência numérica rápida
data/*.json                   respostas históricas preservadas
requirements*.txt           dependências de execução e testes
tests/                     testes científicos e smoke simulados
docs/                      rastreabilidade e limitações
.github/workflows/ci.yml     CI (a ser executada no GitHub do usuário)
```

**Escopo:** este repositório é o laboratório **DoE 11–16**. Ele não substitui
nem foi integrado/validado automaticamente com o projeto CNT735 DataLab v4
completo (Assuntos 0–16 e laboratórios transversais).


## Atualização v2.2.1 — Paletas

A paleta **Arco-íris (Plotly `Rainbow`)** voltou a ser a seleção inicial para superfícies 3D, mapas de contorno e mapas de calor. As paletas Viridis e Cividis continuam disponíveis para escolha manual. A mudança é exclusivamente visual e não altera dados, modelos ou resultados numéricos.
