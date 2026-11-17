"""Pergunta, previsão, observação e decisão por assunto; conteúdo editável em PT-BR."""
ROTEIROS = {
    "11": {
        "pergunta": "Por que um modelo 2² com quatro parâmetros passa por todos os pontos?",
        "previsao": "Se você alterar uma das quatro respostas, o que acontecerá com R² e resíduos?",
        "observacao": "Compare XᵀX, os quatro coeficientes, os resíduos e os graus de liberdade.",
        "decisao": "Que réplicas ou validação externa seriam necessárias para defender a predição?",
        "aprofundamento": "Demonstre XᵀX=4I e Efeito=2β para os contrastes em ±1.",
    },
    "12": {
        "pergunta": "A aparente repetibilidade permite estimar variabilidade experimental?",
        "previsao": "O que mudaria se as réplicas fossem medidas independentes em vez de ±5%/−3%?",
        "observacao": "Compare a equação linear com Q; verifique C²=T² na matriz original.",
        "decisao": "Proponha novas condições axiais para estimar C² e T² separadamente.",
        "aprofundamento": "Prove a colinearidade de C² e T² no desenho e identifique a combinação estimável.",
    },
    "13": {
        "pergunta": "Quais variáveis de formulação controlam o avanço da perfuração?",
        "previsao": "Como o efeito aparente de ácido depende da quantidade de SiC e parafina?",
        "observacao": "Investigue resíduos, curvaturas e superfícies em níveis distintos do terceiro fator.",
        "decisao": "Qual região merece ensaio confirmatório independente, e com que critérios?",
        "aprofundamento": "Compare a matriz completa de termos lineares, quadráticos e interações com sua hierarquia.",
    },
    "14": {
        "pergunta": "Concentração de CdS altera só altura ou também a inclinação da superfície?",
        "previsao": "Qual seria a diferença entre superfícies se DC=CE=0?",
        "observacao": "Compare D×E nos dois níveis C; a curvatura D² e E² é compartilhada no modelo atual.",
        "decisao": "Quais interações com quadráticos seriam exigidas para mudar a curvatura?",
        "aprofundamento": "Mostre C²=1 e derive ΔY=2(βC+βDC D+βCE E).",
    },
    "15": {
        "pergunta": "13 corridas preservam precisão e capacidade de prever 27 condições?",
        "previsao": "O RMSE fora do ajuste será maior ou menor que o RMSE de treino?",
        "observacao": "Compare 13 pontos treinados × 14 não utilizados; contrastes no mesmo conjunto, nunca apenas R² treino.",
        "decisao": "Compare o próximo ensaio por divergência preditiva e ganho de informação D condicional.",
        "aprofundamento": "Explique por que 13/27 não é porcentagem de informação retida e o papel da matriz XᵀX.",
    },
    "16": {
        "pergunta": "É possível separar o efeito do fabricante da interação de terceira ordem?",
        "previsao": "O que ocorre com o posto de X ao incluir simultaneamente Bloco e ABC?",
        "observacao": "Veja o cubo, os contrastes ortogonais e as condições em cada lote.",
        "decisao": "Desenhe oito combinações em cada lote e múltiplos lotes por fabricante para inferência geral.",
        "aprofundamento": "Mostre a igualdade colunar ABC=Bloco e explique a consequência para estimabilidade.",
    },
}
