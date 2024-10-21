# Estratégia Quantitativa para WIN e WDO

Este projeto implementa uma estratégia quantitativa para operar contratos futuros de mini índice (WIN) e mini dólar (WDO) na B3, utilizando análise técnica e machine learning.

## Funcionalidades

- Utiliza indicadores técnicos como EMA, RSI, MACD e Bollinger Bands
- Implementa um modelo de machine learning (Random Forest) para previsão de direção do mercado
- Realiza backtesting utilizando o framework Backtrader
- Gera relatórios de desempenho incluindo ROI e número de operações

## Como utilizar

1. Instale as dependências:
   ```
   pip install backtrader pandas numpy scikit-learn
   ```

2. Execute o script principal:
   ```
   python estrategia_quantitativa.py
   ```

3. Os resultados do backtesting serão exibidos no console.

## Parâmetros da estratégia

Os principais parâmetros da estratégia podem ser ajustados na classe `EstrategiaQuantitativa`:

- `periodo_media_rapida`: Período da média móvel rápida
- `periodo_media_lenta`: Período da média móvel lenta
- `periodo_rsi`: Período do RSI
- `sobrevenda_rsi`: Nível de sobrevenda do RSI
- `sobrecompra_rsi`: Nível de sobrecompra do RSI
- `stop_loss`: Porcentagem de stop loss
- `take_profit`: Porcentagem de take profit

## Resultados

No último backtesting, a estratégia obteve um ROI de 5.71% com 13 operações realizadas.

## Melhorias futuras

- Otimização de parâmetros
- Implementação de stop móvel
- Incorporação de dados fundamentais
- Validação cruzada para o modelo de machine learning

## Aviso

Esta estratégia é apenas para fins educacionais e não constitui recomendação de investimento. Opere por sua conta e risco.
