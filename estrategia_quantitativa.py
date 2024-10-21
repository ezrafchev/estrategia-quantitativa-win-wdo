
import backtrader as bt
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class EstrategiaQuantitativa(bt.Strategy):
    params = (
        ('periodo_media_rapida', 5),
        ('periodo_media_lenta', 20),
        ('periodo_rsi', 14),
        ('sobrevenda_rsi', 35),
        ('sobrecompra_rsi', 65),
        ('periodo_macd', 26),
        ('periodo_sinal', 9),
        ('stop_loss', 0.02),
        ('take_profit', 0.04),
        ('atr_period', 14),
        ('atr_multiplier', 1.5),
        ('bollinger_period', 20),
        ('bollinger_dev', 2),
    )

    def __init__(self):
        self.media_rapida = bt.indicators.EMA(period=self.params.periodo_media_rapida)
        self.media_lenta = bt.indicators.EMA(period=self.params.periodo_media_lenta)
        self.cruzamento = bt.indicators.CrossOver(self.media_rapida, self.media_lenta)
        self.rsi = bt.indicators.RSI(period=self.params.periodo_rsi)
        self.macd = bt.indicators.MACD(period_me1=12, period_me2=self.params.periodo_macd, period_signal=self.params.periodo_sinal)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        self.bollinger = bt.indicators.BollingerBands(period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)
        
        # Machine Learning
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X = []
        self.y = []
        
        # Contador de operações
        self.trades = 0
        
        # Preço de entrada
        self.entry_price = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def next(self):
        # Coleta de dados para ML
        features = [
            self.data.close[0],
            self.media_rapida[0],
            self.media_lenta[0],
            self.rsi[0],
            self.macd.macd[0],
            self.macd.signal[0],
            self.atr[0],
            self.bollinger.lines.top[0],
            self.bollinger.lines.mid[0],
            self.bollinger.lines.bot[0]
        ]
        self.X.append(features)
        
        if len(self.X) > 1:
            self.y.append(1 if self.data.close[0] > self.data.close[-1] else 0)
        
        if len(self.X) > 50:  # Treinar o modelo a cada 50 candles
            X_train, X_test, y_train, y_test = train_test_split(self.X[:-1], self.y, test_size=0.2, random_state=42)
            self.clf.fit(X_train, y_train)
            
            prediction = self.clf.predict([self.X[-1]])[0]
            
            self.log(f'Close: {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}, Cruzamento: {self.cruzamento[0]}, Prediction: {prediction}, ATR: {self.atr[0]:.2f}')
            
            if not self.position:
                if prediction == 1 and (self.cruzamento > 0 or self.rsi < self.params.sobrevenda_rsi or self.data.close[0] < self.bollinger.lines.bot[0]):
                    self.buy()
                    self.entry_price = self.data.close[0]
                    self.trades += 1
                    self.log(f'BUY EXECUTED, Price: {self.data.close[0]:.2f}')
            else:
                # Verificar stop loss e take profit
                stop_loss = self.entry_price * (1 - self.params.stop_loss)
                take_profit = self.entry_price * (1 + self.params.take_profit)
                
                if self.data.close[0] <= stop_loss or                    self.data.close[0] >= take_profit or                    (prediction == 0 and (self.cruzamento < 0 or self.rsi > self.params.sobrecompra_rsi or self.data.close[0] > self.bollinger.lines.top[0])):
                    self.close()
                    self.trades += 1
                    self.log(f'SELL EXECUTED, Price: {self.data.close[0]:.2f}')
                    self.entry_price = None

def run_strategy(data):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EstrategiaQuantitativa)
    cerebro.adddata(data)
    cerebro.broker.setcash(500.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% de comissão
    
    print('Saldo Inicial: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Saldo Final: %.2f' % cerebro.broker.getvalue())
    
    # Cálculo de métricas adicionais
    strategy = results[0]
    roi = (cerebro.broker.getvalue() / 500.0 - 1.0) * 100
    print(f'ROI: {roi:.2f}%')
    print(f'Número de operações: {strategy.trades}')
    
    # Desativando a plotagem
    # cerebro.plot()

# Simulação de dados para WIN e WDO (substitua por dados reais quando disponíveis)
def create_mock_data(symbol, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(date_range)).cumsum() + 100,
        'high': np.random.randn(len(date_range)).cumsum() + 101,
        'low': np.random.randn(len(date_range)).cumsum() + 99,
        'close': np.random.randn(len(date_range)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(date_range))
    }, index=date_range)
    return bt.feeds.PandasData(dataname=data, name=symbol)

if __name__ == '__main__':
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime(2023, 1, 1)
    
    win_data = create_mock_data('WIN', start_date, end_date)
    wdo_data = create_mock_data('WDO', start_date, end_date)
    
    print("Executando estratégia para WIN:")
    run_strategy(win_data)
    
    print("\nExecutando estratégia para WDO:")
    run_strategy(wdo_data)

    # Desativar a plotagem para economizar tempo
    bt.Cerebro.plot = lambda self: None
