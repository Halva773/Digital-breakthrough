import pandas as pd
import numpy as np



def stochastic_oscillator(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет стохастический осциллятор для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для вычисления осциллятора (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями стохастического осциллятора.
    """
    low_min = series.rolling(window=window, min_periods=1).min()
    high_max = series.rolling(window=window, min_periods=1).max()
    stochastic = ((series - low_min) / (high_max - low_min)) * 100
    return stochastic

def relative_strength_index(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Вычисляет индекс относительной силы (RSI) для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для вычисления RSI (по умолчанию 7).

    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями RSI.
    """
    # Вычисляем изменения между последовательными значениями
    delta = series.diff()
    # Вычисляем приросты (gains) и потери (losses)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Средние значения приростов и потерь
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    # Вычисляем RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, short_window: int = 5, long_window: int = 12, signal_window: int = 3) -> pd.DataFrame:
    """
    Вычисляет MACD и сигнальную линию для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - short_window (int): Период для короткой EMA (по умолчанию 5).
    - long_window (int): Период для длинной EMA (по умолчанию 12).
    - signal_window (int): Период для сигнальной линии (по умолчанию 3).

    Возвращает:
    - pd.DataFrame: Датафрейм с колонками MACD и сигнальная линия.
    """
    # Вычисляем экспоненциальные скользящие средние (EMA)
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()

    # MACD линия
    macd_line = ema_short - ema_long

    # Сигнальная линия
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

    # Возвращаем датафрейм с двумя сериями
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line
    })


def commodity_channel_index(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Вычисляет Commodity Channel Index (CCI) для временного ряда.
    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для расчёта CCI (по умолчанию 7).
    Возвращает:
    - pd.Series: Преобразованный временной ряд с значениями CCI.
    """
    # Средняя цена (можно модифицировать для high, low, close)
    typical_price = series
    # Скользящая средняя типичной цены
    sma = typical_price.rolling(window=window, min_periods=1).mean()
    # Среднее отклонение
    mean_deviation = typical_price.rolling(window=window, min_periods=1).apply(
        lambda x: abs((x - x.mean())).mean(), raw=True
    )
    # Вычисление CCI
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci


def exponential_moving_average(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Вычисляет экспоненциальное скользящее среднее (EMA) для временного ряда.
    Параметры:
    - series (pd.Series): Временной ряд.
    - period (int): Период для вычисления EMA (по умолчанию 12).
    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями EMA.
    """
    ema = series.ewm(span=window, adjust=False).mean()
    return ema


def rate_of_change(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет индекс скорости изменения (ROC) для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - period (int): Период для вычисления ROC (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями ROC.
    """
    roc = (series.diff(periods=window) / series.shift(periods=window))
    return roc


def keltner_channel(series: pd.Series, window: int = 14, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Вычисляет динамический канал Кельтнера для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для расчета ATR и EMA (по умолчанию 14).
    - multiplier (float): Множитель для расчета каналов (по умолчанию 1.5).

    Возвращает:
    - pd.DataFrame: Центральная линия (EMA), верхний и нижний каналы.
    """
    # Расчет ATR (Средний истинный диапазон)
    high = series.rolling(window=window, min_periods=1).max()
    low = series.rolling(window=window, min_periods=1).min()
    close = series
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1)
    atr = tr.max(axis=1).rolling(window=window, min_periods=1).mean()

    # Расчет EMA
    ema = series.ewm(span=window, min_periods=1).mean()

    # Верхний и нижний каналы
    upper_band = ema + (multiplier * atr)
    lower_band = ema - (multiplier * atr)

    return pd.DataFrame({'EMA': ema, 'Upper Band': upper_band, 'Lower Band': lower_band})

def tenkan_sen(series: pd.Series, window: int = 9) -> pd.Series:
    """
    Вычисляет линию Tenkan-sen (Conversion Line) для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для расчета (по умолчанию 9).

    Возвращает:
    - pd.Series: Линия Tenkan-sen.
    """
    # Расчет максимума и минимума за период окна
    high = series.rolling(window=window, min_periods=1).max()
    low = series.rolling(window=window, min_periods=1).min()

    # Расчет Tenkan-sen
    tenkan = (high + low) / 2
    return tenkan

def tema(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет Трёхлинейный Пробой (TEMA) для временного ряда.

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для вычисления (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд с TEMA.
    """
    # Вычисляем экспоненциальные скользящие средние
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()

    # Вычисляем TEMA
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema

def cmo(series, window=100):
    delta = series.diff()
    # Вычисляем gain и loss
    gain = np.where(delta > 0, delta, 0)  # Только положительные изменения
    loss = np.where(delta < 0, -delta, 0)  # Только отрицательные изменения
    # Вычисляем CMO
    rolling_gain = pd.Series(gain).rolling(window=window).sum()
    rolling_loss = pd.Series(loss).rolling(window=window).sum()
    cmo = 100 * (rolling_gain - rolling_loss) / (rolling_gain + rolling_loss)
    return cmo

def detrended_price_oscillator(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет осциллятор цены без тренда (DPO).

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для сдвига (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд со значениями DPO.
    """
    # Сдвигаем временной ряд на заданное количество периодов (окно)
    shifted_series = series.shift(window)
    # Разница между текущей ценой и сдвинутой ценой
    dpo = series - shifted_series
    return dpo

def momentum_indicator(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Вычисляет индикатор импульса (Momentum Indicator).

    Параметры:
    - series (pd.Series): Временной ряд.
    - window (int): Период для вычисления индикатора импульса (по умолчанию 14).

    Возвращает:
    - pd.Series: Преобразованный временной ряд с значениями индикатора импульса.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Входные данные должны быть pd.Series")

    # Индикатор импульса: разница между текущей ценой и ценой за n периодов назад
    momentum = series - series.shift(window)
    return momentum

def generate_features(dataset, windows=[100], columns=['FrL', 'FrR', 'OcR']):
    for column in columns:
        for window in windows:
            dataset[f"{column}_stachostic_{window}"] = stochastic_oscillator(dataset[column], window=window)
            dataset[f"{column}_RSI_{window}"] = relative_strength_index(dataset[column], window=window)
            macd_colmns = macd(dataset[column]).rename(columns={'MACD': f'{column}_MACD_{window}', 'Signal': f'{column}_Signal_{window}'})
            pd.concat([dataset, macd_colmns], axis=1)
            dataset[f"{column}_EMA_{window}"] = exponential_moving_average(dataset[column], window=window)
            dataset[f"{column}_RoC_{window}"] = rate_of_change(dataset[column], window=window)
            keltner_columns = keltner_channel(dataset[column], window=window).rename(columns={'EMA': f"{column}_EMA_{window}",
                                                                                             "Upper Band": f"{column}_UpperBand_{window}",
                                                                                             "Lower Band": f"{column}_LowerBand_{window}"})
            pd.concat([dataset, keltner_columns], axis=1)
            dataset[f"{column}_tenkan_sen_{window}"] = tenkan_sen(dataset[column], window=window)
            dataset[f"{column}_TEMA_{window}"] = tema(dataset[column], window=window)
            dataset[f"{column}_CMO_{window}"] = cmo(dataset[column], window=window)
            dataset[f"{column}_DPO_{window}"] = detrended_price_oscillator(dataset[column], window=window)
            dataset[f"{column}_momentum_{window}"] = momentum_indicator(dataset[column], window=window)
        # dataset[f"{column}_CCI_{window}"] = commodity_channel_index(dataset[column]) # Считается долго
    return dataset


def add_base_features(data, window_sizes=[400]):
    for window in window_sizes:
        data[f'mean_window_{window}_FrL'] = data['FrL'].rolling(window=window).mean()
        data[f'mean_window_{window}_FrR'] = data['FrR'].rolling(window=window).mean()
        data[f'mean_window_{window}_OcR'] = data['OcR'].rolling(window=window).mean()
        data[f'min_window_{window}_FrL'] = data['FrL'].rolling(window=window).min()
        data[f'max_window_{window}_FrL'] = data['FrL'].rolling(window=window).max()
        data[f'min_window_{window}_FrR'] = data['FrR'].rolling(window=window).min()
        data[f'max_window_{window}_FrR'] = data['FrR'].rolling(window=window).max()
        data[f'min_window_{window}_OcR'] = data['OcR'].rolling(window=window).min()
        data[f'max_window_{window}_OcR'] = data['OcR'].rolling(window=window).max()
        data[f'corr_window_{window}_FrL_FrR'] = data['FrL'].rolling(window=window).corr(data['FrR'])
        data[f'corr_window_{window}_FrL_OcR'] = data['FrL'].rolling(window=window).corr(data['OcR'])
        data[f'corr_window_{window}_FrR_OcR'] = data['FrR'].rolling(window=window).corr(data['OcR'])
    return data

if __name__ == '__main__':
    data = pd.read_csv('marked_dataset.csv', index_col=0)
    print(data.shape)
    fish_genered_data = generate_features(data)[10000:-10000]
    print(fish_genered_data.shape)
    print(fish_genered_data.head())