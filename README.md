# Цифровой прорыв. Хакактон 2024.
![image](https://github.com/user-attachments/assets/d8d69951-30f4-4b79-8e57-488614132a5f)
## Описание кейса
Сегодня фундаментальная и клиническая наука рассматривает сон как сложный физиологический процесс, который обеспечивает протекание процессов восстановления организма. Структура сна здорового человека хорошо известна. Большинство системных заболеваний сопровождаются нарушением структуры сна. Анализ сна и его нарушений является ключевым элементом в диагностике и прогнозировании различных психических и нервных заболеваний. Изучение структуры сна позволяет не только диагностировать существующие расстройства, но и предсказывать их развитие. Существует тесная взаимосвязь между патогенезом генерализованных эпилепсий и нарушениями сна на уровне таламо-кортикальной системы. Крысы WAG/Rij являются надежной моделью абсанс-эпилепсии человека и широко используются в доклинических исследованиях.
Участникам хакатона предлагается создать программный модуль для распознавания фазы глубокого сна и промежуточной фазы сна по данным электрокортикограмм у крыс WAG/Rij, используемых в доклинических исследованиях абсанс-эпилепсии.

## Используемый стек
pandas, numpy, sklearn, tensorflow
CNN, LSTM, Ансамблирование
### Итоговое решение
![image](https://github.com/user-attachments/assets/46c201a8-51a8-4680-9c06-0aa28a056d98)

## Структура репозитория
- `/data` - исходные, частично размеченные данные от организаторов
- `cnnmodel.ipynb` - Лучшая получившаяся модель с точностью 0.92
- `ensemble_classification.ipynb` - Модель ансамбля на HGBR, CatBoost и XGB_Model с решающей моделью логистической регрессии
- `feature_generating.py` and `feature_generator.ipynb` - Генерация параметров по временным рядам
- `load_data.py` and `read_dataset.ipynb` - Чтение и разметка EDF данных
- `model.ipynb` - Версия модели однослойной BiLSTM (0.9059 accuracy)
- `selected_data.csv` and `to_train_data.csv` - Отобранные для обучения данные
- `use_model.ipynb` - Использование загруженной модели и проверка по метрикам
- `valid_file.csv` - файл на валидацию модели


## References
- Metrics: https://github.com/maxto/Time-Series-Clustering
- MP-SeizNet: A Multi-Path CNN Bi-LSTM Network for Seizure-Type Classification Using EEG: https://arxiv.org/abs/2211.04628
- SWD-Detection-in-Humans: https://github.com/Berken-demirel/SWD_Detect
- [Automatic detection of the spike-and-wave discharges in absence epilepsy for humans and rats using deep learning](https://www.sciencedirect.com/science/article/abs/pii/S1746809422002488?via%3Dihub)
- Предсказание временных рядов: [PDTrans](https://github.com/JL-tong/PDTrans)
- Алгоритм декомпозиции временных рядов: [RobustSTL](https://github.com/LeeDoYup/RobustSTL)
### Готовые решения
- https://github.com/TimeEval/evaluation-paper
- https://github.com/ikatsov/tensor-house/blob/master/smart-manufacturing/anomaly-detection-time-series.ipynb
- https://github.com/unit8co/darts
- https://github.com/thuml/Anomaly-Transformer
