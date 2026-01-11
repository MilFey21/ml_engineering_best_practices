# API Overview

Обзор API проекта для предсказания оттока клиентов.

## Структура API

Проект предоставляет следующие модули:

### Модели

- **[Models API](models.md)** - модули для работы с моделями:
  - `src.modeling.train` - обучение моделей
  - `src.modeling.train_with_hydra` - обучение с использованием Hydra
  - `src.modeling.predict` - предсказания
  - `src.modeling.experiments` - запуск экспериментов
  - `src.modeling.compare_models` - сравнение моделей
  - `src.modeling.clearml_utils` - утилиты для ClearML
  - `src.churn_prediction.model_registry` - реестр моделей

### Утилиты

- **[Utils API](utils.md)** - вспомогательные модули:
  - `src.dataset` - работа с датасетами
  - `src.features` - извлечение признаков
  - `src.plots` - визуализация

## Использование

Все модули используют docstrings в формате Google Style для автоматической генерации документации.
