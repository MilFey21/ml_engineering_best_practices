# Инструкции по воспроизведению

Этот раздел содержит подробные инструкции по воспроизведению результатов проекта.

## Предварительные требования

- Python 3.12+
- Git
- [Pixi](https://pixi.sh) или pip/conda
- Docker (опционально, для ClearML Server)

## Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/your-username/ml_engineering_best_practices.git
cd ml_engineering_best_practices
```

## Шаг 2: Установка зависимостей

```bash
pixi install
pixi shell
```

## Шаг 3: Настройка окружения

### Настройка Kaggle API

1. Создайте файл `~/.kaggle/kaggle.json` с вашими credentials
2. Установите права доступа: `chmod 600 ~/.kaggle/kaggle.json`

### Настройка ClearML (опционально)

```bash
pixi run clearml-server-start
clearml-init
```

## Шаг 4: Воспроизведение pipeline

### Полный pipeline

```bash
pixi run pipeline
```

### Пошаговое выполнение

```bash
# 1. Загрузка данных
pixi run data

# 2. Создание признаков
pixi run features

# 3. Обучение модели
pixi run train
```

## Шаг 5: Воспроизведение экспериментов

```bash
pixi run churn-experiments
```

## Проверка результатов

### Проверка данных

```bash
# Проверка загруженных данных
ls -lh data/raw/

# Проверка обработанных данных
ls -lh data/processed/
```

### Проверка моделей

```bash
# Проверка обученных моделей
ls -lh models/

# Просмотр метрик
pixi run churn-model-registry compare --metric-name test_f1_score --top-n 5
```

### Генерация отчетов

```bash
# Генерация отчета об экспериментах (с графиками)
pixi run generate-report

# Генерация без графиков
pixi run generate-report --no-plots

# Генерация всех отчетов
pixi run generate-all-reports
```

**Особенности генерации отчетов:**
- Автоматическое скачивание графиков из ClearML для топовых экспериментов
- Визуализация метрик обучения и confusion matrix
- Сохранение графиков в `reports/experiments/plots/`
- Встраивание графиков в markdown-отчеты

Отчеты сохраняются в `reports/experiments/experiment_report.md`.

## Воспроизводимость через DVC

Если данные версионированы через DVC:

```bash
# Загрузка данных из DVC
pixi run dvc-pull

# Воспроизведение pipeline
pixi run dvc-repro
```

## Воспроизводимость через Docker

```bash
# Сборка образа
docker build -t churn-model .

# Запуск pipeline
docker run --rm churn-model
```

## Устранение проблем

### Проблема: Разные результаты

**Возможные причины:**
- Разные версии библиотек
- Разные seed значения
- Разные данные

**Решение:**
- Используйте `pixi install` для установки точных версий
- Убедитесь, что `random_state=42` установлен везде
- Проверьте версию данных через DVC

### Проблема: Ошибки зависимостей

**Решение:**
- Убедитесь, что используете Python 3.12+
- Переустановите зависимости: `pixi install`

## Дополнительные ресурсы

- [Руководство по развертыванию](../deployment.md)
- [Результаты экспериментов](../experiments/results.md)
- [Начало работы](../getting-started.md)

