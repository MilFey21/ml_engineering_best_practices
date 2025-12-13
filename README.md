# ml_engineering_best_practices

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Telco Customer Churn Prediction

Проект для предсказания оттока клиентов телекоммуникационной компании на основе датасета [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) с Kaggle.

### Описание проекта

Этот проект демонстрирует применение современных инженерных практик в Data Science:
- Структурированная организация проекта по шаблону CookieCutter Data Science
- Управление зависимостями через Pixi
- Качество кода с использованием pre-commit hooks, форматирования и линтеров
- Версионирование данных через DVC (Data Version Control)
- Версионирование моделей через MLflow (Model Registry)
- Контейнеризация с Docker
- Полный ML pipeline от загрузки данных до обучения модели

### Baseline Модель

#### Архитектура модели

Проект реализует **baseline модель** для предсказания оттока клиентов на основе алгоритма **Random Forest Classifier** из библиотеки scikit-learn.

**Архитектура решения:**

1. **Предобработка данных:**
   - Обработка пропущенных значений (заполнение пустых значений в TotalCharges)
   - Кодирование категориальных переменных с помощью LabelEncoder
   - Разделение признаков и целевой переменной (Churn: Yes/No → 1/0)

2. **Модель: Random Forest Classifier**
   - **Алгоритм**: Ансамбль решающих деревьев (100 деревьев)
   - **Глубина деревьев**: max_depth=10 (ограничение для предотвращения переобучения)
   - **Обработка дисбаланса классов**: class_weight="balanced" (автоматическая балансировка классов)
   - **Параллельная обработка**: n_jobs=-1 (использование всех доступных ядер CPU)
   - **Воспроизводимость**: random_state=42

3. **Валидация:**
   - Разделение данных: Train/Test split (80/20)
   - Стратифицированная выборка для сохранения распределения классов
   - Метрики оценки: Accuracy, Precision, Recall, F1-Score, ROC-AUC

4. **Особенности реализации:**
   - Автоматическое определение важности признаков
   - Сохранение обученной модели в формате pickle
   - Сохранение важности признаков в CSV для анализа
   - Детальное логирование метрик и confusion matrix

#### Метрики модели

Результаты обучения baseline модели представлены на скриншоте ниже:

![Classification Report](reports/figures/classification_report.png)

Модель показывает неплохие результаты на задаче предсказания оттока клиентов и может быть использована как отправная точка для дальнейшего улучшения.

### Быстрый старт

1. **Установка зависимостей:**
   ```bash
   pixi install
   # или
   pixi run requirements
   ```

2. **Активация окружения:**
   ```bash
   pixi shell
   ```

3. **Запуск полного pipeline:**
   ```bash
   pixi run pipeline
   ```

   Или пошагово:
   ```bash
   pixi run data      # Загрузка данных
   pixi run features  # Создание признаков
   pixi run train     # Обучение модели
   ```

**Альтернатива:** Используйте `make` команды (если установлен):
```bash
make pipeline
```

**Просмотр всех доступных задач:**
```bash
pixi task list
```

### Настройка Kaggle API

Для загрузки датасета необходимо настроить Kaggle API credentials:

1. Создайте аккаунт на [Kaggle](https://www.kaggle.com/) (если еще нет)
2. Перейдите в Settings -> API -> Create New Token
3. Скачайте `kaggle.json`
4. Поместите файл в:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

**Альтернатива:** Используйте переменные окружения:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

### Использование Docker

Проект включает Dockerfile для контейнеризации приложения. Docker-образ включает:
- Все зависимости через Pixi
- Исходный код проекта
- Данные для обучения (копируются в образ)
- Настроенное окружение DVC и MLflow

**Сборка образа:**
```bash
docker build -t churn-model .
```

**Запуск контейнера:**
```bash
# Обучение модели (контейнер удаляется после выполнения)
docker run --rm churn-model

# Запуск с пробросом порта для MLflow UI
docker run -p 5000:5000 --rm churn-model pixi run mlflow-ui
```

**Примечание:** Данные копируются в образ при сборке. Если данные большие, рассмотрите использование volume mounts:
```bash
docker run --rm -v ${PWD}/data:/app/data churn-model
```

### Версионирование данных и моделей

Проект использует **DVC** для версионирования данных и **MLflow** для версионирования моделей.

**Инициализация DVC:**
```bash
pixi run dvc-init
```

**Версионирование данных:**
```bash
pixi run dvc-track-data  # Версионировать все данные
pixi run dvc-push        # Отправить данные в remote storage
pixi run dvc-pull        # Загрузить данные из remote storage
```

**Просмотр результатов в MLflow:**
```bash
pixi run mlflow-ui       # Запустить MLflow UI (http://localhost:5000)
pixi run mlflow-list-runs  # Список всех запусков
```

**Сравнение моделей:**
```bash
# Через MLflow UI (рекомендуется)
pixi run mlflow-ui
# Откройте http://localhost:5000

# Или через Python API
pixi run python -c "import mlflow; mlflow.set_tracking_uri('file:./mlruns'); runs = mlflow.search_runs(experiment_names=['churn_prediction']); print(runs[['run_id', 'metrics.test_accuracy', 'metrics.test_f1_score']])"
```

Подробные инструкции см. в [документации](docs/docs/reproducibility.md)

### Дополнительные команды

Все команды доступны через `pixi run` (кроссплатформенно):

```bash
pixi run format      # Форматирование кода
pixi run lint        # Проверка линтинга
pixi run test        # Запуск тестов
pixi run clean       # Очистка временных файлов
pixi run setup       # Первоначальная настройка (установка зависимостей + pre-commit)
```

**Альтернатива:** Используйте `make` команды (если установлен):
```bash
make format
make lint
make test
```

**Просмотр всех доступных задач:**
```bash
pixi task list
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands (опционально, для совместимости)
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
