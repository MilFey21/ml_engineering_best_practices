# Инструкции по воспроизведению результатов

## Предварительные требования

1. **Python 3.12**
2. **Pixi** - менеджер пакетов (установка: https://pixi.sh/)
3. **Git** - система контроля версий
4. **Docker** (опционально, для контейнеризации)

## Быстрая настройка

### 1. Клонирование репозитория

```bash
git clone https://github.com/MilFey21/ml_engineering_best_practices.git
cd ml_engineering_best_practices
```

### 2. Установка зависимостей

```bash
# Установка всех зависимостей через Pixi
pixi install

# Или используйте задачу
pixi run requirements
```

### 3. Настройка Kaggle API

Для загрузки данных необходимо настроить Kaggle API credentials:

**Вариант 1: Переменные окружения**

Linux
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

Windows
```bash
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
```

**Вариант 2: Файл kaggle.json**
```bash
# Linux/Mac
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows PowerShell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Copy-Item kaggle.json "$env:USERPROFILE\.kaggle\kaggle.json"
```

---

## Версионирование данных (DVC)

### Инициализация DVC

```bash
# Инициализация DVC (безопасно, можно запускать повторно)
pixi run dvc-init

# Или вручную (если DVC еще не инициализирован)
dvc init --no-scm
dvc remote add -d local ./dvc_storage
```

### Версионирование данных

```bash
# После загрузки данных
pixi run data

# Версионирование данных
pixi run dvc-track-data

# Или вручную
dvc add data/raw/customer_churn.csv
dvc add data/processed/features.csv
dvc add data/processed/labels.csv
```

### Работа с версиями данных

```bash
# Проверка отслеживаемых файлов (работает без Git)
pixi run dvc-check

# Просмотр статуса (требует Git)
dvc status

# Сравнение версий (работает без Git)
pixi run dvc-diff data/raw/customer_churn.csv

# Сравнение версий через Git (требует Git)
dvc diff data/raw/customer_churn.csv

# Переключение на конкретную версию (требует Git)
git checkout <commit_hash> data/raw/customer_churn.csv.dvc
dvc checkout data/raw/customer_churn.csv

# Отправка в remote storage
pixi run dvc-push

# Загрузка из remote storage
pixi run dvc-pull
```

**Примечание:** Команды `dvc status` и `dvc diff` требуют Git для работы. Если DVC инициализирован с `--no-scm`, используйте:
- `pixi run dvc-check` для проверки отслеживаемых файлов
- `pixi run dvc-diff <file>` для проверки соответствия файла его DVC-записи (сравнение хешей)

---

## Версионирование моделей (MLflow)

### Автоматическое логирование

При обучении модели (`pixi run train`) автоматически логируются:
- Параметры модели (n_estimators, max_depth, test_size, random_state)
- Метрики производительности (accuracy, precision, recall, F1-Score, ROC-AUC)
- Модель (зарегистрирована как `RandomForestChurn`)
- Артефакты (feature importance, confusion matrix)

### Просмотр результатов

```bash
# Запуск MLflow UI
pixi run mlflow-ui

# Откройте браузер: http://localhost:5000
```

### Сравнение моделей

**Визуальное сравнение через MLflow UI:**
```bash
pixi run mlflow-ui
# Откройте http://localhost:5000
```

В MLflow UI доступно:
- Список всех запусков эксперимента
- Таблица сравнения метрик и параметров
- Графики метрик для визуального сравнения
- Model Registry для управления версиями моделей

**Сравнение через Python API:**
```bash
pixi run python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
runs = mlflow.search_runs(experiment_names=['churn_prediction'])
print(runs[['run_id', 'metrics.test_accuracy', 'metrics.test_f1_score']])
"
```

### Загрузка модели

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

---

## Полный workflow

### 1. Загрузка и версионирование данных

```bash
# Загрузка данных с Kaggle
pixi run data

# Версионирование данных с DVC
pixi run dvc-track-data

# Коммит версий в Git
git add data/*.dvc .gitignore
git commit -m "Add data version"
```

### 2. Обработка данных и создание признаков

```bash
# Создание признаков
pixi run features

# Версионирование обработанных данных
pixi run dvc-track-data

# Коммит версий в Git
git add data/*.dvc
git commit -m "Update features version"
```

### 3. Обучение модели

```bash
# Обучение модели (автоматически логируется в MLflow)
pixi run train
```

Модель будет автоматически зарегистрирована в MLflow с метриками и параметрами.

### 4. Просмотр результатов

```bash
# Запуск MLflow UI
pixi run mlflow-ui

# Откройте браузер: http://localhost:5000
```

---

## Воспроизведение с Docker

Docker-образ включает все необходимое для воспроизведения результатов:
- Python 3.12 и все зависимости через Pixi
- Исходный код проекта
- Данные для обучения (копируются в образ при сборке)
- Настроенное окружение DVC и MLflow

### Сборка образа

```bash
docker build -t churn-model .
```

При сборке образа:
- Устанавливаются системные зависимости
- Устанавливается Pixi и все Python-зависимости
- Копируются исходный код и данные
- Инициализируется DVC (если необходимо)

### Запуск контейнера

```bash
# Обучение модели (контейнер автоматически удаляется после выполнения благодаря флагу --rm)
docker run --rm churn-model

# Запуск с пробросом порта для MLflow UI
docker run -p 5000:5000 --rm churn-model pixi run mlflow-ui
# Откройте браузер: http://localhost:5000
```

---

## Воспроизведение конкретных версий

### Воспроизведение конкретной версии данных

```bash
# Просмотр версий данных
git log --oneline data/raw/customer_churn.csv.dvc

# Переключение на конкретную версию
git checkout <commit_hash> data/raw/customer_churn.csv.dvc
dvc checkout data/raw/customer_churn.csv
```

### Воспроизведение конкретной версии модели

```bash
# Просмотр зарегистрированных моделей в MLflow
pixi run mlflow-ui

# Загрузка модели по run_id
pixi run python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
model = mlflow.sklearn.load_model('runs:/<run_id>/model')
print('Model loaded successfully')
"
```

## Troubleshooting

### Проблема с DVC

```bash
# Переинициализация DVC
rm -rf .dvc
dvc init --no-scm
dvc remote add -d local ./dvc_storage
```

### Проблема с MLflow

```bash
# Очистка старых запусков (опционально)
rm -rf mlruns/
```

### Проблема с зависимостями

```bash
# Переустановка зависимостей
rm -rf .pixi
pixi install
```

### Проблема с Kaggle API

Если возникают проблемы с загрузкой данных, убедитесь что:
- Файл `kaggle.json` находится в правильной директории
- Права доступа к файлу корректны (Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`)
- Учетные данные действительны

---

## Дополнительные команды

```bash
# Полный pipeline (данные -> признаки -> обучение)
pixi run pipeline

# Форматирование кода
pixi run format

# Проверка линтинга
pixi run lint

# Запуск тестов
pixi run test

# Просмотр всех доступных задач
pixi task list
```
