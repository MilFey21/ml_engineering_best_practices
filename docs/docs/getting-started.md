# Начало работы

Это руководство поможет вам установить и настроить проект для предсказания оттока клиентов.

## Требования

- Python 3.12+
- Git
- [Pixi](https://pixi.sh) (рекомендуется) или pip/conda
- Docker (опционально, для ClearML Server)

## Установка Pixi

### Windows

```powershell
iwr https://pixi.sh/install.ps1 | iex
```

### Linux/Mac

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Клонирование репозитория

```bash
git clone https://github.com/MilFey21/ml_engineering_best_practices.git
cd ml_engineering_best_practices
```

## Установка зависимостей

```bash
# Установка всех зависимостей
pixi install

# Активация окружения
pixi shell
```

## Настройка Kaggle API

Для загрузки датасета необходимо настроить Kaggle API credentials:

1. Создайте аккаунт на [Kaggle](https://www.kaggle.com/)
2. Перейдите в **Settings → API → Create New Token**
3. Скачайте `kaggle.json`
4. Поместите файл в `~/.kaggle/kaggle.json` (Linux/Mac) или `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

**Альтернатива:** Используйте переменные окружения:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

## Настройка ClearML (опционально)

```bash
# Запуск локального ClearML Server
pixi run clearml-server-start

# Настройка credentials
clearml-init
```

При настройке укажите:
- **API Server**: `http://localhost:8008`
- **Web UI**: `http://localhost:8080`
- **Access Key** и **Secret Key**: получите из Web UI

## Примеры использования

### Базовое обучение модели

```bash
# Полный pipeline
pixi run pipeline

# Или пошагово
pixi run data      # Загрузка данных
pixi run features  # Создание признаков
pixi run train     # Обучение модели
```

### Запуск экспериментов

```bash
# Запуск множественных экспериментов
pixi run churn-experiments

# Сравнение моделей
pixi run churn-model-registry compare --metric-name test_f1_score --top-n 5
```

### Генерация отчетов

```bash
# Генерация отчета об экспериментах
pixi run generate-report

# Генерация всех отчетов
pixi run generate-all-reports
```

### Использование DVC

```bash
# Инициализация DVC
pixi run dvc-init

# Версионирование данных
pixi run dvc-track-data

# Воспроизведение pipeline
pixi run dvc-repro
```

## Следующие шаги

- [Руководство по развертыванию](deployment.md)
- [Результаты экспериментов](experiments/results.md)
- [Воспроизводимость](reproducibility/instructions.md)
