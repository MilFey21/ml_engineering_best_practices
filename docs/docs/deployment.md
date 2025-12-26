# Руководство по развертыванию

Это руководство описывает различные способы развертывания проекта предсказания оттока клиентов.

## Локальное развертывание

### Предварительные требования

- Python 3.12+
- Git
- [Pixi](https://pixi.sh) (рекомендуется)

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/ml_engineering_best_practices.git
cd ml_engineering_best_practices

# Установка зависимостей
pixi install
pixi shell
```

### Настройка окружения

1. **Настройка Kaggle API:**
   ```bash
   # Создайте файл ~/.kaggle/kaggle.json с вашими credentials
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Запуск pipeline:**
   ```bash
   # Полный pipeline
   pixi run pipeline
   
   # Или пошагово
   pixi run data
   pixi run features
   pixi run train
   ```

## Docker развертывание

### Предварительные требования

- Docker 20.10+
- Docker Compose 2.0+ (опционально)

### Быстрый старт

```bash
# Сборка образа
docker build -t churn-model .

# Запуск контейнера
docker run --rm churn-model

# С монтированием томов для сохранения данных
docker run --rm \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  churn-model
```

### Использование Docker Compose

```bash
# Запуск проекта
docker-compose up

# Запуск в фоновом режиме
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## ClearML Server

### Предварительные требования

- Docker 20.10+
- Docker Compose 2.0+
- Минимум 4GB свободной памяти

### Запуск ClearML Server

```bash
# Windows
pixi run clearml-server-start

# Linux/Mac
pixi run clearml-server-start-unix
```

Или вручную:

```bash
docker-compose -f docker-compose.clearml.yml up -d
```

### Настройка credentials

1. Откройте Web UI: http://localhost:8080
2. Перейдите в **Settings → Workspace → Create new credentials**
3. Скопируйте Access Key и Secret Key
4. Настройте credentials:

```bash
clearml-init
```

При настройке укажите:
- **API Server**: `http://localhost:8008`
- **Web UI**: `http://localhost:8080`
- **Access Key** и **Secret Key**: из Web UI

### Использование

```bash
# Запуск экспериментов
pixi run churn-experiments

# Версионирование данных
pixi run churn-data-versioning

# Регистрация моделей
pixi run churn-model-registry

# Сравнение моделей
pixi run churn-model-registry compare --metric-name test_f1_score --top-n 5
```

### Просмотр результатов

Откройте http://localhost:8080 в браузере и перейдите в:
- **Projects** → "Churn Prediction" - просмотр всех экспериментов
- **Experiments** - список экспериментов с метриками
- **Models** - зарегистрированные модели
- **Datasets** - версионированные датасеты

### Остановка сервера

```bash
docker-compose -f docker-compose.clearml.yml down
```

## Устранение неполадок

### Проблема: Ошибка при загрузке данных

**Решение:** Проверьте настройки Kaggle API и убедитесь, что файл `kaggle.json` находится в правильном месте.

### Проблема: Ошибки зависимостей

**Решение:** Убедитесь, что используете правильную версию Python (3.12+) и все зависимости установлены.

### Проблема: Сервер не запускается

**Решение:** Проверьте логи: `pixi run clearml-server-logs` и убедитесь, что порты не заняты.

