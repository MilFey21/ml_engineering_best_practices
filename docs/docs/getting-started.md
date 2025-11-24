# Инструкция по настройке проекта

## Быстрая настройка

### 1. Установка Pixi (если еще не установлен)

**Windows (PowerShell):**
```powershell
iwr https://pixi.sh/install.ps1 -useb | iex
```

**Linux/Mac:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Установка зависимостей проекта

```bash
pixi install
```

или

```bash
pixi run requirements
```

или (если установлен make):

```bash
make requirements
```

### 3. Активация окружения

```bash
pixi shell
```

### 4. Настройка pre-commit hooks

```bash
pixi run setup
```

Это автоматически установит зависимости и настроит pre-commit hooks.

Или вручную:
```bash
pixi run pre-commit install
```

Проверка работы hooks:
```bash
pixi run pre-commit run --all-files
```

### 5. Настройка Kaggle API (для загрузки данных)

Для загрузки датасета с Kaggle необходимо настроить API credentials:

1. Создайте аккаунт на Kaggle (если еще нет)
2. Перейдите в Settings -> API -> Create New Token
3. Скачайте `kaggle.json`
4. Поместите файл в `~/.kaggle/kaggle.json` (Linux/Mac) или `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

**Альтернатива:** Можно использовать переменные окружения:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

### 6. Запуск ML Pipeline

**Полный pipeline (кроссплатформенно):**
```bash
pixi run pipeline
```

**Пошагово:**
```bash
pixi run data      # Загрузка данных
pixi run features  # Создание признаков
pixi run train     # Обучение модели
```

**Альтернатива (если установлен make):**
```bash
make pipeline
```

### 7. Форматирование и линтинг кода

**Форматирование (кроссплатформенно):**
```bash
pixi run format
```

**Проверка линтинга:**
```bash
pixi run lint
```

**Альтернатива (если установлен make):**
```bash
make format
make lint
```

### 8. Запуск тестов

```bash
pixi run test
```

**Альтернатива (если установлен make):**
```bash
make test
```

### 9. Просмотр всех доступных задач

```bash
pixi task list
```

### 9. Использование Docker

**Сборка образа:**
```bash
docker build -t churn-model .
```

**Запуск контейнера:**
```bash
docker run churn-model
```

## Структура команд Makefile

Запустите `make help` для просмотра всех доступных команд.

## Troubleshooting

### Проблема с Kaggle API
Если возникают проблемы с загрузкой данных, убедитесь что:
- Файл `kaggle.json` находится в правильной директории
- Права доступа к файлу корректны (Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`)
- Учетные данные действительны

### Проблема с Pixi
Если `pixi` не найден, добавьте путь в PATH:
```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

### Проблема с pre-commit
Если pre-commit не работает, попробуйте:
```bash
pre-commit clean
pre-commit install --install-hooks
```
