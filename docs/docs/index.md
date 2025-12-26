# ML Engineering Best Practices - Churn Prediction

Добро пожаловать в документацию проекта **предсказания оттока клиентов телекоммуникационной компании**!

## О проекте

Этот проект демонстрирует применение современных инженерных практик в Data Science и Machine Learning:

- 🏗️ **Структурированная организация проекта** по шаблону CookieCutter Data Science
- 📦 **Управление зависимостями** через Pixi
- ✅ **Качество кода** с использованием pre-commit hooks, форматирования и линтеров
- 📊 **Версионирование данных** через DVC (Data Version Control)
- 🤖 **Версионирование моделей** через ClearML Model Registry
- 🐳 **Контейнеризация** с Docker
- 🔄 **Полный ML pipeline** от загрузки данных до обучения модели
- 📈 **Отслеживание экспериментов** через ClearML

## Быстрый старт

```bash
# 1. Установка зависимостей
pixi install

# 2. Запуск полного pipeline
pixi run pipeline

# 3. Просмотр результатов
pixi run churn-model-registry list
```

## Основные возможности

### 🎯 Предсказание оттока клиентов

Проект реализует модели машинного обучения для предсказания оттока клиентов на основе датасета [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) с Kaggle.

### 📚 Поддерживаемые модели

- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM
- K-Nearest Neighbors
- Decision Tree
- AdaBoost
- Naive Bayes

### 🔧 Инструменты

- **Pixi** - управление зависимостями и окружениями
- **DVC** - версионирование данных
- **ClearML** - отслеживание экспериментов и управление моделями
- **Docker** - контейнеризация
- **Hydra** - управление конфигурациями
- **Pre-commit** - проверка качества кода

## Структура документации

- **[Начало работы](getting-started.md)** - установка, настройка и примеры использования
- **[Руководство по развертыванию](deployment.md)** - инструкции по развертыванию (локальное, Docker, ClearML)
- **[Результаты экспериментов](experiments/results.md)** - результаты экспериментов и генерация отчетов
- **[Воспроизводимость](reproducibility/instructions.md)** - инструкции по воспроизведению результатов

## Отчеты

Отчеты о выполнении заданий доступны в репозитории GitHub в директории [`reports/`](https://github.com/your-username/ml_engineering_best_practices/tree/main/reports):
- [HW1 Report](https://github.com/your-username/ml_engineering_best_practices/blob/main/reports/hw1_report.md)
- [HW2 Report](https://github.com/your-username/ml_engineering_best_practices/blob/main/reports/hw2_report.md)
- [HW3 Report](https://github.com/your-username/ml_engineering_best_practices/blob/main/reports/hw3_report.md)
- [HW4 Report](https://github.com/your-username/ml_engineering_best_practices/blob/main/reports/hw4_report.md)
- [HW5 Report](https://github.com/your-username/ml_engineering_best_practices/blob/main/reports/hw5_report.md)
- [HW6 Report](https://github.com/your-username/ml_engineering_best_practices/blob/main/reports/hw6_report.md)
- [Experiment Report](https://github.com/your-username/ml_engineering_best_practices/blob/main/reports/experiments/experiment_report.md)

## Полезные ссылки

- [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [ClearML Documentation](https://clear.ml/docs)
- [DVC Documentation](https://dvc.org/doc)
- [Pixi Documentation](https://pixi.sh)

## Лицензия

Проект создан в рамках курса ML Engineering в ITMO.
