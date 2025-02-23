# MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

# Описание проекта

Этот проект предназначен для загрузки, предобработки, обучения и инференса моделей машинного обучения, включая экспорт в формат ONNX.

## Установка

1. Клонируйте репозиторий:
   ```sh
   git clone <репозиторий>
   cd <папка_репозитория>
   ```

2. Установите зависимости:
   ```sh
   pip install -r requirements.txt
   ```

3. Запустите обучение модели:
   ```sh
   python -m mlops.modeling.train main
   ```

4. Для инференса используйте:
   ```sh
   python -m mlops.modeling.predict main
   ```

## Экспорт модели в ONNX

### Описание
Этот проект включает функциональность экспорта обученной модели в формат ONNX.

### Инструкции по экспорту

1. Обучите модель, запустив `train.py`.
2. После завершения обучения модель автоматически экспортируется в ONNX.
3. Файл модели будет сохранен в `models/model.onnx`.

### Требования
- `torch`
- `torchvision`
- `onnx`

### Проверка экспорта
Вы можете проверить экспортированную модель с помощью:
```python
import onnx
model = onnx.load("models/model.onnx")
onnx.checker.check_model(model)
print("Модель успешно загружена и проверена!")
```



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
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
│                         mlops and configuration for tools like black
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
└── mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops a Python module
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

