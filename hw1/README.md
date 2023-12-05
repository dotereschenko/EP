# Установка пакетного менеджера для MacOS

``` bash 
brew install poetry
```
Дальше нужно, перейти в нужную папку и сделать ряд действий:

``` bash
poetry new ep_hw1
```

``` bash
poetry add --group=dev black flake8 pylint isort
```

``` bash
poetry add streamlit requests openai
```

#  Развертывание окружения

``` bash
poetry shell
```

# Форматирование и линтнинг кода

``` bash
isort streamlit_app.py
black streamlit_app.py
flake8 
pylint streamlit_app.py
```

``` bash
pre-commit run hw1
```

# Сборка

``` bash
poetry build
```

# Запуск приложения:

``` bash
poetry run streamlit run streamlit_app.py
``` 



