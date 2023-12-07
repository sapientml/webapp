---
title: SapientML
emoji: 📚
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
app_port: 8501
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


## Usage
### Running on local


Please install the required packages.
```
python -m venv venv
. venv/bin/activate
pip install -U pip setuptools
pip install poetry
poetry install
```

#### Starting Streamlit
Perform classification/regression using tabular data.
```
streamlit run sapientml_webapp/main.py
```

Perform classification/regression using tabular data in debug mode.
```
streamlit run sapientml_webapp/main.py -- -d
```


### Running on Docker


```
docker buildx build -t sapientml/sapientml:0.0.1-snapshot
docker run -d sapientml/sapientml:0.0.1-snapshot
```



