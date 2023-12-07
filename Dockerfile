# Copyright 2023 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM debian:12-slim AS build
ARG POETRY_VIRTUALENVS_CREATE=false
RUN apt-get update -qqy && apt-get install --no-install-suggests --no-install-recommends -y \
    wget \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /
RUN python3 -m venv /venv
RUN /venv/bin/pip install -U pip setuptools
RUN /venv/bin/pip install poetry
COPY pyproject.toml poetry.lock /
RUN /venv/bin/poetry install --no-root --without=dev
RUN wget -P /venv/lib/python3.11/site-packages/sapientml_preprocess/lib https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
RUN mkdir /app
COPY sapientml_webapp /app/

# Copy /venv and /app into a distroless image
FROM gcr.io/distroless/python3-debian12
ENV PATH="/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS false
ENV OMP_NUM_THREADS=1
ENV JUPYTER_DATA_DIR="/app/jupyter"
COPY --chown=1000:1000 --from=build /venv /venv
COPY --chown=1000:1000 --from=build /app /app
WORKDIR /app

ENTRYPOINT ["python", "entrypoint.py"]
