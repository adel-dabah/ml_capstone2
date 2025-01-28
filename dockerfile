FROM python:3.8.11-slim
RUN pip install pipenv
WORKDIR /app
RUN ls
COPY ["Pipfile","Pipfile.lock","MRI_v1_09_0.932.keras","./"]
RUN pipenv  install --system --deploy
