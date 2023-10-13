
FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv
RUN pip install gunicorn


WORKDIR /app                                                                

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["predictQ6.py", "dv.bin", "model1.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predictQ6:app"]