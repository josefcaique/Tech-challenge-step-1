FROM public.ecr.aws/lambda/python:3.9

COPY docker/fastapi-app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ${LAMBDA_TASK_ROOT}/src

CMD ["src.api.main.handler"]
