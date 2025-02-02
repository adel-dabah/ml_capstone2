FROM public.ecr.aws/lambda/python:3.8
RUN pip install numpy  tflite pipenv requests
RUN pip install keras-image-helper 
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl
RUN pip install --extra-index-url \ 
https://google-coral.github.io/py-repo/ tflite_runtime
COPY ["mri.model.tflite","lambda_fuction.py","./"]
CMD ["lambda_fuction.lambda_handler"]
