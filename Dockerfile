FROM python:3.10-buster
RUN pip install -U pip
COPY . .
#RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
EXPOSE 8000