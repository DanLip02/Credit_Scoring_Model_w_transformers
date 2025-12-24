#BASIC EXAMPLE OF SIMPLE DOCKERFILE (BASE FOR OTHER DOCKERFILES)


# load Python 3.9
FROM python:3.9

# environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# work dir
WORKDIR /backend

# copy depend
COPY requirements.txt /backend/

# upd pip
RUN pip3 install --upgrade pip --no-cache-dir \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org

# Load depend
RUN pip3 install --no-cache-dir -r requirements.txt \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org

# load system depend and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 wget gnupg2 unzip curl \
    && rm -rf /var/lib/apt/lists/*

# copy project
COPY . /backend/

# Port uvicorn
EXPOSE 8080

# start point for uvicorn
ENTRYPOINT ["uvicorn"]
# example run:
# docker run -p 8080:8080 <image_name> uvicorn main:app --host 0.0.0.0 --port 8080
