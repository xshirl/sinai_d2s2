# FROM python

FROM debian:stable



RUN apt-get update && \
    apt-get -y install \
        nginx \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        vim \
        uwsgi-core



RUN pip3 install -Iv uwsgi Flask

ADD requirements.txt /requirements.txt
RUN pip3 install -Ivr /requirements.txt

VOLUME /ssl
EXPOSE 5000
ENV SERVERNAME "amp.pharm.mssm.edu"

ADD . /app
RUN chmod -R 777 /app
CMD /app/boot.sh