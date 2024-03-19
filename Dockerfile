# ENV variables can be overridden on the `docker run` command
# cannot use python 3.11.x due to library `pysha3` breaking
FROM python:3.10.13-bullseye

WORKDIR /usr/src/app

# Install the libs needed by leveldb
RUN apt-get update \
    && apt-get -y install \ 
    build-essential libc6-dev libncurses5-dev libncursesw5-dev libreadline-dev libleveldb-dev

RUN python -m venv venv \
    && venv/bin/pip install plyvel setuptools

COPY requirements.txt ./
RUN venv/bin/pip install -r requirements.txt

COPY electrumx_server electrumx_server
COPY electrumx_rpc electrumx_rpc
COPY electrumx_compact_history electrumx_compact_history
COPY electrumx/ electrumx/

ENV COIN=Bitcoin
ENV NET=mainnet
ENV DB_ENGINE=leveldb
ENV DB_DIRECTORY=/var/lib/electrumx/db
ENV DAEMON_URL="http://username:password@hostname:port"
ENV SERVICES="tcp://0.0.0.0:50010,ws://:50020,rpc://:8000,http://:80"
ENV ALLOW_ROOT=true
ENV CACHE_MB=2000
# set this to true to enable reporting of indexer stats to gaze network, with additional envs. See installation guide for more details.
ENV INDEXER_REPORTING_ENABLED=false

VOLUME /var/lib/electrumx/db

RUN mkdir -p "$DB_DIRECTORY"

CMD ["/usr/src/app/venv/bin/python3", "/usr/src/app/electrumx_server"]
