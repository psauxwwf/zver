FROM debian:13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    TZ=UTC

RUN apt-get update --quiet --quiet && \
    apt-get install --quiet --quiet --yes \
    --no-install-recommends --no-install-suggests \
    curl git libsqlite3-0 ca-certificates \
    && apt-get --quiet --quiet clean \
    && rm --recursive --force /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /zver

COPY pyproject.toml uv.lock .python-version ./

RUN uv python install cpython-3.12.13-linux-x86_64-gnu

RUN uv sync

COPY . .

ENTRYPOINT ["uv","run","main.py"]

CMD ["mcp"]
