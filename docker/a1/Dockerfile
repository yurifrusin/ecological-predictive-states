# Preparation only: context contains the exact reviewed source.bundle.
FROM ghcr.io/astral-sh/uv@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 AS uv
FROM python@sha256:35d3a4a3d5e42e02ab916d44513a050689f12c0533d45598d229672503fe77ca
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install --yes --no-install-recommends git libosmesa6 libgl1-mesa-dri libglfw3 && rm -rf /var/lib/apt/lists/*
ARG SOURCE_HEAD
ARG SOURCE_TREE
ARG SOURCE_URL
COPY source.bundle /preparation/source.bundle
RUN git clone /preparation/source.bundle /source && cd /source && git checkout --detach "$SOURCE_HEAD" && test "$(git rev-parse HEAD^{tree})" = "$SOURCE_TREE" && git remote set-url origin "$SOURCE_URL" && test -z "$(git status --porcelain)"
WORKDIR /source
RUN mkdir /retained
RUN uv python install 3.11.15 && uv sync --locked --python 3.11.15 && .venv/bin/python -c 'import platform,importlib.metadata as m; assert platform.python_version()=="3.11.15"; assert all(m.version(k)==v for k,v in {"mujoco":"3.12.0","numpy":"2.4.6","PyOpenGL":"3.1.10","glfw":"2.10.2"}.items())'
ENV PATH="/source/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 MESA_SHADER_CACHE_DISABLE=true GIT_OPTIONAL_LOCKS=0
LABEL eps.a1.source-head=$SOURCE_HEAD eps.a1.source-tree=$SOURCE_TREE
