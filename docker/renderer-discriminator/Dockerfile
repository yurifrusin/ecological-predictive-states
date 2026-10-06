# Isolated prospective recipe. Building/preparation require separate authorization.
FROM ghcr.io/astral-sh/uv@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 AS uv
FROM python@sha256:35d3a4a3d5e42e02ab916d44513a050689f12c0533d45598d229672503fe77ca
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install --yes --no-install-recommends git libosmesa6 libgl1-mesa-dri libglfw3 && mkdir /preparation && dpkg-query -W > /preparation/os-packages.txt && rm -rf /var/lib/apt/lists/*
ARG SOURCE_HEAD
ARG SOURCE_TREE
ARG SOURCE_URL=https://github.com/yurifrusin/ecological-predictive-states.git
COPY source.bundle /preparation/source.bundle
RUN git clone /preparation/source.bundle /source && cd /source && git checkout --detach "$SOURCE_HEAD" && test "$(git rev-parse HEAD^{tree})" = "$SOURCE_TREE" && git remote set-url origin "$SOURCE_URL" && test -z "$(git status --porcelain)"
WORKDIR /source
RUN uv python install 3.11.15 && uv sync --locked --python 3.11.15 && .venv/bin/python -c 'import platform,importlib.metadata as m; assert platform.python_version()=="3.11.15"; assert all(m.version(k)==v for k,v in {"mujoco":"3.12.0","numpy":"2.4.6","PyOpenGL":"3.1.10","glfw":"2.10.2"}.items())'
ENV PATH="/source/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 MESA_SHADER_CACHE_DISABLE=true GIT_OPTIONAL_LOCKS=0
RUN SOURCE_HEAD="$SOURCE_HEAD" SOURCE_TREE="$SOURCE_TREE" python scripts/renderer_discriminator_execution.py --prepare && rm /preparation/source.bundle
ARG CONFIG_FILE=a3a72ef5210582ef67091b2905d17d2c9de9db0c110d1d912f967d91838aabc4
ARG ASSET_ROOT
ARG MEMBERSHIP_ROOT
ARG PROTECTION_ROOT
ARG SCENE_ROOT
LABEL eps.renderer-discriminator.version=renderer_illumination_sampling_execution_v1 eps.renderer-discriminator.source-url=$SOURCE_URL eps.renderer-discriminator.source-head=$SOURCE_HEAD eps.renderer-discriminator.source-tree=$SOURCE_TREE eps.renderer-discriminator.config-file=$CONFIG_FILE eps.renderer-discriminator.asset-root=$ASSET_ROOT eps.renderer-discriminator.membership-root=$MEMBERSHIP_ROOT eps.renderer-discriminator.protection-root=$PROTECTION_ROOT eps.renderer-discriminator.scene-root=$SCENE_ROOT
