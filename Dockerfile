# qaari-eval research workspace: Python 3.11 engine + Julia 1.11.5 (SciML lab) + Octave 6 (DSP cross-check).
# No credentials are baked in — mount or pass them at run time (see DOCKER-RESUME.md).
#
#   docker build -t qaari-workspace .
#   docker run --rm -it -v "$PWD":/work --env-file /path/to/.env qaari-workspace
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential pkg-config \
        ffmpeg libsndfile1 \
        octave octave-signal octave-control \
    && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Julia 1.11.5 — the Manifest is resolved with it (stdlib compat pins 1.11); 1.10 will not instantiate.
ARG JULIA_VERSION=1.11.5
RUN curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
      | tar -xz -C /opt && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia
ENV JULIA_DEPOT_PATH=/opt/julia-depot

WORKDIR /work
# Python env (engine + CLIs used by the compute bridge)
COPY requirements.txt requirements-ml.txt pyproject.toml ./
RUN python3.11 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -U pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt modal kaggle
ARG WITH_ML=0
RUN if [ "$WITH_ML" = "1" ]; then /opt/venv/bin/pip install --no-cache-dir -r requirements-ml.txt; fi
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv

# Julia research environment, precompiled into the image
COPY research_agency_lab/substrate_library/julia/Project.toml research_agency_lab/substrate_library/julia/Manifest.toml \
     /opt/qaarilab/
RUN julia --project=/opt/qaarilab -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Octave: Debian's octave-signal omits lpc(); the repo ships a shim (octave/lpc.m) used via --path.
CMD ["bash"]
