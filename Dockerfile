FROM mambaorg/micromamba:1.5.8

WORKDIR /app

# evitar conflito OpenMP em Linux
ENV MKL_THREADING_LAYER=GNU

COPY . /app

# conda-forge para stack geoespacial + pip só para topojson
RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    pip \
    fastapi \
    uvicorn \
    pydantic \
    python-dotenv \
    numpy \
    pandas \
    scikit-learn \
    geopandas \
    rasterio \
    shapely \
    pyproj \
    fiona \
    && micromamba run -n base python -m pip install --no-cache-dir topojson \
    && micromamba run -n base python -c "from topojson import Topology; import fastapi, geopandas, rasterio, sklearn; print('build dependencies ok')" \
    && micromamba clean --all --yes

EXPOSE 8040

CMD ["micromamba", "run", "-n", "base", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8040"]
