# Módulo de Geração de Zonas de Manejo

## Objetivo

Este módulo gera **Zonas de Manejo (UGDs)** a partir de rasters previamente interpolados.

O módulo **não realiza interpolação**.  
Os rasters devem ser gerados previamente pelo módulo de interpolação da plataforma.

Os rasters fornecidos devem estar:

- na mesma projeção
- na mesma resolução espacial
- com a mesma extensão espacial
- alinhados pixel a pixel

Essas condições são normalmente garantidas pelo pipeline de interpolação da plataforma.

Caso não estejam, o módulo atual possui a funcionalidade de verificação e ajuste.

---

# Modos de Operação

O módulo suporta três modos principais de geração de zonas:

| Modo | Descrição |
|-----|-----|
| `auto` | agrupamento automático de pixels baseado em clustering |
| `threshold` | divisão de zonas baseada em limites definidos pelo usuário |
| `hotspot` | identificação de zonas críticas com base em regras agronômicas |

---

# Estrutura Geral do Pipeline

O pipeline segue as etapas abaixo:

1️⃣ **Validação de entrada**

- valida estrutura do JSON
- valida caminhos de arquivos
- verifica consistência do modo de operação

2️⃣ **Carregamento**

- carregar AOI
- carregar rasters de atributos
- alinhar rasters se necessário

3️⃣ **Classificação**

Dependente do modo escolhido:

- clustering (`auto`)
- classificação por limite (`threshold`)
- classificação por biblioteca de recomendação (`hotspot`)

O resultado desta etapa é um **raster de zonas**.

4️⃣ **Pós-processamento raster**

- identificação de componentes conectadas
- remoção de regiões menores que a área mínima

5️⃣ **Vetorização**

- poligonização do raster de zonas
- dissolução por zona
- remoção de geometrias inválidas

6️⃣ **Suavização geométrica**

- simplificação topológica
- suavização das fronteiras
- preservação da contiguidade espacial

7️⃣ **Estatísticas por zona**

Para cada zona são calculadas:

- média dos atributos
- mediana
- área

8️⃣ **Exportação**

- camada vetorial final
- camadas intermediárias opcionais
- relatórios de execução

---

# Estrutura do projeto

config_examples/                            # contrato de cada modo
    config_auto.json
    config_hotspot_library.json
    config_hotspot_target.json
    config_threshold_dry_run.json           # quando dry_run = true, apenas exibe estatística do atributo selecionado
    config_threshold.json                   # quando dry_run = false, gera as zonas de fato
core/                                       # arquivos de processamento
    models.py
    io.py
    alignment.py
    classification_auto.py
    classification_threshold.py
    classification_hotspot.py
    threshold_preview.py
    raster_postprocess.py
    polygonize.py
    smoothing.py
    vector_postprocess.py
    statistics.py
    pipeline.py

api/
    __init__.py
    main.py

---

# API

O container sobe a API FastAPI na porta `8040`.

## Rodando com Docker Compose

Na raiz do projeto:

```bash
docker compose up --build --force-recreate
docker compose down
```

Arquivos de ambiente disponíveis:

- `.env`: arquivo pronto para preencher no ambiente local
- `.env.example`: modelo de referência

Variáveis mais importantes:

- `FACILITAGRO_FRONTEND_BASE_URL`: base do Facilita Agro usada para resolver blobs e notificar o `add_raster_interpolados`
- `BLOB_READ_WRITE_TOKEN` ou `NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN`: token do Blob Storage
- `BLOB_UPLOAD_BASE_URL`: override opcional da URL de upload do blob; padrão `https://blob.vercel-storage.com`

Exemplo de subida:

```bash
docker compose up --build --force-recreate
docker compose logs -f zoneamento-api
```

Endpoints principais:

- `GET /health`
- `POST /zones/generate`

## Entradas aceitas

O endpoint `POST /zones/generate` aceita os insumos de duas formas:

- caminho local no container/servidor
- URL pública, incluindo URLs do Blob Storage usadas nos projetos Facilita

Campos aceitos:

- `aoi.id`: caminho local do AOI
- `aoi.url`: URL pública do AOI
- `rasters[].path`: caminho local do raster
- `rasters[].url`: URL pública do raster

Se uma URL for enviada, o serviço baixa o arquivo para `outputs/temp/<area_name>/downloads/`
antes de processar.

## Compatibilidade com Blob Storage

Quando a URL pertence ao Blob Storage da Vercel, o serviço segue a mesma estratégia dos
outros projetos Facilita:

- tenta resolver a `downloadUrl` via `FACILITAGRO_FRONTEND_BASE_URL/api/v1/blobs/listBlob`
- adiciona header `Authorization: Bearer <token>` quando `BLOB_READ_WRITE_TOKEN` ou
  `NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN` estiver configurado

Essas variáveis são opcionais para blobs realmente públicos, mas recomendadas para manter
compatibilidade com o restante da plataforma.

## Observações

- Para AOI remoto, o formato mais seguro é `KML`, que foi o cenário solicitado.
- Também funciona com arquivos vetoriais de um único arquivo, como `GPKG` e `GeoJSON`.
- Para `Shapefile`, o ideal é enviar um `.zip` com todos os sidecars.
