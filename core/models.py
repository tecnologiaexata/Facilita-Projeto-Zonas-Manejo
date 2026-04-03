from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


Mode = Literal["auto", "threshold", "hotspot"]
OutputFormat = Literal["gpkg", "shp"]


class AOIRef(BaseModel):
    source: str = Field("file", description="file_or_url_or_staging")
    id: Optional[str] = Field(None, description="AOI id or local file path")
    url: Optional[str] = Field(None, description="Public URL for AOI file")
    layer: Optional[str] = Field(None, description="Layer name if applicable")

    @model_validator(mode="after")
    def validate_reference(self):
        if not (self.id or self.url):
            raise ValueError("AOI must provide either 'id' or 'url'.")
        return self


class RasterRef(BaseModel):
    attribute: str
    source: str = Field("file", description="file_or_url_or_temp_or_staging")
    path: Optional[str] = Field(None, description="Local raster path")
    url: Optional[str] = Field(None, description="Public URL for raster file")
    raster_id: Optional[int] = Field(None, description="Optional source raster id")

    @model_validator(mode="after")
    def validate_reference(self):
        if not (self.path or self.url):
            raise ValueError("Raster must provide either 'path' or 'url'.")
        return self


class UserChoices(BaseModel):
    k: Optional[int] = Field(None, ge=2, description="Only used in auto mode")
    min_zone_area_ha: float = Field(..., gt=0)


class ThresholdClass(BaseModel):
    id: int
    label: str
    min: Optional[float] = None
    max: Optional[float] = None


class ModeParamsThreshold(BaseModel):
    attribute: str
    classes: List[ThresholdClass]
    units: Optional[str] = "same_as_raster"


class ClassificationLibrary(BaseModel):
    source: str = Field(..., description="embedded_example_or_external")
    version: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class HotspotLogic(BaseModel):
    method: Literal["score_sum"] = "score_sum"
    weights: Dict[str, float] = Field(default_factory=dict)
    zone_thresholds: List[Dict[str, Any]] = Field(default_factory=list)


class ModeParamsHotspot(BaseModel):
    hotspot_mode: Literal["library", "target"]
    selected_attributes: List[str]
    classification_library: Optional[ClassificationLibrary] = None
    hotspot_logic: Optional[HotspotLogic] = None
    target_rules: Optional[List[Dict[str, Any]]] = None


class OutputOptions(BaseModel):
    format: OutputFormat = "gpkg"
    include_stats: bool = True
    include_quicklooks: bool = True


class JobOptions(BaseModel):
    area_name: str = Field(..., min_length=1, description="Area name used to organize output folders.")


class ZonesRequest(BaseModel):
    processo: str = Field("Zoneamento", description="Process label for storage notification")
    atributo: str = Field("Zoneamento", description="Raster attribute label for storage notification")
    tipo: Optional[str] = Field(None, description="Area type used by add_raster_interpolados")
    id: Optional[int] = Field(None, description="Area identifier used by add_raster_interpolados")
    tipo_importacao: Optional[int] = Field(None, description="Import type used by add_raster_interpolados")
    cliente_id: Optional[int] = None
    fazenda: Optional[int] = None
    fazenda_id: Optional[int] = None
    talhao: Optional[int] = None
    talhao_id: Optional[int] = None
    talhao_nome: Optional[str] = None
    gleba: Optional[int] = None
    gleba_id: Optional[int] = None
    id_amostragem: Optional[int] = None
    safra: Optional[str] = None
    safra_id: Optional[int] = None
    profundidade: Optional[int | str] = None
    profundidade_id: Optional[int] = None
    data: Optional[str] = None
    date: Optional[str] = None
    descricao: Optional[str] = None
    usuario_id_cadastro: Optional[List[int]] = None
    raster_id: Optional[int] = None

    aoi: AOIRef
    rasters: List[RasterRef]
    mode: Mode
    user_choices: UserChoices
    mode_params: Optional[Dict[str, Any]] = None
    output: OutputOptions = Field(default_factory=OutputOptions)
    job: JobOptions
    metadata: Optional[Dict[str, Any]] = None

    dry_run: bool = Field(
        default=False,
        description="If true, returns preview only (threshold mode can omit classes).",
    )

    @model_validator(mode="after")
    def validate_hotspot_modes(self):
        if self.mode == "hotspot":
            if not self.mode_params:
                raise ValueError("mode_params é obrigatório no modo hotspot.")

            hotspot_mode = self.mode_params.get("hotspot_mode")

            if hotspot_mode not in ["library", "target"]:
                raise ValueError("hotspot_mode deve ser 'library' ou 'target'.")

            selected_attributes = self.mode_params.get("selected_attributes")
            if not selected_attributes or not isinstance(selected_attributes, list):
                raise ValueError(
                    "selected_attributes é obrigatório no mode_params do modo hotspot."
                )

            if hotspot_mode == "library":
                lib = self.mode_params.get("classification_library")

                if not lib:
                    raise ValueError(
                        "classification_library é obrigatório quando hotspot_mode='library'."
                    )

                if "attributes" not in lib:
                    raise ValueError(
                        "classification_library deve conter a chave 'attributes'."
                    )

                if not isinstance(lib.get("attributes"), dict) or not lib.get("attributes"):
                    raise ValueError(
                        "classification_library.attributes deve ser um dicionário não vazio."
                    )

            if hotspot_mode == "target":
                target_rules = self.mode_params.get("target_rules")
                if target_rules is None:
                    raise ValueError(
                        "target_rules é obrigatório quando hotspot_mode='target'."
                    )

                if not isinstance(target_rules, list) or not target_rules:
                    raise ValueError(
                        "target_rules deve ser uma lista não vazia quando hotspot_mode='target'."
                    )

                negative_split_quantile = self.mode_params.get("negative_split_quantile")
                if negative_split_quantile is not None:
                    try:
                        value = float(negative_split_quantile)
                    except Exception as e:
                        raise ValueError(
                            "negative_split_quantile deve ser numérico."
                        ) from e

                    if value < 0 or value > 1:
                        raise ValueError(
                            "negative_split_quantile deve estar entre 0 e 1."
                        )

        return self
