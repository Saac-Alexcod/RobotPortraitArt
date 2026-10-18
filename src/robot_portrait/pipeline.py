from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import platform
import random
import re
from time import perf_counter_ns
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from .generators import (
    BaselineGenerator,
    GeneratorRequest,
    GeneratorUnavailableError,
    PortraitGenerator,
)
from .io import write_image
from .preprocessing import (
    PreprocessResult,
    preprocess_portrait,
)
from .vectorize import (
    raster_to_svg,
    svg_to_raster,
)


class PipelineStage(str, Enum):
    PREPROCESS = "preprocess"
    GENERATE = "generate"
    VECTORIZE = "vectorize"
    RASTERIZE = "rasterize"
    TOTAL = "total"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration used by GenerationPipeline."""

    preprocess_size: int = 512
    seed: int = 42

    baseline: dict[str, Any] = field(
        default_factory=dict
    )

    vector: dict[str, Any] = field(
        default_factory=dict
    )

    source_path: Path | None = None

    def __post_init__(self) -> None:
        if int(self.preprocess_size) <= 0:
            raise ValueError(
                "preprocess_size 必须是正整数"
            )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        preprocess_size: int = 512,
        seed: int = 42,
    ) -> "PipelineConfig":
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"找不到配置文件：{config_path}"
            )

        if not config_path.is_file():
            raise ValueError(
                f"配置路径不是文件：{config_path}"
            )

        loaded = (
            yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8"
                )
            )
            or {}
        )

        if not isinstance(
            loaded,
            Mapping,
        ):
            raise ValueError(
                "配置文件根节点必须是映射对象"
            )

        baseline = loaded.get(
            "baseline"
        )

        vector = loaded.get(
            "vector"
        )

        if (
            not isinstance(
                baseline,
                Mapping,
            )
            or not isinstance(
                vector,
                Mapping,
            )
        ):
            raise ValueError(
                "配置文件必须包含 "
                "baseline 和 vector 两个映射节"
            )

        return cls(
            preprocess_size=int(
                preprocess_size
            ),
            seed=int(seed),
            baseline=dict(
                baseline
            ),
            vector=dict(
                vector
            ),
            source_path=config_path,
        )

    def to_metadata(
        self,
    ) -> dict[str, Any]:
        return {
            "preprocess_size":
                self.preprocess_size,

            "seed":
                self.seed,

            "source_path": (
                str(self.source_path)
                if self.source_path
                else None
            ),

            "baseline":
                dict(self.baseline),

            "vector":
                dict(self.vector),
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Paths and metadata produced by one completed run."""

    input_path: Path
    output_dir: Path

    method: str

    prepared_path: Path
    raster_path: Path
    svg_path: Path
    vector_render_path: Path
    manifest_path: Path

    preprocessing: dict[str, Any]
    generation: dict[str, Any]
    vectorization: dict[str, Any]
    rasterization: dict[str, Any]

    timings_ms: dict[str, float]

    warnings: tuple[str, ...] = ()

    def to_manifest(
        self,
        *,
        config: PipelineConfig,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "completed",

            "method":
                self.method,

            "input":
                str(self.input_path),

            "output_dir":
                str(self.output_dir),

            "files": {
                "prepared":
                    self.prepared_path.name,

                "raster_portrait":
                    self.raster_path.name,

                "vector_svg":
                    self.svg_path.name,

                "vector_render":
                    self.vector_render_path.name,

                "manifest":
                    self.manifest_path.name,
            },

            "preprocessing":
                self.preprocessing,

            "generation":
                self.generation,

            "vectorization":
                self.vectorization,

            "rasterization":
                self.rasterization,

            "timings_ms":
                dict(self.timings_ms),

            "warnings":
                list(self.warnings),

            "config":
                config.to_metadata(),

            "environment": {
                "python":
                    platform.python_version(),

                "opencv":
                    cv2.__version__,

                "platform":
                    platform.platform(),
            },
        }


class GenerationPipeline:
    """End-to-end pipeline with pluggable portrait generators."""

    _METHOD_PATTERN = re.compile(
        r"^[a-z0-9][a-z0-9_-]*$"
    )

    def __init__(
        self,
        config: PipelineConfig,
        *,
        generators: Mapping[
            str,
            PortraitGenerator,
        ] | None = None,
    ):
        self.config = config

        if generators is None:
            generators = {
                BaselineGenerator.name:
                    BaselineGenerator(
                        config.baseline
                    ),
            }

        registry: dict[
            str,
            PortraitGenerator,
        ] = {}

        for name, generator in generators.items():
            if (
                not isinstance(
                    name,
                    str,
                )
                or not self
                ._METHOD_PATTERN
                .fullmatch(name)
            ):
                raise ValueError(
                    "generator 名称必须匹配 "
                    "^[a-z0-9][a-z0-9_-]*$"
                )

            if not isinstance(
                generator,
                PortraitGenerator,
            ):
                raise TypeError(
                    f"generator '{name}' "
                    "必须实现 PortraitGenerator"
                )

            if generator.name != name:
                raise ValueError(
                    f"generator 注册名 '{name}' "
                    f"与实现名称 "
                    f"'{generator.name}' 不一致"
                )

            registry[name] = generator

        if not registry:
            raise ValueError(
                "至少需要注册一个 "
                "portrait generator"
            )

        self._generators = registry

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        preprocess_size: int = 512,
        seed: int = 42,
        generators: Mapping[
            str,
            PortraitGenerator,
        ] | None = None,
    ) -> "GenerationPipeline":
        return cls(
            PipelineConfig.from_yaml(
                path,
                preprocess_size=(
                    preprocess_size
                ),
                seed=seed,
            ),
            generators=generators,
        )

    @property
    def supported_methods(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._generators
            )
        )

    def generator_statuses(
        self,
    ) -> dict[
        str,
        dict[str, Any],
    ]:
        """JSON-ready generator preflight information."""

        return {
            name:
                generator
                .status()
                .to_metadata()

            for name, generator
            in sorted(
                self._generators.items()
            )
        }

    @staticmethod
    def _elapsed_ms(
        start_ns: int,
    ) -> float:
        return round(
            (
                perf_counter_ns()
                - start_ns
            )
            / 1_000_000,
            3,
        )

    @staticmethod
    def _write_json_atomic(
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_name(
            path.name + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.replace(
            path
        )

    def _seed(
        self,
    ) -> None:
        random.seed(
            self.config.seed
        )

        np.random.seed(
            self.config.seed
        )

    def _generator_for(
        self,
        method: str,
    ) -> PortraitGenerator:
        try:
            return self._generators[
                method
            ]

        except KeyError as exc:
            supported = ", ".join(
                self.supported_methods
            )

            raise ValueError(
                f"不支持的生成方法：{method}。"
                f"当前支持：{supported}"
            ) from exc

    def _managed_paths(
        self,
        output_dir: Path,
    ) -> tuple[Path, ...]:
        paths: list[Path] = [
            output_dir
            / "prepared.png",

            output_dir
            / "manifest.json",

            output_dir
            / "manifest.json.tmp",
        ]

        for name in self.supported_methods:
            paths.extend(
                [
                    output_dir
                    / f"{name}.png",

                    output_dir
                    / f"{name}.svg",

                    output_dir
                    / (
                        f"{name}"
                        "_vector_render.png"
                    ),
                ]
            )

        return tuple(paths)

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        method: str = "baseline",
    ) -> PipelineResult:
        generator = self._generator_for(
            method
        )

        input_path = Path(
            input_path
        )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prepared_path = (
            output_dir
            / "prepared.png"
        )

        raster_path = (
            output_dir
            / f"{method}.png"
        )

        svg_path = (
            output_dir
            / f"{method}.svg"
        )

        vector_render_path = (
            output_dir
            / (
                f"{method}"
                "_vector_render.png"
            )
        )

        manifest_path = (
            output_dir
            / "manifest.json"
        )

        # Reusing an output directory must never expose stale
        # managed artifacts from an earlier successful run.
        for stale_path in (
            self._managed_paths(
                output_dir
            )
        ):
            if stale_path.exists():
                stale_path.unlink()

        status = generator.status()

        if not status.available:
            reason = (
                status.reason
                or "未提供不可用原因"
            )

            raise GeneratorUnavailableError(
                f"生成器 '{method}' "
                f"当前不可用：{reason}"
            )

        self._seed()

        total_start = (
            perf_counter_ns()
        )

        timings: dict[
            str,
            float,
        ] = {}

        # --------------------------------------------------
        # Stage 1: preprocessing
        # --------------------------------------------------

        stage_start = (
            perf_counter_ns()
        )

        preprocess_result: (
            PreprocessResult
        ) = preprocess_portrait(
            input_path,
            size=(
                self.config
                .preprocess_size
            ),
        )

        write_image(
            prepared_path,
            preprocess_result.image,
        )

        timings[
            PipelineStage
            .PREPROCESS
            .value
        ] = self._elapsed_ms(
            stage_start
        )

        # --------------------------------------------------
        # Stage 2: portrait generation
        # --------------------------------------------------

        stage_start = (
            perf_counter_ns()
        )

        generation_output = (
            generator.generate(
                GeneratorRequest(
                    prepared_image=(
                        preprocess_result.image
                    ),

                    input_path=(
                        input_path
                    ),

                    prepared_path=(
                        prepared_path
                    ),

                    output_dir=(
                        output_dir
                    ),

                    seed=(
                        self.config.seed
                    ),
                )
            )
        )

        raster = (
            generation_output.image
        )

        write_image(
            raster_path,
            raster,
        )

        timings[
            PipelineStage
            .GENERATE
            .value
        ] = self._elapsed_ms(
            stage_start
        )

        # --------------------------------------------------
        # Stage 3: vectorization
        # --------------------------------------------------

        stage_start = (
            perf_counter_ns()
        )

        vectorization = (
            raster_to_svg(
                raster,
                svg_path,
                **self.config.vector,
            )
        )

        timings[
            PipelineStage
            .VECTORIZE
            .value
        ] = self._elapsed_ms(
            stage_start
        )

        # --------------------------------------------------
        # Stage 4: SVG rasterization
        # --------------------------------------------------

        stage_start = (
            perf_counter_ns()
        )

        rasterization = (
            svg_to_raster(
                svg_path,
                vector_render_path,
                output_width=(
                    self.config
                    .preprocess_size
                ),
            )
        )

        timings[
            PipelineStage
            .RASTERIZE
            .value
        ] = self._elapsed_ms(
            stage_start
        )

        timings[
            PipelineStage
            .TOTAL
            .value
        ] = self._elapsed_ms(
            total_start
        )

        preprocessing = (
            preprocess_result
            .to_metadata(
                input_path=input_path,
                output_path=prepared_path,
            )
        )

        generation = {
            "name":
                generator.name,

            "status":
                status.to_metadata(),

            "metadata":
                dict(
                    generation_output.metadata
                ),
        }

        warnings = (
            tuple(
                preprocess_result.warnings
            )
            +
            tuple(
                generation_output.warnings
            )
        )

        result = PipelineResult(
            input_path=input_path,
            output_dir=output_dir,

            method=method,

            prepared_path=(
                prepared_path
            ),

            raster_path=(
                raster_path
            ),

            svg_path=(
                svg_path
            ),

            vector_render_path=(
                vector_render_path
            ),

            manifest_path=(
                manifest_path
            ),

            preprocessing=(
                preprocessing
            ),

            generation=(
                generation
            ),

            vectorization=(
                vectorization
            ),

            rasterization=(
                rasterization
            ),

            timings_ms=(
                timings
            ),

            warnings=(
                warnings
            ),
        )

        self._write_json_atomic(
            manifest_path,
            result.to_manifest(
                config=self.config,
            ),
        )

        return result