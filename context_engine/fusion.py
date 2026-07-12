"""LAVIS registry를 직접 재사용해 context embedding fusion 전략을 관리합니다."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from third_party.lavis.lavis.common.registry import registry


FusionFunction = Callable[[np.ndarray, list[str]], np.ndarray]


def _normalize(vector: np.ndarray) -> np.ndarray:
    # 서로 다른 modality의 scale을 제거하기 위해 최종 group embedding을 L2 정규화합니다.
    vector = np.asarray(vector, dtype=np.float32)
    # 0 벡터가 들어와 NaN이 생기지 않도록 최소 norm을 보장합니다.
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def mean_fusion(embeddings: np.ndarray, modalities: list[str]) -> np.ndarray:
    """같은 group에 속한 모든 modality를 동일 가중치로 평균합니다."""

    # item embedding의 단순 평균으로 group의 공통 semantic direction을 구합니다.
    fused = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0)
    # 검색 similarity가 안정적이도록 평균 결과를 다시 정규화합니다.
    return _normalize(fused)


def balanced_modality_fusion(embeddings: np.ndarray, modalities: list[str]) -> np.ndarray:
    """같은 modality 파일이 여러 개 있어도 modality 하나가 과도한 비중을 차지하지 않게 결합합니다."""

    # 입력을 NumPy 행렬로 명시적으로 변환합니다.
    matrix = np.asarray(embeddings, dtype=np.float32)
    # group에 포함된 modality 종류를 최초 등장 순서대로 추출합니다.
    unique_modalities = list(dict.fromkeys(modalities))
    # modality별 평균 embedding을 담을 리스트를 준비합니다.
    modality_vectors: list[np.ndarray] = []

    # image가 10개이고 audio가 1개여도 두 modality가 동등한 비중을 갖도록 먼저 modality 내부 평균을 냅니다.
    for modality in unique_modalities:
        # 현재 modality에 해당하는 item row index만 선택합니다.
        indices = [index for index, value in enumerate(modalities) if value == modality]
        # 선택한 item들의 평균 embedding을 modality 대표 벡터로 사용합니다.
        modality_vectors.append(np.mean(matrix[indices], axis=0))

    # modality 대표 벡터들을 다시 평균해 최종 context vector를 만듭니다.
    fused = np.mean(np.stack(modality_vectors, axis=0), axis=0)
    # 최종 cosine 검색을 위해 L2 정규화합니다.
    return _normalize(fused)


# LAVIS의 generic state registry를 그대로 사용해 기본 fusion 전략을 등록합니다.
registry.register("context.fusion.mean", mean_fusion)
# modality 수 균형을 보정하는 사용자 전략도 같은 registry에 등록합니다.
registry.register("context.fusion.balanced", balanced_modality_fusion)


def get_fusion(name: str) -> FusionFunction:
    """등록된 fusion 이름으로 실제 함수를 가져옵니다."""

    # 사용자가 짧은 이름만 넘겨도 registry의 전체 key 경로로 변환합니다.
    fusion = registry.get(f"context.fusion.{name}", default=None, no_warning=True)
    # 존재하지 않는 전략을 조용히 무시하지 않고 명확한 오류를 반환합니다.
    if fusion is None:
        raise ValueError(f"unknown fusion strategy: {name}")
    # 타입 힌트 기준으로 callable fusion 함수를 반환합니다.
    return fusion

