"""멀티모달 item과 group-level context embedding을 보관하고 검색합니다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .fusion import get_fusion


@dataclass(frozen=True)
class MediaItem:
    """한 개의 image/audio/video 입력과 그 context group을 표현합니다."""

    id: str
    group_id: str
    modality: str
    path: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceResult:
    """group 검색 결과를 뒷받침하는 개별 media item 점수입니다."""

    item_id: str
    modality: str
    path: str
    score: float


@dataclass(frozen=True)
class ContextResult:
    """하나의 fused context group 검색 결과입니다."""

    group_id: str
    score: float
    evidence: list[EvidenceResult]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    # cosine similarity를 dot product로 계산하기 위해 모든 row를 L2 정규화합니다.
    matrix = np.asarray(matrix, dtype=np.float32)
    # 0 norm에서 division error가 발생하지 않도록 최소값을 적용합니다.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 정규화된 행렬을 반환합니다.
    return matrix / np.clip(norms, 1e-12, None)


class ContextIndex:
    """item embedding과 group fusion 결과를 함께 관리하는 persistent context index입니다."""

    def __init__(
        self,
        items: list[MediaItem],
        embeddings: np.ndarray,
        fusion: str = "balanced",
    ) -> None:
        # manifest의 원래 item 순서를 그대로 보존합니다.
        self.items = items
        # 모든 item embedding을 동일 cosine 공간으로 정규화해 보관합니다.
        self.embeddings = _normalize_rows(embeddings)
        # group embedding을 만들 때 사용할 fusion 전략 이름을 저장합니다.
        self.fusion = fusion

        # manifest item 수와 embedding 수가 다르면 index가 잘못 연결된 것이므로 실패시킵니다.
        if len(self.items) != self.embeddings.shape[0]:
            raise ValueError("number of items must match number of embeddings")
        # embedding은 반드시 [N, D] 행렬이어야 합니다.
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")

        # group search를 빠르게 수행할 수 있도록 생성 시 fused group matrix를 계산합니다.
        self.group_ids, self.group_embeddings = self._build_group_embeddings()

    def _build_group_embeddings(self) -> tuple[list[str], np.ndarray]:
        # fusion 구현은 직접 분기하지 않고 LAVIS-derived registry에서 가져옵니다.
        fusion_function = get_fusion(self.fusion)
        # group 순서를 최초 등장 순서로 고정해 저장/로드 시 결과를 재현 가능하게 합니다.
        group_ids = list(dict.fromkeys(item.group_id for item in self.items))
        # 각 group의 fused embedding을 담을 리스트를 준비합니다.
        group_vectors: list[np.ndarray] = []

        # group별 item embedding과 modality를 모아 하나의 context vector로 결합합니다.
        for group_id in group_ids:
            # 현재 group에 해당하는 item index를 선택합니다.
            indices = [index for index, item in enumerate(self.items) if item.group_id == group_id]
            # 선택된 item의 modality 이름을 동일 순서로 가져옵니다.
            modalities = [self.items[index].modality for index in indices]
            # registry에서 가져온 fusion 함수를 호출해 group 대표 embedding을 계산합니다.
            group_vectors.append(fusion_function(self.embeddings[indices], modalities))

        # 모든 group vector를 하나의 검색 행렬로 합칩니다.
        return group_ids, _normalize_rows(np.stack(group_vectors, axis=0))

    def save(self, directory: Path) -> None:
        """context index를 NumPy 행렬과 JSON metadata로 저장합니다."""

        # 지정한 저장 폴더가 없으면 생성합니다.
        directory.mkdir(parents=True, exist_ok=True)
        # item embedding은 빠른 로딩을 위해 NumPy binary로 저장합니다.
        np.save(directory / "item_embeddings.npy", self.embeddings)
        # manifest와 fusion 설정을 JSON metadata로 변환합니다.
        metadata = {
            "format": "multimodal-context-index-v1",
            "fusion": self.fusion,
            "items": [
                {
                    "id": item.id,
                    "group_id": item.group_id,
                    "modality": item.modality,
                    "path": item.path,
                    "metadata": item.metadata,
                }
                for item in self.items
            ],
        }
        # 한글 metadata를 그대로 읽을 수 있도록 UTF-8 JSON으로 저장합니다.
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "ContextIndex":
        """저장된 context index를 복원합니다."""

        # JSON metadata를 읽어 manifest와 fusion 정보를 복원합니다.
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        # 다른 버전 파일을 잘못 읽지 않도록 format identifier를 검사합니다.
        if metadata.get("format") != "multimodal-context-index-v1":
            raise ValueError("unsupported context index format")
        # object pickle을 허용하지 않는 방식으로 item embedding 행렬을 읽습니다.
        embeddings = np.load(directory / "item_embeddings.npy", allow_pickle=False)
        # JSON item을 MediaItem dataclass로 변환합니다.
        items = [MediaItem(**item) for item in metadata["items"]]
        # 저장 당시 fusion 전략까지 복원해 동일 group vector를 재생성합니다.
        return cls(items=items, embeddings=embeddings, fusion=str(metadata["fusion"]))

    def search(self, query_embedding: np.ndarray | torch.Tensor, top_k: int = 5) -> list[ContextResult]:
        """텍스트 query embedding과 가까운 fused context group을 반환합니다."""

        # PyTorch Tensor query를 NumPy로 변환합니다.
        if isinstance(query_embedding, torch.Tensor):
            query = query_embedding.detach().cpu().numpy()
        else:
            query = np.asarray(query_embedding, dtype=np.float32)
        # [1, D] 또는 [D] 입력을 모두 하나의 벡터로 정리합니다.
        query = query.reshape(-1).astype(np.float32)
        # cosine similarity 계산을 위해 query도 L2 정규화합니다.
        query = query / max(float(np.linalg.norm(query)), 1e-12)

        # group 수준 similarity를 한 번에 계산합니다.
        group_scores = self.group_embeddings @ query
        # 반환 결과 개수를 실제 group 수 범위로 제한합니다.
        limit = min(max(top_k, 1), len(self.group_ids))
        # 큰 index에서 전체 sort 비용을 피하기 위해 우선 top 후보만 선택합니다.
        candidates = np.argpartition(-group_scores, limit - 1)[:limit]
        # 선택한 후보를 실제 점수 기준으로 정렬합니다.
        ordered_groups = candidates[np.argsort(-group_scores[candidates])]
        # 개별 evidence 점수도 같은 query에 대해 미리 계산합니다.
        item_scores = self.embeddings @ query
        # 최종 ContextResult 목록을 준비합니다.
        results: list[ContextResult] = []

        # group 결과마다 어떤 media item이 점수를 뒷받침했는지 함께 계산합니다.
        for group_index in ordered_groups:
            # 현재 결과 group ID를 가져옵니다.
            group_id = self.group_ids[group_index]
            # 해당 group에 속한 item index만 찾습니다.
            indices = [index for index, item in enumerate(self.items) if item.group_id == group_id]
            # evidence는 query와 더 가까운 item부터 보여주기 위해 점수순으로 정렬합니다.
            indices = sorted(indices, key=lambda index: float(item_scores[index]), reverse=True)
            # 각 item을 사람이 확인할 수 있는 evidence record로 변환합니다.
            evidence = [
                EvidenceResult(
                    item_id=self.items[index].id,
                    modality=self.items[index].modality,
                    path=self.items[index].path,
                    score=float(item_scores[index]),
                )
                for index in indices
            ]
            # group score와 evidence를 하나의 결과 객체로 묶습니다.
            results.append(
                ContextResult(
                    group_id=group_id,
                    score=float(group_scores[group_index]),
                    evidence=evidence,
                )
            )

        # 점수순으로 정렬된 context 결과를 반환합니다.
        return results

