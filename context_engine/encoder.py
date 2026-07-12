"""LanguageBind 원본 model/processor/tokenizer를 직접 사용하는 runtime encoder입니다."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from languagebind import (
    LanguageBind,
    LanguageBindImageTokenizer,
    to_device,
    transform_dict,
)

from .index import MediaItem


DEFAULT_CHECKPOINTS = {
    "image": "LanguageBind_Image",
    "audio": "LanguageBind_Audio_FT",
    "video": "LanguageBind_Video_FT",
    "depth": "LanguageBind_Depth",
    "thermal": "LanguageBind_Thermal",
}


class LanguageBindContextEncoder:
    """필요한 modality encoder만 로드해 manifest item과 text query를 임베딩합니다."""

    def __init__(
        self,
        modalities: list[str],
        device: str | None = None,
        cache_dir: str = "./cache_dir",
    ) -> None:
        # 현재 LanguageBind dependency stack의 호환성을 우선해 CUDA가 없으면 CPU를 사용합니다.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # 이후 모든 processor/model 입력에서 동일 장치를 사용하도록 저장합니다.
        self.device = torch.device(device)
        # 중복 modality를 제거하면서 최초 순서를 보존합니다.
        self.modalities = list(dict.fromkeys(modalities))
        # 지원하지 않는 modality가 manifest에 들어오면 모델 다운로드 전에 즉시 알려줍니다.
        unsupported = [modality for modality in self.modalities if modality not in DEFAULT_CHECKPOINTS]
        if unsupported:
            raise ValueError(f"unsupported modalities: {unsupported}")

        # LanguageBind 원본 클래스가 기대하는 modality→checkpoint mapping을 생성합니다.
        clip_type = {modality: DEFAULT_CHECKPOINTS[modality] for modality in self.modalities}
        # multimodal encoder와 projection은 LanguageBind 원본 구현을 그대로 로드합니다.
        self.model = LanguageBind(clip_type=clip_type, cache_dir=cache_dir)
        # inference 전용이므로 dropout 등을 끄고 선택한 device로 이동합니다.
        self.model = self.model.eval().to(self.device)
        # 각 modality processor 역시 원본 config를 사용해 생성합니다.
        self.processors = {
            modality: transform_dict[modality](self.model.modality_config[modality])
            for modality in self.modalities
        }
        # shared language encoder에 맞는 원본 LanguageBind image tokenizer를 그대로 사용합니다.
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
            "LanguageBind/LanguageBind_Image",
            cache_dir=str(Path(cache_dir) / "tokenizer_cache_dir"),
        )

    @staticmethod
    def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
        # LanguageBind의 temperature scaling 여부와 상관없이 index에는 unit vector만 저장합니다.
        return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_items(self, items: list[MediaItem]) -> np.ndarray:
        """manifest 순서를 유지한 채 여러 modality media item을 하나의 embedding 행렬로 변환합니다."""

        # modality별 batch 처리를 위해 원래 item index를 묶습니다.
        indices_by_modality: dict[str, list[int]] = defaultdict(list)
        # 각 item의 modality에 해당 index를 추가합니다.
        for index, item in enumerate(items):
            indices_by_modality[item.modality].append(index)
        # 최종 결과를 원래 manifest 순서대로 배치하기 위한 placeholder를 준비합니다.
        encoded_by_index: dict[int, np.ndarray] = {}

        # modality별 processor는 입력 형식이 다르므로 각각 batch 처리합니다.
        for modality, indices in indices_by_modality.items():
            # 현재 modality에 속한 파일 경로 목록을 순서대로 가져옵니다.
            paths = [items[index].path for index in indices]
            # LanguageBind 공식 processor를 사용해 파일들을 모델 입력 dict로 변환합니다.
            processed = self.processors[modality](paths)
            # processor 결과의 모든 Tensor를 모델과 같은 device로 옮깁니다.
            inputs = {modality: to_device(processed, self.device)}
            # embedding 추론에는 gradient가 필요 없으므로 inference mode를 사용합니다.
            with torch.inference_mode():
                # LanguageBind 원본 forward를 호출해 shared-space embedding을 얻습니다.
                embeddings = self.model(inputs)[modality]
                # group fusion 전 scale 편향을 없애기 위해 L2 정규화합니다.
                embeddings = self._normalize(embeddings.float()).cpu().numpy()
            # batch 결과를 원래 manifest item index에 다시 연결합니다.
            for local_index, original_index in enumerate(indices):
                encoded_by_index[original_index] = embeddings[local_index]

        # 원래 item 순서대로 embedding을 stack해 persistence layer에 전달합니다.
        return np.stack([encoded_by_index[index] for index in range(len(items))], axis=0)

    def encode_text(self, text: str) -> np.ndarray:
        """자연어 query를 media와 동일한 shared semantic space로 변환합니다."""

        # LanguageBind 원본 예제와 동일하게 최대 길이 77 token으로 text를 준비합니다.
        tokenized = self.tokenizer(
            [text],
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # tokenizer 결과를 모델과 같은 device로 옮깁니다.
        inputs = {"language": to_device(tokenized, self.device)}
        # query inference는 gradient가 필요하지 않습니다.
        with torch.inference_mode():
            # shared language encoder를 LanguageBind 원본 forward로 실행합니다.
            embedding = self.model(inputs)["language"]
            # item embedding과 동일한 방식으로 L2 정규화합니다.
            embedding = self._normalize(embedding.float())
        # 검색 계층에서 사용할 NumPy 1차원 벡터로 변환합니다.
        return embedding[0].cpu().numpy()

