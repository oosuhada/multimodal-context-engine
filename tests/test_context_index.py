from pathlib import Path

import numpy as np

from context_engine.index import ContextIndex, MediaItem


def test_balanced_fusion_and_group_search(tmp_path: Path) -> None:
    # group-a에는 image가 두 개, audio가 하나이고 group-b에는 video 하나가 있는 toy manifest를 만듭니다.
    items = [
        MediaItem(id="a-image-1", group_id="group-a", modality="image", path="a1.jpg"),
        MediaItem(id="a-image-2", group_id="group-a", modality="image", path="a2.jpg"),
        MediaItem(id="a-audio", group_id="group-a", modality="audio", path="a.wav"),
        MediaItem(id="b-video", group_id="group-b", modality="video", path="b.mp4"),
    ]
    # group-a는 첫 번째 축, group-b는 두 번째 축을 가리키도록 embedding을 구성합니다.
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # 같은 modality 파일 수가 많아도 균형을 유지하는 fusion으로 index를 생성합니다.
    index = ContextIndex(items=items, embeddings=embeddings, fusion="balanced")
    # persistence까지 함께 검증하기 위해 임시 디렉터리에 저장합니다.
    output = tmp_path / "context-index"
    # index 파일을 disk에 기록합니다.
    index.save(output)
    # 저장된 index를 다시 복원합니다.
    restored = ContextIndex.load(output)
    # 첫 번째 축과 같은 query를 검색합니다.
    results = restored.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)
    # 가장 가까운 context는 group-a여야 합니다.
    assert results[0].group_id == "group-a"
    # group-a의 세 media item이 evidence로 모두 남아 있어야 합니다.
    assert len(results[0].evidence) == 3

