"""Manifest 기반 multimodal context index/search 명령행 인터페이스입니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .encoder import LanguageBindContextEncoder
from .index import ContextIndex, MediaItem


def load_manifest(path: Path) -> list[MediaItem]:
    """JSON manifest를 검증 가능한 MediaItem 목록으로 변환합니다."""

    # UTF-8 JSON manifest를 읽습니다.
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 최상위 items 배열이 없으면 명확한 오류를 반환합니다.
    if not isinstance(payload.get("items"), list):
        raise ValueError("manifest must contain an 'items' array")
    # 각 JSON object를 MediaItem dataclass로 변환합니다.
    items = [MediaItem(**item) for item in payload["items"]]
    # 비어 있는 manifest는 모델을 로드할 필요가 없으므로 즉시 실패시킵니다.
    if not items:
        raise ValueError("manifest must contain at least one item")
    # 실제 파일이 없는 item을 미리 찾아 긴 모델 추론 후 실패하는 상황을 막습니다.
    missing = [item.path for item in items if not Path(item.path).exists()]
    # 누락 파일이 있으면 첫 몇 개 경로를 오류 메시지에 포함합니다.
    if missing:
        raise FileNotFoundError(f"missing media files: {missing[:5]}")
    # 검증된 manifest item 목록을 반환합니다.
    return items


def build_index(args: argparse.Namespace) -> None:
    """manifest 전체 media를 임베딩하고 group context index를 저장합니다."""

    # 입력 JSON에서 media item 목록을 읽습니다.
    items = load_manifest(args.manifest)
    # 필요한 modality 종류만 추출해 불필요한 encoder checkpoint loading을 피합니다.
    modalities = list(dict.fromkeys(item.modality for item in items))
    # LanguageBind 원본 구현을 감싼 runtime encoder를 생성합니다.
    encoder = LanguageBindContextEncoder(
        modalities=modalities,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    # 모든 media item을 shared-space embedding으로 변환합니다.
    embeddings = encoder.encode_items(items)
    # 같은 group_id의 item들을 지정 fusion 전략으로 결합하는 index를 만듭니다.
    index = ContextIndex(items=items, embeddings=embeddings, fusion=args.fusion)
    # 이후 모델을 다시 실행하지 않고 검색할 수 있도록 파일로 저장합니다.
    index.save(args.output)
    # 자동화에서 확인하기 쉬운 완료 메시지를 출력합니다.
    print(f"indexed {len(items)} media items across {len(index.group_ids)} groups -> {args.output}")


def search_index(args: argparse.Namespace) -> None:
    """저장된 group context를 자연어 query로 검색합니다."""

    # 저장된 item/group embedding과 metadata를 복원합니다.
    index = ContextIndex.load(args.index)
    # query text encoder에는 index에 실제 존재하는 modality 중 하나가 함께 로드되어야 shared text tower가 초기화됩니다.
    modalities = list(dict.fromkeys(item.modality for item in index.items))
    # 검색용 LanguageBind encoder를 생성합니다.
    encoder = LanguageBindContextEncoder(
        modalities=modalities,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    # 자연어 query를 shared-space embedding으로 변환합니다.
    query = encoder.encode_text(args.text)
    # fused group context를 점수순으로 검색합니다.
    results = index.search(query, top_k=args.top_k)

    # 각 group 결과를 사람이 읽기 쉬운 한 줄 요약으로 출력합니다.
    for result in results:
        print(f"{result.score:.6f}\t{result.group_id}")
        # group 판단에 기여한 개별 modality evidence도 들여쓰기해서 표시합니다.
        for evidence in result.evidence:
            print(
                f"  {evidence.score:.6f}\t{evidence.modality}\t{evidence.item_id}\t{evidence.path}"
            )


def create_parser() -> argparse.ArgumentParser:
    """index/search subcommand parser를 생성합니다."""

    # 최상위 CLI parser를 생성합니다.
    parser = argparse.ArgumentParser(prog="multimodal-context")
    # 서로 다른 실행 흐름을 subcommand로 나눕니다.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # manifest를 context index로 변환하는 명령을 정의합니다.
    index_parser = subparsers.add_parser("index", help="build a fused multimodal context index")
    # media manifest JSON 파일을 필수 위치 인자로 받습니다.
    index_parser.add_argument("manifest", type=Path)
    # 생성할 index 디렉터리를 필수 옵션으로 받습니다.
    index_parser.add_argument("--output", type=Path, required=True)
    # 같은 modality가 많이 반복될 때 편향을 줄이는 balanced fusion을 기본값으로 사용합니다.
    index_parser.add_argument("--fusion", choices=["mean", "balanced"], default="balanced")
    # CUDA 또는 CPU 장치를 필요하면 직접 지정할 수 있습니다.
    index_parser.add_argument("--device", default=None)
    # Hugging Face checkpoint cache 위치를 프로젝트 밖으로 변경할 수 있습니다.
    index_parser.add_argument("--cache-dir", default="./cache_dir")
    # 실행 함수를 subcommand에 연결합니다.
    index_parser.set_defaults(handler=build_index)

    # 이미 생성한 context index를 검색하는 명령을 정의합니다.
    search_parser = subparsers.add_parser("search", help="search fused contexts with natural language")
    # 검색 대상 index 디렉터리를 필수 위치 인자로 받습니다.
    search_parser.add_argument("index", type=Path)
    # 자연어 query는 필수 옵션으로 받습니다.
    search_parser.add_argument("--text", required=True)
    # 반환할 group 결과 개수를 설정합니다.
    search_parser.add_argument("--top-k", type=int, default=5)
    # 검색 모델 실행 장치를 필요하면 직접 지정할 수 있습니다.
    search_parser.add_argument("--device", default=None)
    # 모델/tokenizer cache 위치를 설정합니다.
    search_parser.add_argument("--cache-dir", default="./cache_dir")
    # 실행 함수를 subcommand에 연결합니다.
    search_parser.set_defaults(handler=search_index)

    # 완성된 parser를 반환합니다.
    return parser


def main() -> None:
    """CLI entry point입니다."""

    # 현재 argv를 parser로 해석합니다.
    args = create_parser().parse_args()
    # 선택한 subcommand handler를 실행합니다.
    args.handler(args)


if __name__ == "__main__":
    # module 직접 실행도 동일하게 지원합니다.
    main()

