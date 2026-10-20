# CH1·CH2 초기 프로젝트 계획

## 범위

Notion CH1·CH2의 개념 학습을 정리하고, 책에서 소개한 OCR 방향으로 확장할 수 있는
8×8 손글씨 숫자 분류 예제를 만든다. 실제 문서 OCR, 추천 시스템, RAG 구현은 이후 단계다.
이번 원격 반영 대상은 GitHub `jongko54/mlops`와 Forgejo `sam/mlops`의 동일한 main 커밋이다.

## 단계와 완료 조건

1. `docs/ch01-ch02.md`: 수명주기, DevOps 차이, 성숙도와 현재 실습의 대응 관계 정리.
2. `src/mlops_lab/train.py`: 고정 시드 데이터 분할, 학습, 검증 기준, 테스트 평가,
   모델·지표·데이터 해시 저장 및 재로딩 추론. CPU 테스트로 실패·정상 경로 검증.
3. `Dockerfile`, `deploy/k8s/`: e101/e102 노드별 단일 GPU Job, 영속 아티팩트,
   명시적 CUDA 요구, 자원·실행 시간 제한. YAML 렌더와 서버 dry-run으로 스키마 검증.
4. 테스트·린트·로컬 학습 결과 확인 후 한 커밋을 두 저장소에 push하고 원격 ref 비교.

## 경계와 실패 처리

- e102 GPU는 조사 시점에 다른 실행 중인 워크로드가 점유했다. 이를 중단하지 않는다.
- 이번 작업은 저장소 초기 구현과 반영이다. 클러스터 실제 apply 및 이미지 배포는
  실행 가이드로 제공하고, dry-run과 실제 GPU 실행을 구분한다.
- 데이터 분할에 테스트 샘플이 섞이면 실패. 검증 기준 미달 시 모델을 승격하지 않는다.
- 기존 결과 디렉터리는 덮어쓰지 않는다. 클러스터 실행별 이름과 결과 경로를 분리한다.
- 공개 GitHub에 강의 본문·그림, 자격증명, kubeconfig, 운영 로그를 올리지 않는다.

## 검증

`uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
`PYTHONPATH=src uv run python -m mlops_lab.train`, `kubectl kustomize`,
Kubernetes API server dry-run. 이미지 빌드와 실제 GPU 실행 여부는 별도로 기록한다.
