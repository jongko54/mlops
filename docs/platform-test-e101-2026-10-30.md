# CH4·CH5 e101 연결 실습 — 2026-10-30

기존 CH1·2 숫자 분류 학습 코드를 유지하고 MLflow·Feast·KFP를 연결했다.
로컬 CPU 통합 실행을 먼저 통과한 뒤 e101 GPU에서 같은 흐름을 실행했다.

## 실제 실행 결과

| 항목 | 확인 결과 |
|---|---|
| Job | `digits-platform-e101-20261030-01` |
| Pod / 노드 | `digits-platform-e101-20261030-01-slgx9` / e101 |
| 상태 | Job Complete 1/1, Pod Succeeded, 컨테이너 종료 코드 0 |
| Job 실행 시간 | 116초, 이미지 준비와 패키지 설치 포함 |
| 장치 | NVIDIA GeForce RTX 4070 Ti, CUDA 12.6 |
| Python / PyTorch | 3.11.13 / 2.8.0+cu126 |
| MLflow / Feast / KFP | 3.16.0 / 0.67.0 / 2.18.0 |
| 데이터 분할 | train 1,078 / validation 359 / test 360 |
| 검증 / 테스트 정확도 | 95.54% / 96.67% |
| MLflow run ID | `2d428129e0d3405e938993d5f38878b5` |
| 등록 모델 URI | `models:/digits-mlp-ch45/1` |
| 등록 모델 재로딩 | 저장 전 모델의 샘플 출력과 일치 |
| Feast 검증 | 미래 피처 제외, online store 조회값과 정규화 원본 일치 |
| KFP 결과 | read-data, retrieve-features, run-inference, write-data 모두 SUCCESS |
| 배치 추론 | 360개 중 348개 정답, 정확도 0.9666666667 |
| 결과 저장 | `platform-artifacts-e101`, Ceph RBD, 5Gi, RWO |

실행 소스 ConfigMap SHA-256은 `47395b9374af9964520dadaf3840832ac6d4f96f0716e53aad2b6dd750621abe`이다.
학습 코드 SHA-256은 `9895286f7c3245a5b2951795679b5f1f785bab32f002e0726713bd22886f1156`이다.
Pod에는 Git 메타데이터를 넣지 않아 `metrics.json`의 git_commit은 `unknown`이다.
실행한 코드는 불변 ConfigMap과 위 소스 해시로 식별한다.

## 무엇을 검증했는가

- 기존 모델을 CUDA에서 학습하고 검증 기준을 통과한 모델만 MLflow Registry에 등록했다.
- 모델 버전 URI로 다시 불러와 저장 전 모델의 출력과 비교했다.
- Feast 피처는 정규화한 64개 픽셀이다. sample 0에 미래 버전을 넣고 과거 조회에 섞이지 않는지 검사했다.
- Online store에서 테스트 샘플 360개를 읽어 원본 정규화 값과 비교했다.
- KFP 배치에서는 요청 ID와 시점 → 피처 조회 → 등록 모델 추론 → 결과 저장을 실행했다.
- 피처 스키마 누락, null, 범위 이탈, 중복 ID, 다른 ID를 거부하고 순서를 맞추는 테스트를 추가했다.
- 기존 학습 테스트를 포함한 로컬 pytest 14개, Ruff 검사·포맷 검사, Kubernetes 서버 dry-run을 통과했다.

## 실행 범위와 한계

MLflow는 실행별 SQLite DB와 파일 아티팩트 저장소를 사용했다. Feast는 Parquet offline /
SQLite online store를 사용했다. KFP는 Python DSL을 YAML로 컴파일한 뒤 단일 Pod 안에서
`SubprocessRunner`로 실행했다. 각 단계는 별도 프로세스이며 별도 Pod가 아니다.

공유 MLflow HTTP 서버, Redis·MinIO, KFP 서버/UI/컨트롤러, 캐시·재시도, 실시간 서빙,
자동 재학습·모델 승격은 이번 실습 범위에 포함하지 않았다. 생성된 YAML의 원격 실행에는
코드 이미지와 원격 저장소 또는 PVC 전달 설정이 추가로 필요하다.

Feast 시간은 실습용으로 생성했다. 배치 추론은 기존 테스트 세트를 다시 읽은 연결 검증이며
새 평가 세트로 일반화 성능을 검증한 결과가 아니다. GPU 속도 비교나 분산 학습도 수행하지
않았다. e102는 사용하지 않았다.

패키지는 실행 시 임시 볼륨에 설치했다. 주요 버전은 고정했고 설치 결과를 PVC에 기록했지만
전이 의존성까지 잠근 배포 이미지 빌드·레지스트리 push는 검증하지 않았다.

## 재현과 증거

[실행 방법](ch04-ch05-runbook.md), [실습 코드](../src/mlops_lab/platform_lab.py),
[KFP 코드](../src/mlops_lab/platform_pipeline.py), [Job 렌더러](../scripts/render_platform_lab.py).

로컬 `artifacts/platform-e101-20261030-01.log`, `platform-job-result.json`,
`platform-pod-result.json`, `platform-e101-verification.json`에 로그·상태·결과를 보관했다.
이 디렉터리는 Git에서 제외했다. PVC에는 모델, DB, Parquet, KFP 단계별 산출물과
`verification.json`을 보관했다.

학습 Job 종료 후 별도 읽기 전용 Job `platform-artifact-check-20261030`으로 PVC를 다시 열었다. 모델·MLflow DB·Feast registry/online DB·Parquet·파이프라인·배치 결과 등 9개 파일이 존재하고 비어 있지 않은지 확인했고, 저장된 `verification.json`의 상태와 정답 수를 검사했다. 이 Job도 Complete 1/1로 종료했다.
