# CH4·CH5 연결 실습 실행 방법

기존 `train.py`는 그대로 사용했다. `platform_lab.py`는 MLflow 실험·모델 등록,
Feast 피처 등록·조회 검증을 맡고, `platform_pipeline.py`는 배치 추론을 네 단계로 연결한다.

## 이번 구성

- MLflow 3.16.0: 실행별 SQLite DB와 로컬 파일 아티팩트 저장소. 공유 HTTP 서버는 아니다.
- Feast 0.67.0: Parquet offline store, SQLite online store. Redis/MinIO 서버는 사용하지 않는다.
- KFP 2.18.0: Python DSL을 YAML로 컴파일하고 `SubprocessRunner`로 실제 실행했다.
- GPU 실행: 기존 k3s의 e101에서 단일 Job으로 실행하고 Ceph RBD PVC에 결과를 보관했다.
- CPU에서 배치 추론한다. GPU는 기존 PyTorch MLP 훈련에 사용했다.

KFP 서버/UI/컨트롤러를 설치한 구성은 아니다. 컴포넌트들은 같은 Pod 안의 별도 프로세스로
실행했다. 각 컴포넌트를 별도 Pod로 실행하려면 의존성을 포함한 이미지와 원격 저장소 또는
PVC 전달 설정이 필요하다. 생성된 YAML만 서버에 업로드하면 바로 동작하는 구성은 아니다.

## 로컬 CPU 재현

저장소 루트에서 실행했다. Python 3.11과 uv가 필요하다.

```bash
uv venv --python 3.11 artifacts/platform-venv
uv pip install --python artifacts/platform-venv/bin/python \
  -r requirements-platform.txt torch==2.8.0
PYTHONPATH="$PWD/src" FEAST_USAGE=False MLFLOW_ENABLE_TELEMETRY=false \
  artifacts/platform-venv/bin/python -m mlops_lab.platform_lab \
  --device cpu --output artifacts/ch45-cpu-new-run
```

출력 경로는 실행마다 새로 지정해야 한다. 기존 디렉터리를 덮어쓰지 않는다.
직접 생성한 모델만 재로딩한다. PyTorch MLflow 모델은 이번 버전에서 `pickle` 형식을
명시했으므로 외부에서 받은 신뢰할 수 없는 모델을 이 경로로 로딩하면 안 됨.

## e101 GPU 재현

먼저 e101의 실제 GPU 할당을 확인한 뒤 고유한 Job 이름으로 실행했다. 렌더러는 e101에만
배치하며 e102 설정이나 기존 서비스는 변경하지 않는다. 사용 context는 명시적으로 지정한다.

```bash
uv run --frozen python scripts/render_platform_lab.py \
  --name digits-platform-e101-new-run > artifacts/platform-job.yaml
kubectl --context keiailab apply --dry-run=server -f artifacts/platform-job.yaml
kubectl --context keiailab apply -f artifacts/platform-job.yaml
kubectl --context keiailab logs -n mlops-lab job/digits-platform-e101-new-run -f
kubectl --context keiailab get job -n mlops-lab digits-platform-e101-new-run
```

공식 PyTorch 이미지 digest를 고정했다. 소스와 requirements를 해시한 불변 ConfigMap을
만들고 Pod annotation에 해당 해시를 남김. 런타임에 임시 볼륨으로 패키지를 설치하는
학습용 방식이며 모든 전이 의존성을 고정한 배포 이미지 빌드 경로는 아니다.
실행 시 설치된 패키지 목록도 PVC에 기록했다.

CPU request/limit은 500m/2, 메모리는 2Gi/6Gi, GPU는 1/1이다. 시간 상한 1,800초,
재시도 0, 비루트 사용자, 읽기 전용 루트 파일시스템, 서비스 계정 토큰 미마운트를 적용했다.
결과용 PVC `platform-artifacts-e101`은 5Gi이고 기존 CH1·2 PVC와 분리했다.

## 결과에서 확인할 파일

실행 디렉터리는 `/artifacts/<Pod 이름>/`이다.

| 파일 | 확인하는 내용 |
|---|---|
| `training/metrics.json` | 실제 장치, GPU, 데이터·분할 해시, 검증·테스트 정확도 |
| `training/model.pt` | 기존 학습 코드의 체크포인트 |
| `mlflow.db`, `mlruns/` | 실험 메타데이터·모델 버전과 아티팩트 |
| `model-record.json` | run ID, 모델 URI, 재로딩 확인 |
| `feast/` | 피처 원본 Parquet, registry, online SQLite |
| `feast-checks.json` | 미래 피처 제외와 온라인 값 일치 검사 |
| `pipeline.yaml`, `kfp-outputs/` | KFP 정의와 컴포넌트별 출력 |
| `batch-predictions.json` | 테스트 360개 예측과 정답 수 |
| `verification.json` | 전체 검증 결과와 실행 범위 |

digits에는 실제 이벤트 시각이 없으므로 피처 시간은 실습용으로 생성했다. 과거 시점 조회에
미래 값이 섞이지 않는지 확인하려고 sample 0의 미래 버전을 따로 넣었음. 배치 추론 지표는
기존 테스트 세트를 다시 사용한 연결 검증이며 새로운 독립 일반화 성능 평가가 아니다.

검증 기준 미달 모델은 등록하지 않는다. 모델 입력 피처는 64개이고 정규화 범위는 [0, 1]이다.
누락·null·범위 이탈·중복 ID·다른 ID는 추론 전에 거부한다. 반환 행의 순서가 달라도 요청한
sample ID 순서로 맞춘 뒤 추론한다.

공식 참고: [KFP 로컬 실행](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/),
[Feast online store](https://github.com/feast-dev/feast/blob/master/docs/getting-started/components/online-store.md),
[MLflow 모델 저장 API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html).
