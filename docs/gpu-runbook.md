# K3s GPU 실행 가이드

## 확인된 구성

2026-10-20 조사 당시 `keiailab` 컨텍스트의 e101/e102는 Ready였다.
e101은 RTX 4070 Ti, e102는 RTX 4070이며 노드별 GPU 할당량은 1개다.
각 GPU 메모리는 노드 라벨 기준 약 12GB다. e101은 할당 0개, e102는 실행 중인
다른 워크로드가 1개를 할당받았다. 물리 GPU 이용률과 Kubernetes 할당량은 다른 값이다.

`nvidia` RuntimeClass와 `ceph-rbd` StorageClass가 존재한다.
이 최초 조사는 CUDA 프로그램 실행 성공이나 새 PVC 생성 성공의 증거가 아니다.
이후 2026-10-21 e101에서 공식 런타임 이미지와 코드 ConfigMap으로 GPU 학습·PVC 저장을
검증했다. [실행 기록](gpu-test-e101-2026-10-21.md)을 참고한다.
아래는 별도 애플리케이션 이미지를 빌드해 Namespace, PVC, Job을 생성하는 배포 절차다.

## 실행 전 상태

```bash
kubectl --context keiailab get nodes e101 e102
kubectl --context keiailab describe node e101
kubectl --context keiailab describe node e102
kubectl --context keiailab get runtimeclass nvidia
kubectl --context keiailab get storageclass ceph-rbd
```

노드의 Allocated resources에서 GPU 점유를 확인한다. e102가 점유 중이면 작업이 Pending으로
대기할 수 있다. 기존 서비스 축소, GPU 공유 정책 변경, 우선순위 선점은 이 실습에 포함하지 않는다.

## 이미지 빌드

클러스터는 amd64이므로 Apple Silicon에서도 `--platform linux/amd64`를 지정한다.
접근 권한이 있는 레지스트리에 아래 명령으로 빌드·push한 후 생성된 digest를 사용한다.
`REGISTRY_IMAGE`는 사용 가능한 레지스트리 저장소로 교체한다. 비공개 이미지라면
해당 Namespace의 imagePullSecret 구성이 별도로 필요하다. 토큰을 파일에 넣지 않는다.

```bash
export REGISTRY_IMAGE=ghcr.io/jongko54/mlops
export REVISION="$(git rev-parse HEAD)"
docker buildx build --platform linux/amd64 \
  --build-arg GIT_COMMIT="$REVISION" \
  --tag "$REGISTRY_IMAGE:$REVISION" --push .
docker buildx imagetools inspect "$REGISTRY_IMAGE:$REVISION"
```

베이스 이미지는 digest로 고정하고 추가 Python 패키지는 버전을 고정한다.
로컬 uv 환경과 CUDA 이미지의 PyTorch 부가 의존성은 완전히 같지 않을 수 있으며
실행별 메타데이터에서 라이브러리·CUDA 버전을 확인한다.

## 매니페스트 검토와 실행

아래 `IMAGE`에는 방금 빌드한 **애플리케이션 이미지** digest를 넣는다.
기본 YAML의 `build-required` 이미지는 실행용이 아니며 render 스크립트는 digest를 강제한다.

```bash
export IMAGE="$REGISTRY_IMAGE@sha256:실제_64자리_digest"
export RUN_ID="$(date -u +%Y%m%d%H%M%S)"
uv run --frozen python scripts/render_job.py \
  --node e101 --image "$IMAGE" --run-id "$RUN_ID" > /tmp/mlops-e101.yaml
cat /tmp/mlops-e101.yaml

kubectl --context keiailab apply -f deploy/k8s/namespace.yaml
kubectl --context keiailab apply --dry-run=server -f /tmp/mlops-e101.yaml
kubectl --context keiailab apply -f /tmp/mlops-e101.yaml
kubectl --context keiailab -n mlops-lab wait \
  --for=condition=complete "job/digits-train-e101-$RUN_ID" --timeout=1800s
kubectl --context keiailab -n mlops-lab logs "job/digits-train-e101-$RUN_ID"
```

e102는 GPU가 사용 가능할 때 별도로 `--node e102`로 렌더한다.
동시 실행은 독립 실험 두 개이며 DDP 같은 분산 학습이 아니다.

## 결과와 문제 해결

Job은 노드별 1Gi PVC에 `/artifacts/<pod-name>/metrics.json`과 통과한 `model.pt`를 저장한다.
두 노드는 별도 PVC를 쓴다. CPU 500m/1Gi를 요청하고 CPU 2개/4Gi와 GPU 1개로 제한한다.
자동 재시도 없이 최대 30분 동안 실행되며, Job이나 Pod를 지워도 PVC는 별도로 유지된다.

```bash
kubectl --context keiailab -n mlops-lab get jobs,pods,pvc
kubectl --context keiailab -n mlops-lab describe job "digits-train-e101-$RUN_ID"
kubectl --context keiailab -n mlops-lab get events --sort-by=.lastTimestamp
```

- `Pending`: GPU 할당, 노드 자원, PVC 상태와 스케줄러 이벤트 확인.
- `ImagePullBackOff`: 애플리케이션 이미지 digest와 해당 레지스트리 접근 권한 확인.
- 종료 코드 1: CUDA·설정·파일 접근 오류 확인.
- 종료 코드 2: 검증 기준 미달. 로그의 지표를 살펴보고 새 run ID로 새 실험 실행.
- 완료: 로그의 `device=cuda`, 실제 GPU 이름, `status=passed`까지 확인.

완료된 Pod에서 `kubectl cp`가 안 되는 경우 해당 PVC를 마운트한 별도 조회 Pod가 필요하다.
이 초기 구성은 아티팩트 다운로드 API나 모델 레지스트리를 제공하지 않는다.
운영 데이터를 포함할 수 있으므로 PVC 자동 삭제 명령을 제공하지 않는다.
실험 취소 시 해당 run ID의 Job만 삭제하고 PVC는 보존한다.
