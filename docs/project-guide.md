# 프로젝트 실행 가이드

CH1·CH2의 MLOps 개념을 정리하고 CPU 및 K3s GPU에서 같은 작은 학습 파이프라인을 실행한다.

- [학습 정리](ch01-ch02.md): 수명주기, DevOps와의 차이, 성숙도, 구현과 개념의 대응.
- [GPU 실행 가이드](gpu-runbook.md): e101/e102의 개별 GPU Job.
- [검증 기록](verification.md): 실행한 검사와 아직 검증하지 않은 범위.
- [초기 구현 계획](implementation-plan.md).

## 빠른 시작

Python 3.11과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
uv sync --frozen --python 3.11
uv run --frozen pytest -q
uv run --frozen ruff check .
PYTHONPATH=src uv run --frozen python -m mlops_lab.train \
  --device cpu --output artifacts/first-run
```

동일 경로를 재사용하면 실패한다. 다음 실험에는 `artifacts/second-run`처럼 새 경로를 지정한다.
기본값은 seed=42, epochs=30, 검증 정확도 기준 0.90이다. 결과는 다음과 같다.

| 파일 | 의미 |
|---|---|
| `metrics.json` | 설정, 검증·테스트 정확도, 데이터·분할·소스 해시, 커밋, 실행 환경 |
| `model.pt` | 검증 기준을 통과한 모델의 가중치와 메타데이터 |

검증 기준을 통과하면 종료 코드 0, 기준 미달은 2, 잘못된 설정이나 실행 오류는 1이다.
실패한 모델은 저장하지 않으며 테스트 정확도도 계산하지 않는다.

```bash
PYTHONPATH=src uv run --frozen python - <<'PY'
from sklearn.datasets import load_digits
from mlops_lab.train import predict
print(predict("artifacts/first-run/model.pt", load_digits().data[:5]))
PY
```

이 명령은 저장된 모델의 재로딩 예제다. 처음 5개 샘플의 출력으로 일반화 성능을 평가하지 않는다.

## 데이터와 범위

scikit-learn 내장 digits의 1,797개 8×8 이미지로 10개 숫자를 분류한다.
학습/검증/테스트를 약 60/20/20으로 층화 분할하고 픽셀을 16으로 나눈다.
독립된 테스트 세트는 검증을 통과한 뒤 보고용으로만 평가한다.
원본 데이터 설명과 출처는
[scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)를 참고한다.

이 예제는 실제 문서 OCR 서비스, 분산 학습, 자동 재학습 플랫폼을 구현한 것이 아니다.
e101/e102는 노드별 단일 GPU 실행 대상이며 각각의 VRAM을 합산하지 않는다.
`--device cuda`에서 CUDA를 사용할 수 없으면 CPU로 전환하지 않고 실패한다.
같은 환경에서 재현 가능한 연산을 사용하지만 서로 다른 하드웨어·라이브러리의 비트 단위
동일 결과까지 보장하지 않는다.

## 저장소와 CI

- GitHub: <https://github.com/jongko54/mlops> — 공개 코드와 학습 정리.
- Forgejo: <https://git.keiailab.com/sam/mlops> — 같은 커밋을 보관하는 비공개 저장소.

GitHub push/PR에서 CPU 테스트·린트·매니페스트 렌더를 수행한다.
CI는 클러스터 접근 권한을 요구하지 않으며 배포와 저장소 간 자동 동기화는 구성하지 않았다.
`origin`은 GitHub, `forgejo`는 Forgejo로 사용한다. 변경 반영은 두 원격에 각각 push한다.
강의 원문·그림, 데이터 파일, 모델, 자격증명은 커밋하지 않는다.
