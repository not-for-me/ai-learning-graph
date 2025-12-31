# Math Foundations - Review Questions

> Questions and insights from actual learning sessions

## 2026-01-01: 미분(Derivative) 개념 검증

### Q1: 미분값이 0인 점은 극값인가? 변곡점인가?

**Context**: YAML quiz에서 "미분값이 0인 점의 의미"를 확인하던 중, 극값과 변곡점의 차이가 헷갈림
**Source**: [Claude conversation - 2026-01-01](https://claude.ai/share/e101ef63-4cba-4085-8284-bc2634f2cb2d)

**Answer**: 
- 미분값(f'=0)이 0인 점은 **극값 또는 변곡점의 후보**
- **극값**: 이계도함수(f'')로 판별 또는 전후 기울기 변화 확인 필요
  - f''(x) > 0 → 극소
  - f''(x) < 0 → 극대
- f''(x)=0이면 판정 불가 → 고차 도함수나 좌우 기울기 변화를 추가 확인
- **변곡점**: 이계도함수가 0이고 오목/볼록이 바뀌는 점
- **중요**: f'(x)=0이라고 무조건 극값은 아님 (예: f(x)=x³에서 x=0)

**Related nodes**: 
- `math_foundations:derivative` - 미분의 기본 개념
- `math_foundations:gradient` - 다변수로 확장
- `deep_learning:gradient_descent` - 학습에서의 응용

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-01: 첫 학습 및 완전 이해

---

### Q2: 함수의 연속성과 미분가능성의 관계는?

**Context**: ReLU의 미분 불가능성을 이해하는 과정에서 f(x)=|x| 예시 학습
**Source**: [Claude conversation - 2026-01-01](https://claude.ai/share/e101ef63-4cba-4085-8284-bc2634f2cb2d)

**Answer**: 
- **연속 ≠ 미분가능**: 연속이지만 미분 불가능한 함수 존재
- **미분가능 ⇒ 연속**: 미분되면 반드시 연속이지만, 연속이라고 항상 미분가능한 것은 아님
- **f(x)=|x|의 x=0**: 
  - 연속 ✓ (함수값 끊어지지 않음)
  - 미분 불가능 ✗ (좌미분=-1, 우미분=+1, 좌≠우)
- **미분가능의 조건**: 좌미분 = 우미분 (극한의 정의상 필수)
- **기하학적 의미**: 유일한 접선이 존재해야 함 (V자 꺾임 불가)

**Related nodes**: 
- `math_foundations:derivative` - 미분의 정의
- `deep_learning:activation_function` - ReLU와의 연결

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-01: 좌미분/우미분 개념 완전 이해

---

### Q3: ReLU는 x=0에서 미분 불가능한데 왜 학습이 되는가?

**Context**: 미분 불가능한 활성화 함수가 실전에서 작동하는 이유 의문
**Source**: [Claude conversation - 2026-01-01](https://claude.ai/share/e101ef63-4cba-4085-8284-bc2634f2cb2d)

**Answer**: 
- **확률적 이유**: 부동소수점 특성상 x=0 정확히 떨어질 확률 ≈ 0
- **Subgradient 해결책**: x=0일 때 [0,1] 범위의 subgradient 사용
- **PyTorch 구현**: x=0에서 gradient=0으로 설정 (왼쪽 극한과 일치)
```python
  gradient = (x > 0).float()  # x=0이면 False → 0
```
- **실전 영향**: x=0 발생이 거의 없어서 subgradient 선택이 학습에 미미한 영향
- **Dead ReLU 리스크**: x<0 구간의 0 기울기로 뉴런이 멈출 수 있어 Leaky/Parametric ReLU로 완화

**Related nodes**: 
- `math_foundations:derivative` - 미분의 정의
- `deep_learning:backpropagation` - gradient 계산
- `deep_learning:activation_function` - ReLU 구현

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-01: 이론과 구현의 간극 이해

---

### Q4: Plateau에서 학습이 느린 이유는?

**Context**: Gradient가 작은 지역에서 학습 효율성 문제 이해
**Source**: [Claude conversation - 2026-01-01](https://claude.ai/share/e101ef63-4cba-4085-8284-bc2634f2cb2d)

**Answer**: 
- **핵심**: gradient ≈ 0 → 가중치 업데이트량 미미
- **수식**: `w_new = w_old - lr * gradient`
  - gradient가 0.0001이면 변화량 거의 없음
- **학습 효율**: (손실 개선량) / (iteration 횟수) ↓↓↓
- **실전 문제**: 같은 계산 자원으로 개선 효과 매우 작음

**Related nodes**: 
- `math_foundations:gradient` - gradient 정의
- `math_foundations:derivative` - 미분값=0의 의미
- `deep_learning:gradient_descent` - 최적화 과정
- `deep_learning:optimizer` - Adam 등 해결책

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-01: 이론적 이해 및 실전 문제 연결

---

### Q5: Local minimum vs Plateau (Saddle point) - 실전에서 구분 가능한가?

**Context**: 둘 다 gradient=0인데 수학적/실전적 차이 의문
**Source**: [Claude conversation - 2026-01-01](https://claude.ai/share/e101ef63-4cba-4085-8284-bc2634f2cb2d)

**Answer**: 
- **수학적 정의**:
  - Local minimum: ∇f=0, Hessian ($H$)이 **양의 정부호(Positive Definite, $H \succ 0$)** (모든 방향 위로 볼록)
  - Saddle point: ∇f=0, Hessian의 고유값(eigenvalue)이 양수와 음수가 섞여 있음
- **실전에서 구분 불가능**:
  - 유한 정밀도: gradient=1e-8이 0인가?
  - 고차원: 100만 파라미터 중 일부는 min, 일부는 saddle
  - 시각화 한계: 2D로 보면 둘 다 비슷
- **중요한 통찰**: 고차원에서 순수 local minimum은 극히 드묾 (대부분 saddle)
- **실전적 결론**: 구분보다 "빠져나오기"가 중요 (Adam, momentum)

**Related nodes**: 
- `math_foundations:derivative` - f'=0의 의미
- `math_foundations:gradient` - 고차원 확장
- `deep_learning:optimizer` - Saddle point 탈출 전략

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-01: 이론과 실전의 간극 완전 이해, Goodfellow 논문 언급
- **추가 제언**: 향후 2주 후 복습 시, 실제 PyTorch의 `torch.autograd.functional.hessian`을 사용해 간단한 함수의 saddle point를 직접 계산해 볼 것

---

## 🚀 심화 학습 및 확장 계획 (Next Steps)

### 1. Hessian Matrix & 2차 최적화
- **과제**: `math_foundations` 도메인에 `hessian_matrix` 노드를 추가하거나 `gradient` 노드의 심화 개념으로 확장.
- **이유**: Saddle point 판정 및 Newton's Method 등 2차 최적화 알고리즘 이해를 위한 필수 관문.

### 2. 이론-코드 매핑 검증
- **검증 대상**: `torch.nn.functional.relu`의 $x=0$에서의 실제 기울기 반환값 확인.
- **방법**: `x = torch.tensor([0.0], requires_grad=True)`에 대해 직접 backward를 수행하여 `subgradient` 구현체 확인.

---

## 학습 요약

**주요 성과**:
- 미분의 기하학적/수학적 의미 명확히 구분
- 이론(수학)과 구현(PyTorch)의 차이 이해
- 고차원 최적화 문제의 실전적 특성 파악

**다음 복습 예정**: 2026-01-15 (2주 후)