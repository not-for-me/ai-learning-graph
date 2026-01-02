# Math Foundations - Review Questions

> Questions and insights from actual learning sessions

## 2026-01-02: 편미분(Partial Derivative) 개념 학습

### Q1: 편미분이란 무엇이고, 왜 "편"미분인가?

**Context**: 다변수 함수에서 각 변수의 영향을 개별적으로 파악하는 방법 학습
**Source**: [GPT conversation - 2026-01-02](https://chatgpt.com/share/6957cb3d-ef2c-8000-a3ea-f0677b260e2d), Claude conversation - 2026-01-02

**Answer**: 
- **정의**: 다변수 함수에서 **한 변수만 변화**시키고 나머지는 상수로 고정한 채 미분
- **"편"의 의미**: 전체 변화가 아닌 **부분적(partial)** 변화만 측정
- **표기법**: ∂f/∂x (일반 미분 d와 구분하여 ∂ 사용)
- **예시**: f(x,y) = x²y + y³
  - ∂f/∂x = 2xy (y는 상수 취급)
  - ∂f/∂y = x² + 3y² (x는 상수 취급)

**Related nodes**: 
- `math_foundations:derivative` - 단일 변수 미분
- `math_foundations:gradient` - 편미분들의 벡터
- `deep_learning:backpropagation` - 역전파에서의 활용

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-02: 첫 학습 및 개념 이해

---

### Q2: "각 가중치가 손실에 미치는 영향을 개별적으로 파악한다"는 것의 구체적 의미는?

**Context**: 신경망 학습에서 편미분의 실제 역할 이해
**Source**: Claude conversation - 2026-01-02

**Answer**: 
- **핵심 질문**: "가중치 $w_i$를 살짝 바꾸면 손실 $L$이 얼마나 변할까?"
- **수학적 표현**: $\frac{\partial L}{\partial w_i}$
- **해석 예시**:

| 편미분 값 | 의미 | 업데이트 방향 |
|-----------|------|---------------|
| $\frac{\partial L}{\partial w_1} = +2$ | $w_1$ 증가 → 손실 증가 | $w_1$을 **줄여야** 함 |
| $\frac{\partial L}{\partial w_2} = -0.5$ | $w_2$ 증가 → 손실 감소 | $w_2$를 **키워야** 함 |
| $\frac{\partial L}{\partial w_3} = 0$ | $w_3$는 손실에 영향 없음 | 변경 불필요 |

- **"개별적으로"의 의미**: $w_1$의 영향을 볼 때 $w_2, w_3$는 고정 (통제 변인 개념)
- **과학 실험 비유**: 한 변수씩 바꿔가며 영향 측정

**Related nodes**: 
- `math_foundations:partial_derivative` - 편미분 정의
- `deep_learning:gradient_descent` - 실제 가중치 업데이트
- `deep_learning:backpropagation` - gradient 계산 과정

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-02: 구체적 예시로 완전 이해

---

### Q3: 왜 신경망 학습에서 편미분이 필수인가?

**Context**: AI Engineer 관점에서 편미분의 필요성과 의미 정리
**Source**: Claude conversation - 2026-01-02

**Answer**: 

**1. Gradient Descent의 작동 원리**
```python
# 학습의 핵심 한 줄
for each weight w:
    w = w - learning_rate * (∂L/∂w)
```
- 편미분 값이 **방향**(증가/감소)과 **크기**(민감도)를 제공
- 큰 gradient = 손실에 큰 영향 → 큰 업데이트

**2. 고차원 문제의 분해**
- 신경망은 수백만 개의 가중치 보유
- 전체를 한꺼번에 최적화 불가능
- **편미분의 역할**: 고차원 최적화 → 1차원 문제들의 조합으로 분해

**3. 실무에서 직접 계산하지 않지만 알아야 하는 이유**
- PyTorch/TensorFlow가 `loss.backward()` 한 줄로 처리
- 하지만 이해가 필요한 상황:

| 상황 | 편미분 이해가 필요한 이유 |
|------|-------------------------|
| Gradient vanishing/exploding | 왜 gradient가 0이나 ∞가 되는지 |
| Learning rate 튜닝 | gradient 크기와 step size의 관계 |
| Batch normalization | gradient flow 안정화 원리 |
| Skip connection (ResNet) | gradient 직접 전파 경로의 이유 |

**Related nodes**: 
- `math_foundations:gradient` - 편미분들의 벡터
- `deep_learning:gradient_descent` - 최적화 알고리즘
- `deep_learning:backpropagation` - 역전파 메커니즘

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-02: AI Engineer 관점에서 정리 완료

---

### Q4: 편미분과 역전파(Backpropagation)의 관계는?

**Context**: GPT 대화에서 역전파 시 손실함수 최소화 과정 학습
**Source**: [GPT conversation - 2026-01-02](https://chatgpt.com/share/6957cb3d-ef2c-8000-a3ea-f0677b260e2d)

**Answer**: 
- **역전파의 목적**: 출력층에서 입력층 방향으로 각 가중치의 $\frac{\partial L}{\partial w}$ 계산
- **Chain Rule 활용**: 
  - $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$
  - 합성함수의 미분을 연쇄적으로 적용
- **편미분의 역할**: 각 층에서 "이 가중치가 최종 손실에 얼마나 기여하는가" 계산
- **핵심 통찰**: 
  - 편미분 = **방향 정보** (증가/감소)
  - 우리가 최소화하는 것 = **손실 함수 값**
  - "기울기를 최소화"가 아니라 **"기울기를 따라 손실을 최소화"**

**Related nodes**: 
- `math_foundations:chain_rule` - 연쇄 법칙
- `math_foundations:partial_derivative` - 편미분 정의
- `deep_learning:backpropagation` - 역전파 구현

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-02: GPT 학습 내용 정리 및 Claude로 심화

---

### Q5: 편미분의 한 문장 요약

**Context**: 핵심 개념을 간결하게 정리
**Source**: Claude conversation - 2026-01-02

**Answer**: 

> **편미분은 "이 가중치 하나를 건드리면 최종 손실이 얼마나 변하는가"를 측정하는 도구이고, 이 정보가 있어야 각 가중치를 손실이 줄어드는 방향으로 업데이트할 수 있다.**

**핵심 포인트**:
1. 다변수 함수에서 **한 변수의 영향만** 측정
2. 나머지 변수는 **상수로 고정**
3. 결과값은 **방향(+/-)과 크기(민감도)** 정보 제공
4. 이를 모아 **gradient 벡터** 구성 → **gradient descent** 적용

**Related nodes**: 
- `math_foundations:partial_derivative` - 정의
- `math_foundations:gradient` - 벡터로 확장
- `deep_learning:gradient_descent` - 실전 적용

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-02: 핵심 요약 완료

---

## 🔗 학습 흐름 정리

```
derivative (미분)
    │
    ▼
partial_derivative (편미분)  ← 현재 학습
    │
    ▼
gradient (그래디언트 = 편미분들의 벡터)
    │
    ▼
gradient_descent (경사하강법)
    │
    ▼
backpropagation (역전파 = chain rule + 편미분)
```

---

## 학습 요약

**주요 성과**:
- 편미분의 정의와 "편(partial)"의 의미 이해
- "각 가중치의 개별적 영향 파악"의 구체적 의미 파악
- AI Engineer 관점에서 편미분이 필요한 이유 정리
- 역전파와의 연결고리 확립

**다음 학습 예정**: 
- `chain_rule` 심화 → 역전파에서의 활용
- `gradient` → 편미분들의 벡터로서의 기하학적 의미

**다음 복습 예정**: 2026-01-16 (2주 후)