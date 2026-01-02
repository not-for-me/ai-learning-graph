# Review System Guide

This folder contains **deep-dive questions and answers** from actual learning sessions, complementing the structured quizzes in YAML files.

## Purpose

While YAML quizzes test core understanding, review files capture:
- Specific confusions you encountered
- Edge cases and nuances
- Connections between concepts
- Real-world applications you discovered

## Structure

**일자별 파일 구성** (Daily session-based files):
```
reviews/
├── 2026-01-01-derivative.md
├── 2026-01-02-partial-derivative.md
├── 2026-01-03-gradient-descent.md
├── 2026-01-10-attention-mechanism.md
└── README.md (this file)
```

**파일명 규칙**:
- 형식: `YYYY-MM-DD-topic.md`
- `topic`: 해당 날짜에 학습한 핵심 주제 (영문, kebab-case)
- 예시: `2026-01-15-transformer-architecture.md`

**각 파일의 템플릿**:
```markdown
# [Domain Name] - Review Questions

> Questions and insights from actual learning sessions

## YYYY-MM-DD: [Study Session Topic (한글 가능)]

### Q1: [Your first question]

**Context**: [What you were studying]
**Source**: [Link to conversation/paper/video]

**Answer**: 
- [Key point 1]
- [Key point 2]

**Related nodes**: 
- `domain:node_id` - [Brief connection explanation]

**Confidence**: ⭐⭐⭐ (1-3 stars)

**Review history**:
- YYYY-MM-DD: First learned
- YYYY-MM-DD: Reviewed, confidence ⭐ → ⭐⭐

---

### Q2: [Your second question]

... (same format)

---

## 학습 요약

**주요 성과**:
- [What you learned today]

**다음 학습 예정**: [Next topics]
**다음 복습 예정**: YYYY-MM-DD (2주 후)
```

## Workflow

### 1. 학습 세션 시작 시

**새 파일 생성**:
```bash
# 오늘 날짜로 파일 생성
cd reviews/
touch $(date +%Y-%m-%d)-[topic].md

# 예: 2026-01-03-chain-rule.md
```

**파일 헤더 작성**:
```markdown
# [Domain Name] - Review Questions

> Questions and insights from actual learning sessions

## 2026-01-03: Chain Rule 심화 학습
```

### 2. 학습 중 질문 기록

**질문이 생기면 즉시 기록**:
1. 파일 하단에 새 질문 섹션 추가
2. **Context**: 어떤 맥락에서 이 질문이 나왔는지
3. **Source**: 대화 링크, 논문, 영상 등 출처 명시

**예시**:
```markdown
### Q3: [질문 내용]

**Context**: [학습 중이던 내용]
**Source**: [링크]

**Answer**: 
- (학습 후 작성)
```

### 3. 답변 및 정리

**학습 후 답변 작성**:
- 자신의 언어로 재구성
- 핵심 포인트를 bullet point로
- 구체적 예시나 수식 포함

**관련 노드 연결**:
- `domain:node_id` 형식으로 그래프 노드 참조
- 간단한 연결 설명 추가

**신뢰도 설정**:
- ⭐ (1/3): 이제 막 배웠고 아직 불확실함
- ⭐⭐ (2/3): 이해했고 설명 가능
- ⭐⭐⭐ (3/3): 완전 숙달, 타인에게 가르칠 수 있음

### 4. 세션 종료 시

**학습 요약 작성**:
```markdown
## 학습 요약

**주요 성과**:
- 오늘 배운 핵심 개념
- 해결한 의문점

**다음 학습 예정**: 
- 이어서 공부할 주제

**다음 복습 예정**: 2026-01-17 (2주 후)
```

### 5. 주간 복습 (Weekly Review)

**매주 일요일**:
- 지난 주 파일들 훑어보기
- 낮은 신뢰도(⭐) 질문 재학습
- 신뢰도 업데이트

```bash
# 최근 7일 파일 확인
ls -lt reviews/*.md | head -7
```

### 6. 월간 심화 복습 (Monthly Deep Review)

**매월 마지막 주**:
- ⭐ 질문들 집중 복습
- 답변 보지 않고 자가 테스트
- `Review history`에 복습 기록 추가
- 신뢰도 상향 조정

```bash
# 낮은 신뢰도 질문 검색
grep -r "⭐ (" reviews/*.md | grep -v "⭐⭐"
```

## Examples

### ✅ Good Example: 2026-01-05-attention-scaling.md

```markdown
# Transformer - Review Questions

> Questions and insights from actual learning sessions

## 2026-01-05: Attention 메커니즘 구현 중 발견한 스케일링 이슈

### Q1: Scaled Dot-Product Attention에서 왜 sqrt(d_k)로 나누는가?

**Context**: Attention을 직접 구현하던 중 d_k=512일 때 softmax 출력이 원-핫에 가깝게 쏠리는 현상 발견
**Source**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) Section 3.2.1

**Answer**: 
- **문제**: 내적값의 크기가 차원 d_k에 비례하여 증가
  - d_k=512일 때 내적 결과가 수백~수천까지 커짐
- **결과**: Softmax가 포화(saturation) → gradient 소실
- **해결**: sqrt(d_k)로 나눠서 분산을 ~1로 정규화
  - 수식: `attention_scores = QK^T / sqrt(d_k)`
- **실험 결과**: 스케일링 전후 gradient 크기 100배 차이 확인

**Related nodes**: 
- `transformer:scaled_attention` - 핵심 메커니즘
- `math_foundations:variance` - 분산 안정화 원리
- `deep_learning:gradient_flow` - Gradient 소실 문제

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-05: 논문 읽고 이해
- 2026-01-06: 직접 구현하며 확인
- 2026-01-20: 복습, 신뢰도 ⭐⭐ → ⭐⭐⭐

---

### Q2: Multi-head는 왜 필요한가? 하나의 큰 head로는 안 되나?

**Context**: Q1을 해결하고 나서 multi-head의 필요성 의문
**Source**: [Claude conversation](https://claude.ai/share/xxx), [논문] 3.2.2절

**Answer**: 
- **단일 head 한계**: 하나의 attention만으로는 다양한 관계 포착 불가
- **Multi-head 장점**: 
  - 각 head가 서로 다른 representation subspace 학습
  - 예: head1은 구문 구조, head2는 의미 관계, head3는 위치 관계
- **실험 증거**: Head들이 실제로 다른 패턴 학습함 (BERTology 연구들)
- **파라미터 효율**: 8 heads × 64 dim = 512 dim (파라미터 수 동일)

**Related nodes**: 
- `transformer:multi_head_attention` - 구조
- `transformer:scaled_attention` - 각 head의 기본 단위

**Confidence**: ⭐⭐ (2/3)

**Review history**:
- 2026-01-05: 논문 읽음, 아직 체감 안 됨

---

## 학습 요약

**주요 성과**:
- Attention 스케일링의 수학적/실무적 이유 완전 이해
- Multi-head의 필요성은 이해했으나 실전 검증 필요

**다음 학습 예정**: 
- Attention visualization으로 head별 패턴 확인
- BERTology 논문 읽기

**다음 복습 예정**: 2026-01-19 (2주 후)
```

### ❌ Bad Example (피해야 할 예시)

```markdown
## 2026-01-05: Attention

### Q: Attention이 뭔가요?

**Answer**: 중요한 부분에 집중하는 메커니즘입니다.

**Related nodes**: `transformer:attention`

**Confidence**: ⭐⭐⭐
```

**문제점**:
- ❌ Context 없음: 왜 이 질문이 나왔는지 불명확
- ❌ Source 누락: 어디서 배웠는지 추적 불가
- ❌ 깊이 부족: 단순 정의 반복, YAML quiz와 차별점 없음
- ❌ 구체성 없음: 실제 학습 과정이 드러나지 않음

## Tips

### 파일명 짓기

✅ **Good**: 
- `2026-01-05-attention-scaling.md` (구체적 주제)
- `2026-02-10-rlhf-implementation.md` (명확한 키워드)

❌ **Bad**:
- `2026-01-05.md` (주제 없음)
- `2026-01-05-study.md` (너무 일반적)
- `attention.md` (날짜 없음)

### 질문 작성 요령

✅ **Good**: "Multi-head attention이 단일 large head보다 나은 이유는?"
❌ **Bad**: "Multi-head attention이 뭐야?"

✅ **Good**: "Softmax의 temperature가 attention 분포의 sharpness에 미치는 영향은?"
❌ **Bad**: "Temperature가 뭐야?"

**핵심**: **"무엇(What)"보다 "왜(Why)"와 "어떻게(How)"** 질문하기

### 답변 작성 요령

**자신의 언어로**:
- ❌ 논문/문서 복붙
- ✅ 내가 이해한 방식으로 재구성

**직관 포함**:
- ❌ "수식이 그래서 그렇습니다"
- ✅ "기하학적으로 보면 ~라는 의미입니다"

**구체적 예시**:
- ❌ "일반적으로 효과적입니다"
- ✅ "d_k=512일 때 gradient가 100배 차이났습니다"

**연결고리**:
- ❌ 고립된 지식
- ✅ "이전에 배운 X와 연결되는 부분은 ~"

### 신뢰도 관리

**정직하게**:
- ⭐ (불확실): 복습 우선순위 높음
- ⭐⭐ (이해): 정기 복습 대상
- ⭐⭐⭐ (숙달): 가끔 확인만

**주기적 업데이트**:
```markdown
**Review history**:
- 2026-01-05: ⭐ (첫 학습)
- 2026-01-19: ⭐⭐ (복습 후 이해도 향상)
- 2026-02-02: ⭐⭐⭐ (실습하며 완전 숙달)
```

### 출처 표기

**필수 포함**:
- Claude: `[Claude 대화](https://claude.ai/share/xxx)`
- GPT: `[GPT 대화](https://chatgpt.com/share/xxx)`
- 논문: `[Attention Is All You Need](arXiv링크) Section 3.2.1`
- 영상: `[Andrej Karpathy - Attention](YouTube링크) @ 15:30`
- 책: `[Deep Learning Book] Chapter 8, Page 234`

**왜 중요한가**:
- 나중에 재확인 가능
- 출처의 신뢰성 평가
- 학습 경로 추적

## Integration with YAML Graphs

일자별 Review 파일은 YAML 퀴즈를 **보완**합니다:

| YAML Quizzes (`learning-graphs/`) | Daily Reviews (`reviews/`) |
|-----------------------------------|----------------------------|
| 넓이(Breadth) 커버리지 | 깊이(Depth) 탐구 |
| 표준 검증 질문 | 개인화된 의문점 |
| 모든 노드 필수 | 깊이 학습한 부분만 |
| 정적 (변경 드뭄) | 동적 (계속 추가) |
| 노드 단위 | 세션 단위 |

**예시**:
- **YAML quiz**: "Positional encoding의 목적은?"
- **Daily review**: "Transformer에서 sinusoidal PE와 learned embedding 중 언제 무엇이 더 나은가? 실험 결과는?"

**파일 헤더로 도메인 명시**:
```markdown
# Math Foundations - Review Questions    ← 도메인 표시
## 2026-01-02: 편미분 학습              ← 날짜 + 주제
```

**Related nodes로 그래프 연결**:
```markdown
**Related nodes**: 
- `math_foundations:partial_derivative` - 이 노드의 심화 내용
- `deep_learning:backpropagation` - 실전 적용 예시
```

## Searching & Filtering

### 도메인별 검색
```bash
# Math Foundations 관련 파일들 찾기
grep -l "^# Math Foundations" reviews/*.md

# Transformer 관련 질문 찾기
grep -r "# Transformer" reviews/*.md
```

### 날짜별 검색
```bash
# 2026년 1월 학습 내용
ls reviews/2026-01-*.md

# 최근 7일 복습
find reviews -name "*.md" -mtime -7 -type f
```

### 주제별 검색
```bash
# "attention" 관련 모든 질문
grep -r "### Q.*attention" reviews/*.md -i

# "gradient" 키워드 포함 파일들
grep -l "gradient" reviews/*.md
```

### 신뢰도별 필터링
```bash
# 낮은 신뢰도 질문들 (복습 필요)
grep -r "⭐ (" reviews/*.md | grep -v "⭐⭐"

# 완전 숙달 질문들
grep -r "⭐⭐⭐" reviews/*.md
```

## Maintenance

### 파일 정리

**언제 아카이빙하나**:
- 모든 질문이 ⭐⭐⭐이고 3개월+ 경과한 파일
- `_archived/YYYY/` 폴더로 이동 (연도별 정리)

```bash
mkdir -p reviews/_archived/2026
mv reviews/2026-01-*.md reviews/_archived/2026/
```

**언제 삭제하나**:
- **기본적으로 삭제 안 함**
- 학습 여정의 기록
- 나중에 돌아보면 인사이트 발견
- 디스크 공간 부담 미미

### 정기 유지보수

**주간 (일요일)**:
```bash
# 이번 주 학습 파일들 확인
ls -lt reviews/2026-01-*.md | head -7

# 낮은 신뢰도 질문 체크
grep -r "⭐ (" reviews/2026-01-*.md
```

**월간 (마지막 주)**:
```bash
# 이번 달 전체 복습
cat reviews/2026-01-*.md | grep "^## " | sort

# 도메인별 학습 분포 확인
grep "^# " reviews/2026-01-*.md | sort | uniq -c
```

**연간 (12월)**:
```bash
# 올해 학습 통계
echo "총 학습 세션: $(ls reviews/2026-*.md | wc -l)"
echo "도메인별 분포:"
grep "^# " reviews/2026-*.md | sort | uniq -c
```

## Quick Start Guide

### 오늘 처음 공부했다면?

```bash
cd reviews/

# 1. 오늘 날짜 파일 생성
touch $(date +%Y-%m-%d)-[topic].md

# 2. 템플릿 복사 (위 구조 참고)
# 3. 학습하며 질문 기록
# 4. 세션 끝에 답변 및 요약 작성
```

### 빠른 복습이 필요하다면?

```bash
# 낮은 신뢰도 질문들만 보기
grep -A 5 "⭐ (" reviews/*.md | grep -v "⭐⭐"

# 최근 2주 학습 내용 훑기
ls -t reviews/*.md | head -14
```

## Future Enhancements

**CLI 도구 계획**:
- [ ] `python review.py new [topic]` - 오늘 날짜 파일 자동 생성
- [ ] `python review.py search [keyword]` - 키워드로 질문 검색
- [ ] `python review.py stats` - 도메인별/신뢰도별 통계
- [ ] `python review.py review` - 복습 필요 질문 필터링

**자동화 계획**:
- [ ] 신뢰도 decay (X일 미복습 시 자동으로 ⭐ 감소)
- [ ] Anki 플래시카드 export (spaced repetition)
- [ ] Source URL 유효성 검사 (broken link 체크)
- [ ] GitHub Actions로 주간 리포트 자동 생성

**시각화**:
- [ ] 도메인별 학습 진도 그래프
- [ ] 신뢰도 분포 차트
- [ ] 시간대별 학습 패턴 히트맵

---

**Remember**: 
> **가장 좋은 복습 시스템은 실제로 사용하는 시스템입니다.**  
> 간단하게 시작하고, 꾸준히 유지하세요.