# Hugging Face LLM Course Chapter 3.4 - Full Training Loop 요약

> **목표**: Trainer API 없이 순수 PyTorch로 fine-tuning 전체 과정을 직접 구현하기

---

[A full traiding loop](https://huggingface.co/learn/llm-course/chapter3/4) 요약

## 전체 파이프라인 개요

```
데이터 준비 → 모델 로드 → 학습 설정 → 학습 루프 → 평가 루프
```

---

## 1단계: 데이터 준비 (Data Preparation)

### 1.1 데이터셋 로드 및 토큰화

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

**각 코드의 역할:**

| 코드 | 역할 | 왜 필요한가? |
|------|------|-------------|
| `load_dataset("glue", "mrpc")` | GLUE 벤치마크의 MRPC(문장 유사도) 데이터셋 로드 | 학습/검증 데이터 확보 |
| `AutoTokenizer.from_pretrained(checkpoint)` | BERT 모델에 맞는 토크나이저 로드 | 모델이 이해할 수 있는 형태로 텍스트 변환 |
| `tokenize_function` | 두 문장을 함께 토큰화 | BERT는 [SEP] 토큰으로 구분된 문장 쌍을 입력받음 |
| `raw_datasets.map(..., batched=True)` | 전체 데이터셋에 토큰화 함수 적용 | 배치 처리로 속도 향상 |
| `DataCollatorWithPadding` | 배치 내 시퀀스 길이를 동적으로 패딩 | GPU 효율적 처리를 위해 동일 길이 필요 |

### 1.2 데이터셋 후처리 (모델 입력 형식 맞추기)

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
```

**각 코드의 역할:**

| 코드 | 역할 | 왜 필요한가? |
|------|------|-------------|
| `remove_columns([...])` | 원본 텍스트 컬럼 제거 | 모델은 토큰화된 숫자만 필요, 불필요한 데이터 제거 |
| `rename_column("label", "labels")` | 컬럼명 변경 | HuggingFace 모델은 `labels`라는 이름을 기대함 |
| `set_format("torch")` | 반환 형식을 PyTorch 텐서로 설정 | PyTorch 학습 루프에서 바로 사용 가능 |

**처리 전후 비교:**
```
처리 전: ["sentence1", "sentence2", "idx", "label", "input_ids", "attention_mask", "token_type_ids"]
처리 후: ["labels", "input_ids", "attention_mask", "token_type_ids"]
```

### 1.3 DataLoader 생성

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

**각 파라미터의 역할:**

| 파라미터 | 값 | 역할 |
|----------|-----|------|
| `shuffle=True` | 학습 데이터만 | 매 에폭마다 데이터 순서를 섞어 과적합 방지 |
| `batch_size=8` | 8개씩 묶음 | GPU 메모리와 학습 안정성의 균형 |
| `collate_fn=data_collator` | 동적 패딩 함수 | 배치 내 최대 길이에 맞춰 패딩 |

### 1.4 배치 구조 확인

```python
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```

**출력 예시:**
```python
{
    'attention_mask': torch.Size([8, 65]),   # 8개 샘플, 최대 65 토큰
    'input_ids': torch.Size([8, 65]),        # 토큰 ID
    'labels': torch.Size([8]),               # 각 샘플의 정답 레이블
    'token_type_ids': torch.Size([8, 65])    # 문장 구분 (0: 첫번째, 1: 두번째)
}
```

---

## 2단계: 모델 및 학습 설정

### 2.1 모델 로드

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

**역할:**
- 사전학습된 BERT 위에 분류 헤드(classification head) 추가
- `num_labels=2`: MRPC는 이진 분류 (유사/비유사)

**모델 구조:**
```
[BERT Encoder] → [Pooler] → [Classification Head (768→2)]
     ↓                              ↓
 Hidden States              Logits (2개 클래스)
```

### 2.2 모델 동작 확인

```python
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
# tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```

**핵심 포인트:**
- `labels`가 제공되면 자동으로 loss 계산
- `logits`: 각 클래스에 대한 점수 (softmax 적용 전)
- `grad_fn`: 역전파를 위한 연산 그래프 연결됨

### 2.3 Optimizer 설정

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

**AdamW vs Adam:**
- AdamW = Adam + Decoupled Weight Decay
- Weight Decay: 큰 가중치에 패널티를 줘서 과적합 방지
- Decoupled: weight decay를 gradient가 아닌 가중치에 직접 적용

### 2.4 Learning Rate Scheduler 설정

```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)  # 1377
```

**Linear Scheduler 동작:**
```
Learning Rate
    │
5e-5├────╲
    │      ╲
    │        ╲
    │          ╲
  0 ├────────────╲────
    └────────────────→ Steps
    0            1377
```

**계산 예시:**
- 학습 데이터: ~3,668개
- batch_size: 8
- 배치 수: 3,668 / 8 ≈ 459
- 총 스텝: 459 × 3 epochs = 1,377

### 2.5 Device 설정

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```

**역할:**
- GPU 사용 가능 여부 확인
- 모델을 해당 디바이스로 이동
- CPU vs GPU: 학습 시간 수 시간 vs 수 분

---

## 3단계: 학습 루프 (Training Loop)

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()  # ① 학습 모드 설정
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # ② 데이터를 GPU로
        outputs = model(**batch)                              # ③ Forward Pass
        loss = outputs.loss                                   # ④ Loss 추출
        loss.backward()                                       # ⑤ Backward Pass

        optimizer.step()                                      # ⑥ 가중치 업데이트
        lr_scheduler.step()                                   # ⑦ LR 조정
        optimizer.zero_grad()                                 # ⑧ Gradient 초기화
        progress_bar.update(1)
```

### 각 단계 상세 설명

| 단계 | 코드 | 역할 | 상세 설명 |
|------|------|------|----------|
| ① | `model.train()` | 학습 모드 활성화 | Dropout, BatchNorm 등이 학습용으로 동작 |
| ② | `v.to(device)` | 데이터 GPU 이동 | 모델과 데이터가 같은 디바이스에 있어야 연산 가능 |
| ③ | `model(**batch)` | Forward Pass | 입력 → 예측값(logits) 계산 |
| ④ | `outputs.loss` | Loss 추출 | Cross-Entropy Loss (labels가 있으면 자동 계산) |
| ⑤ | `loss.backward()` | Backward Pass | 각 파라미터에 대한 gradient 계산 |
| ⑥ | `optimizer.step()` | 가중치 업데이트 | gradient 방향으로 파라미터 조정 |
| ⑦ | `lr_scheduler.step()` | Learning Rate 조정 | 스케줄에 따라 LR 감소 |
| ⑧ | `optimizer.zero_grad()` | Gradient 초기화 | 다음 배치를 위해 gradient 리셋 |

### 학습 루프 순서 (중요!)

```
Forward → Backward → Optimizer Step → Scheduler Step → Zero Grad
   ↓          ↓           ↓              ↓              ↓
예측 계산   기울기 계산  가중치 수정   LR 조정      기울기 리셋
```

⚠️ **순서가 중요한 이유:**
- `zero_grad()`를 먼저 하면? → 계산된 gradient가 사라짐
- `optimizer.step()` 전에 `zero_grad()`하면? → 업데이트할 gradient가 없음

---

## 4단계: 평가 루프 (Evaluation Loop)

```python
import evaluate

metric = evaluate.load("glue", "mrpc")

model.eval()  # ① 평가 모드 설정
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():  # ② Gradient 계산 비활성화
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # ③ 예측 클래스 선택
    metric.add_batch(predictions=predictions, references=batch["labels"])  # ④ 결과 누적

metric.compute()  # ⑤ 최종 메트릭 계산
# {'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}
```

### 각 단계 상세 설명

| 단계 | 코드 | 역할 | 왜 필요한가? |
|------|------|------|-------------|
| ① | `model.eval()` | 평가 모드 | Dropout 비활성화, BatchNorm 고정 |
| ② | `torch.no_grad()` | Gradient 계산 OFF | 메모리 절약 + 속도 향상 (역전파 불필요) |
| ③ | `torch.argmax(logits, dim=-1)` | 가장 높은 점수의 클래스 선택 | logits → 실제 예측 레이블 |
| ④ | `metric.add_batch()` | 배치별 결과 누적 | 전체 데이터에 대한 메트릭 계산 준비 |
| ⑤ | `metric.compute()` | 최종 메트릭 계산 | Accuracy, F1 Score 반환 |

### `model.train()` vs `model.eval()` 비교

| 모드 | Dropout | BatchNorm | 용도 |
|------|---------|-----------|------|
| `train()` | 활성화 (랜덤 드롭) | 배치 통계 사용 | 학습 시 |
| `eval()` | 비활성화 (모든 뉴런 사용) | 학습된 통계 사용 | 평가/추론 시 |

---

## 5단계: 🤗 Accelerate로 분산 학습

### 기본 학습 루프 → Accelerate 적용

```python
from accelerate import Accelerator

accelerator = Accelerator()  # ① Accelerator 초기화

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

# ② 핵심: prepare()로 분산 학습 준비
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# 이후 스케줄러 설정
num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(...)

# 학습 루프 (변경점 표시)
model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        # batch = {k: v.to(device) ...} ← 삭제! Accelerate가 처리
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)  # ③ loss.backward() 대신

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

### 변경 사항 요약

| 기존 코드 | Accelerate 적용 | 이유 |
|-----------|----------------|------|
| `model.to(device)` | 삭제 | `prepare()`가 자동 처리 |
| `batch.to(device)` | 삭제 | `prepare()`가 자동 처리 |
| `loss.backward()` | `accelerator.backward(loss)` | 분산 환경에서 gradient 동기화 |
| - | `accelerator.prepare(...)` | 모든 객체를 분산 학습용으로 래핑 |

### 실행 방법

```bash
# 1. 분산 환경 설정
accelerate config

# 2. 학습 스크립트 실행
accelerate launch train.py
```

---

## 핵심 개념 정리

### 학습 루프 필수 순서
```
Forward Pass → Loss 계산 → Backward Pass → Optimizer Step → Scheduler Step → Zero Grad
```

### 평가 시 필수 설정
```python
model.eval()           # 평가 모드
with torch.no_grad():  # Gradient 계산 OFF
```

### 주요 컴포넌트 역할

| 컴포넌트 | 역할 |
|----------|------|
| **DataLoader** | 데이터를 배치 단위로 공급 |
| **Model** | 입력 → 예측 (+ labels 있으면 loss 계산) |
| **Optimizer** | Gradient 기반 가중치 업데이트 |
| **Scheduler** | Learning Rate 점진적 조정 |
| **Metric** | 모델 성능 측정 (Accuracy, F1 등) |

### Trainer API vs 직접 구현 비교

| 항목 | Trainer API | 직접 구현 |
|------|-------------|----------|
| 코드량 | 적음 | 많음 |
| 유연성 | 제한적 | 완전한 제어 |
| 커스텀 로직 | 콜백으로 제한 | 자유롭게 추가 |
| 디버깅 | 블랙박스 | 모든 과정 확인 가능 |
| 학습 목적 | △ | ✓ (동작 원리 이해) |

---

## 다음 학습 추천

1. **Learning Curves** (Chapter 3.5): 학습 곡선으로 과적합/과소적합 진단
2. **Mixed Precision Training**: `torch.cuda.amp`로 메모리 절약 + 속도 향상
3. **Gradient Accumulation**: 작은 배치로 큰 배치 효과 내기
4. **Gradient Clipping**: `clip_grad_norm_`으로 학습 안정화
