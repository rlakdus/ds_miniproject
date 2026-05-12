# ds_miniproject

Multimodal prototype for CLIP and BLIP style architectures.

## 프로젝트 구조

- `config.py`
  - 전체 모델 설정값을 담는 `MedConfig`
  - 텍스트/이미지 차원과 cross-attention, CLIP projection 크기를 정의
- `attention.py`
  - self-attention / cross-attention 구현
- `ffn.py`
  - transformer feed-forward block
- `bert_layer.py`
  - BertLayer: self-attn + optional cross-attn + FFN
- `bert_model.py`
  - 텍스트 모듈의 encoder/decoder 구현
- `config.py`
  - 전체 모델 설정값을 담는 `MedConfig`
  - 텍스트/이미지 차원과 cross-attention, CLIP projection 크기를 정의
- `attention.py`
  - self-attention / cross-attention 구현
- `ffn.py`
  - transformer feed-forward block
- `bert_layer.py`
  - BertLayer: self-attn + optional cross-attn + FFN
- `bert_model.py`
  - 텍스트 모듈의 encoder/decoder 구현
- `clip/clip_model.py`
  - CLIP 스타일 텍스트/이미지 임베딩과 contrastive logits
- `blip/blip_model.py`
  - BLIP 스타일 cross-attention wrapper
- `losses.py`
  - ITC/CLIP contrastive loss, ITM classification loss, LM cross-entropy loss
- `training_utils.py`
  - optimizer 생성과 CLIP / BLIP 학습 step 예시
- `tests/test_med.py`
  - 기존 BLIP MED 텍스트 shape 검증
- `tests/test_models.py`
  - CLIP / BLIP forward shape 검증
- `tests/test_losses.py`
  - loss 함수의 동작 검증

## 설계 목표

이 저장소는 두 가지 멀티모달 흐름을 분리해서 다룹니다:

1. CLIP
   - 텍스트 인코더와 이미지 인코더를 별도 처리
   - 텍스트/이미지 임베딩을 projection space로 매핑
   - contrastive logits를 출력
2. BLIP
   - 텍스트 모듈에 이미지 feature를 cross-attention으로 연결
   - ITM/ITC 및 비전-언어 교차 모드를 지원

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy
```

## 테스트 실행

기존 BLIP MED shape 검증:

```bash
python tests/test_med.py
```

CLIP / BLIP 모델 forward 검증:

```bash
python tests/test_models.py
```

loss 함수 검증:

```bash
python tests/test_losses.py
```

## 데이터 흐름

### 텍스트 입력

- `input_ids`: `[B, T]`
- `attention_mask`: `[B, T]`

토크나이저에서 생성된 입력을 그대로 사용합니다.

### 이미지 입력

- 외부 이미지 백본(ViT, ResNet 등)으로 feature를 추출
- `encoder_hidden_states`: `[B, I, encoder_width]`
- `encoder_attention_mask`: `[B, I]`

현재 코드는 이미지 백본을 포함하지 않으므로, `image_features`는 외부에서 준비해야 합니다.

## 모델별 사용 예

### CLIP

```python
from config import MedConfig
from clip.clip_model import ClipModel

config = MedConfig(
    num_hidden_layers=2,
    intermediate_size=1024,
    encoder_width=768,
    projection_dim=512,
)
model = ClipModel(config)

outputs = model(
    input_ids,
    attention_mask,
    image_features,
)

text_embeds = outputs['text_embeddings']
image_embeds = outputs['image_embeddings']
logits = outputs['logits_per_text']
```

### BLIP

```python
from config import MedConfig
from blip.blip_model import BlipModel

config = MedConfig(
    num_hidden_layers=2,
    intermediate_size=1024,
    encoder_width=768,
    add_cross_attention=True,
)
model = BlipModel(config, is_decoder=False)

outputs = model(
    input_ids,
    attention_mask,
    encoder_hidden_states=image_features,
    encoder_attention_mask=image_mask,
)

last_hidden = outputs['last_hidden_state']
pooler_output = outputs['pooler_output']
```

## 실제 데이터 적용 포인트

1. 텍스트 토크나이저로 `input_ids`, `attention_mask` 생성
2. 이미지 백본으로 `image_features` 생성
3. CLIP은 `image_features`를 projection하여 임베딩 비교
4. BLIP은 `image_features`를 cross-attention 입력으로 전달

## Loss / Optimizer 파이프라인

- `losses.py`
  - `clip_contrastive_loss`: CLIP / ITC contrastive loss
  - `itm_loss`: ITM binary/2-class 분류 loss
  - `lm_loss`: LM cross entropy loss
- `training_utils.py`
  - `make_optimizer`: AdamW optimizer 생성
  - `clip_train_step`: CLIP contrastive update step 예시
  - `blip_itm_train_step`: BLIP text+classifier ITM update step 예시

### CLIP 학습 예

```python
from config import MedConfig
from clip_model import ClipModel
from training_utils import make_optimizer, clip_train_step

config = MedConfig(projection_dim=512)
model = ClipModel(config)
optimizer = make_optimizer(model, lr=1e-4)
loss = clip_train_step(model, input_ids, attention_mask, image_features, optimizer)
```

### BLIP ITM 학습 예

```python
import torch
from config import MedConfig
from blip_model import BlipModel
from training_utils import make_optimizer, blip_itm_train_step

config = MedConfig(add_cross_attention=True)
text_model = BlipModel(config, is_decoder=False)
classifier = torch.nn.Linear(config.hidden_size, 2)
optimizer = make_optimizer(torch.nn.Sequential(text_model, classifier), lr=1e-4)
labels = torch.tensor([0, 1])
loss = blip_itm_train_step(
    text_model,
    classifier,
    input_ids,
    attention_mask,
    image_features,
    image_mask,
    labels,
    optimizer,
)
```

## 참고

- `clip_model.py`와 `blip_model.py`는 현재 구조화된 wrapper 형태입니다.
- `losses.py`와 `training_utils.py`는 기본 손실/트레이닝 스텝 예시를 제공합니다.
- 실제 학습 파이프라인에는 tokenizer, 이미지 백본, 데이터 로더가 추가로 필요합니다.
- `test_models.py`는 두 모델의 기본 forward shape를 검증하는 데 사용됩니다.

