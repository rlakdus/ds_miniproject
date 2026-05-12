# ds_miniproject

BLIP MED 스타일 텍스트 모듈 구현입니다.

## 구성

- `config.py` : `MedConfig` 설정값
- `attention.py` : self-attention, cross-attention 구현
- `ffn.py` : intermediate + output FFN 블록
- `bert_layer.py` : BERT 레이어 (self-attn + optional cross-attn + FFN)
- `bert_model.py` : 텍스트 모듈 전체, encoder/decoder 모드 지원
- `test_med.py` : 더미 입력 단위 테스트

## 설치

1. 가상환경 생성/활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. 의존성 설치

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy
```

## 테스트 실행

```bash
python test_med.py
```

정상 실행되면 shape 검증 테스트가 모두 통과합니다.

## 실제 데이터 사용 방법

현재 코드베이스는 텍스트 모듈(`BertModel`) 중심입니다. 이미지 입력은 외부 이미지 인코더에서 추출한 feature를 `encoder_hidden_states`로 전달해야 합니다.

### 1) 텍스트 데이터 준비

- 텍스트를 토크나이저로 `input_ids`와 `attention_mask`로 변환
- `input_ids` shape: `[B, T]`
- `attention_mask` shape: `[B, T]`

예:

```python
input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')['input_ids']
attention_mask = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')['attention_mask']
```

### 2) 이미지 데이터 준비

- ViT / 이미지 백본으로 이미지 특징을 추출
- `encoder_hidden_states` shape: `[B, I, encoder_width]`
- `encoder_attention_mask` shape: `[B, I]`

예:

```python
image_features = image_encoder(images)  # [B, I, encoder_width]
image_mask = torch.ones(B, I, dtype=torch.long)
```

### 3) 모델 선택

- ITC/ITM (encoder-only): `BertModel(config, is_decoder=False)`
- LM (decoder): `BertModel(config, is_decoder=True)`

`config.add_cross_attention=True`이면 cross-attention이 활성화됩니다.

### 4) 포워드 예시

```python
from config import MedConfig
from bert_model import BertModel

config = MedConfig(num_hidden_layers=2, intermediate_size=1024, add_cross_attention=True)
model = BertModel(config, is_decoder=False)

outputs = model(
    input_ids,
    attention_mask,
    encoder_hidden_states=image_features,
    encoder_attention_mask=image_mask,
)

last_hidden = outputs['last_hidden_state']
pooler_out = outputs['pooler_output']
```

### 5) 학습/평가

- ITC/ITM: `pooler_output`를 사용해 분류나 유사도 계산
- LM: `last_hidden_state`를 LM 헤드에 연결하여 다음 토큰 예측

## 주의

- 현 코드는 이미지 인코더를 포함하지 않음
- 실제 데이터 처리에는 tokenizer/이미지 백본/데이터셋 클래스가 추가로 필요
- `test_med.py`는 현재 단순 shape 검증용 더미 테스트임
