# EEG Emotion Generation & Comparison

감정 조건부 EEG 신호 합성을 위한 생성 모델 비교 — DREAMER 데이터셋을 활용하여 **cGAN**, **cVAE**, **cDDPM** 세 가지 생성 모델의 성능을 평가합니다.

## 개요

감정 레이블(Valence × Arousal → 4 클래스)을 조건으로 합성 EEG 데이터를 생성하고, 사전 학습된 분류기를 통해 세 가지 생성 모델의 품질을 비교합니다.

```
실제 EEG (DREAMER) → 전처리 → 생성 모델 학습 → 합성 EEG 생성 → 분류기 평가
```

## 데이터셋

**DREAMER** — 감정 유발 자극 하에서 측정된 14채널 EEG (128 Hz 샘플링).

| 처리 단계 | 세부 사항 |
|---|---|
| 세그멘테이션 | 10초 윈도우 (1280 샘플), 2초 오버랩 |
| 대역 통과 필터링 | 5개 주파수 대역 × 14채널 = **70채널** |
| 주파수 대역 | Delta (0.5–4), Theta (4–8), Alpha (8–14), Beta (14–30), Gamma (30–50 Hz) |
| 레이블 | Valence (이진) × Arousal (이진) → 4클래스: `y = 2V + A` |
| 정규화 | 채널별 StandardScaler |

**최종 입력 형상**: `(N, 70, 1280)`

## 모델

### Classifier (기준 CNN)

Valence + Arousal 이중 출력 헤드를 가진 4블록 CNN.

- Conv1d 블록 + AvgPool, BatchNorm, SpatialDropout
- Global Average Pooling → 공유 FC 임베딩 → 두 개의 sigmoid 출력 헤드
- 손실 함수: BCEWithLogitsLoss

### cGAN

Projection Discriminator 기반 조건부 GAN.

- **Generator**: 노이즈 `z(128)` → Conditional BatchNorm(CBN)을 적용한 4단계 업샘플링
- **Discriminator**: Spectral Normalization + 클래스 프로젝션을 적용한 5블록 다운샘플링
- 손실 함수: BCE 적대적 손실

### cVAE

Beta annealing을 적용한 조건부 VAE.

- **Encoder**: 클래스 임베딩을 입력에 결합 → 4개 conv 레이어 → μ, logσ² (잠재 차원=128)
- **Decoder**: 잠재 벡터 + 클래스 임베딩 → 4개 transposed conv 레이어 → 복원
- 손실 함수: SmoothL1 (복원) + KL divergence (free-bits) + 채널별 스케일 페널티
- 사후 분포 붕괴 방지를 위해 10 에폭에 걸친 beta warm-up

### cDDPM

조건부 Denoising Diffusion Probabilistic Model.

- **U-Net1D**: skip connection을 가진 4레벨 인코더-디코더
- **조건 부여**: FiLM 변조 (사인파 시간 임베딩 + 클래스 임베딩)
- **스케줄**: Cosine noise schedule, T=1000 타임스텝
- 손실 함수: 예측 노이즈에 대한 MSE
- 추론 시 Classifier-Free Guidance 지원

## 프로젝트 구조

```
├── models/
│   ├── classifier.py      # 4블록 CNN 분류기
│   ├── cGAN.py             # Generator + Projection Discriminator
│   ├── cVAE.py             # Encoder + Decoder (beta annealing)
│   └── cDDPM.py            # U-Net1D + cosine diffusion schedule
├── train/
│   ├── train_classifier.py
│   ├── train_cGAN.py
│   ├── train_cVAE.py
│   └── train_cDDPM.py
├── eval/
│   ├── eval_cGAN.py
│   └── eval_cVAE.py
├── data_processing/
│   ├── data_load.py        # DREAMER 로딩 + 대역 통과 필터링
│   ├── data_object.py      # PyTorch Dataset 클래스
│   └── data_spilt.py       # 학습/검증 계층적 분할
├── utils/
│   ├── bandpass_filter.py   # Butterworth 대역 통과 필터 (5개 대역)
│   ├── device_selection.py  # 자동 디바이스 선택 (CUDA/MPS/CPU)
│   └── load_classifier.py  # 사전 학습 분류기 로더
├── notebooks/               # 실험 로그 및 결과
│   ├── 20250815_final_classifier.ipynb
│   ├── 20250816_final.cGAN.ipynb
│   ├── 20250816_final_cVAE.ipynb
│   ├── 20250816_final_Diffusion.ipynb
│   └── ...
└── data/                    # (gitignored) DREAMER.mat + 전처리된 .npz
```

## 결과

평가 방식: 클래스당 128개 합성 샘플 생성 → 사전 학습된 CNN으로 분류 → 지표 산출.

| 모델 | 4클래스 정확도 | 4클래스 F1 (macro) | Valence F1 | Arousal F1 | Valence AUC | Arousal AUC |
|---|---|---|---|---|---|---|
| **cGAN** | **0.7502** | **0.6684** | **0.8001** | **1.0000** | **0.9128** | **1.0000** |
| cVAE | 0.2500 | 0.1000 | 0.6667 | 0.6667 | 0.5403 | 0.5639 |
| cDDPM | — | — | — | — | — | — |

## 사용법

### 1. 분류기 학습

```bash
python -m train.train_classifier --epochs 200
```

### 2. 생성 모델 학습

```bash
python -m train.train_cGAN --epochs 50
python -m train.train_cVAE --epochs 30
python -m train.train_cDDPM --epochs 100
```

### 3. 평가

```bash
python -m eval.eval_cGAN
python -m eval.eval_cVAE
```

## 주요 기법

- **대역 통과 분해**: 5개 주파수 대역 추출로 감정 관련 EEG 패턴 포착
- **Conditional BatchNorm**: Generator에서 클래스 조건부 아핀 변조
- **Projection Discriminator**: Spectral Normalization + 클래스 프로젝션 기반 판별기
- **Beta Annealing + Free-bits**: VAE 사후 분포 붕괴 방지
- **FiLM Conditioning**: Diffusion 모델을 위한 feature-wise 선형 변조
- **Cosine Noise Schedule**: 안정적 학습을 위한 코사인 노이즈 스케줄
