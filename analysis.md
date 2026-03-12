# 감정 조건부 EEG 합성 데이터 생성 및 생성 모델 비교

> **cGAN · cVAE · cDDPM 세 가지 조건부 생성 모델의 EEG 합성 품질 비교 연구**
> Python · PyTorch · SciPy · scikit-learn
> 개발 기간: 2025년 7월 ~ 2025년 8월 / 1인 프로젝트

---

## 1. 프로젝트 도입 배경 (Background)

### 문제 인식

뇌-컴퓨터 인터페이스(BCI) 및 감정 인식 연구에서 EEG 데이터는 핵심 자원이지만, 데이터 수집에 고가의 장비와 피험자 모집이 필요하고 세션당 획득 가능한 샘플 수가 제한적입니다. 특히 감정 레이블(Valence, Arousal)별로 균형 잡힌 데이터를 확보하기 어렵다는 구조적 한계가 존재합니다. 이러한 데이터 부족 문제를 생성 모델을 활용한 합성 데이터로 보완할 수 있는지가 본 프로젝트의 출발점이었습니다.

### 프로젝트 목표 및 나의 역할

* **세 가지 생성 패러다임의 체계적 비교**: 동일한 데이터와 평가 체계 하에서 cGAN, cVAE, cDDPM 세 가지 조건부 생성 모델을 구현하고, 각각의 합성 EEG 품질을 정량적으로 비교했습니다.
* **데이터 전처리 파이프라인 설계**: DREAMER 원시 EEG 데이터를 주파수 대역별로 분해하여 감정 관련 신경 활동 패턴을 극대화하는 전처리 파이프라인을 직접 설계했습니다.
* **분류기 기반 평가 프레임워크 구축**: 생성된 합성 EEG의 품질을 객관적으로 측정하기 위해, 사전 학습된 CNN 분류기를 평가 도구로 활용하는 프레임워크를 구축했습니다.

---

## 2. 아키텍처 설계

```bash
┌─────────────────────────────────────────────────────┐
│                DREAMER Dataset (.mat)               │
│            23명 피험자 × 18 영상 × 14채널 EEG            │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     data_processing/      │
         │  Butterworth 대역통과 필터   │
         │  5대역 × 14채널 = 70채널     │
         │  10초 슬라이딩 윈도우 세그멘트  │
         │  채널별 StandardScaler     │
         └─────────────┬─────────────┘
                       │
              (B, 70, 1280) 텐서
                       │
        ┌──────────────┼──────────────┐
        │              │              │
  ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
  │   cGAN    │  │   cVAE    │  │   cDDPM   │
  │           │  │           │  │           │
  │ Generator │  │ Encoder   │  │ UNet1D    │
  │ CBN1d     │  │ Decoder   │  │ FiLM      │
  │ Proj Disc │  │ β-anneal  │  │ Cosine    │
  │ Spec Norm │  │ Free-bits │  │ Schedule  │
  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
              합성 EEG (B, 70, 1280)
                       │
         ┌─────────────▼─────────────┐
         │    Pretrained Classifier  │
         │   4블록 CNN (Dual-Head)    │
         │  Valence / Arousal 분류    │
         │  F1, AUC 기반 품질 평가      │
         └───────────────────────────┘
```

**모듈 분리 원칙**

| 디렉토리 | 역할 |
|---|---|
| `data_processing/` | DREAMER 로딩, 대역통과 필터링, 슬라이딩 윈도우 세그멘테이션, Dataset 클래스 |
| `models/` | Classifier, cGAN(Generator+Discriminator), cVAE(Encoder+Decoder), cDDPM(UNet1D) |
| `train/` | 각 모델별 학습 루프, 체크포인트 관리, 학습 중 분류기 기반 평가 |
| `eval/` | 대량 합성 샘플 생성 및 정량 평가 (F1, AUC, 4클래스 정확도) |
| `utils/` | Butterworth 필터, 디바이스 자동 선택, 분류기 로더 |

---

## 3. 주요 기술적 문제 해결 과정 (Problem Solving)

### 3.1 원시 EEG에서 감정 관련 특징을 극대화하는 전처리 파이프라인 설계

**상황:**
DREAMER 데이터셋의 원시 14채널 EEG는 128Hz 샘플링 주파수의 시계열 데이터입니다. 그런데 감정과 관련된 신경 활동은 특정 주파수 대역에 집중되어 있습니다 — 예를 들어 Alpha(8-14Hz)는 이완 상태, Beta(14-30Hz)는 각성 상태, Gamma(30-50Hz)는 고차 인지 처리와 연관됩니다. 원시 신호를 그대로 사용하면 이러한 주파수별 감정 정보가 혼재되어 생성 모델이 감정 조건부 패턴을 학습하기 어려워집니다.

**해결:**
4차 Butterworth 대역통과 필터를 적용하여 각 채널을 5개 주파수 대역(Delta, Theta, Alpha, Beta, Gamma)으로 분해했습니다. `scipy.signal.filtfilt`를 사용한 영위상(zero-phase) 필터링으로 시간 왜곡 없이 순수한 주파수 성분만 추출했습니다.

```python
# utils/bandpass_filter.py
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs=128, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)  # zero-phase: 위상 왜곡 없음
```

14채널 × 5대역 = 70채널로 확장함으로써 모델이 주파수 대역별 감정 패턴을 독립적으로 학습할 수 있도록 했습니다. 또한 긴 연속 EEG를 10초(1280 샘플) 단위로 세그멘테이션하되, 2초(256 샘플) 오버랩을 적용하여 세그먼트 경계에서의 문맥 손실을 완화하고 학습 데이터를 확보했습니다.

```python
# data_processing/data_load.py — 세그멘테이션 + 대역 분해
segment_len = 1280   # 10초 @ 128Hz
overlap     = 256    # 2초 오버랩
hop         = segment_len - overlap  # 1024 샘플 단위 이동

for start in range(0, total_len - segment_len + 1, hop):
    segment = eeg_raw[start : start + segment_len]  # (1280, 14)
    bands = []
    for ch in range(14):
        for (lo, hi) in [(0.5,4), (4,8), (8,14), (14,30), (30,50)]:
            bands.append(bandpass_filter(segment[:, ch], lo, hi))
    # → (1280, 70) 텐서 생성
```

전처리 결과는 NPZ 파일로 캐싱하여 이후 실행 시 대역통과 필터링을 반복하지 않도록 했습니다. 또한 정규화 시 학습 데이터에 대해서만 `StandardScaler`를 fit하고, 검증 데이터에는 학습 통계를 그대로 적용하여 **데이터 누수를 방지**했습니다.

---

### 3.2 cGAN 학습 불안정성 제어 — Projection Discriminator와 Spectral Normalization

**상황:**
GAN은 Generator와 Discriminator의 적대적 학습 특성상 모드 붕괴(mode collapse)와 학습 불안정성에 취약합니다. EEG 신호는 이미지와 달리 70채널 × 1280 타임스텝의 고차원 시계열 데이터이기 때문에, Discriminator가 과도하게 강해지면 Generator의 그래디언트가 소실되어 학습이 조기에 붕괴하는 현상이 발생했습니다.

**해결:**
세 가지 안정화 기법을 결합하여 이 문제를 해결했습니다.

**첫째, Conditional BatchNorm(CBN)을 통한 Generator의 클래스 조건 주입입니다.** 일반적인 조건부 GAN에서는 노이즈 벡터에 레이블을 연결(concatenate)하지만, CBN은 각 정규화 레이어의 아핀 파라미터(γ, β)를 클래스 임베딩으로부터 동적으로 생성합니다. 초기 γ=1, β=0으로 설정하여 학습 초기에는 항등 변환에 가깝게 시작하고, 점진적으로 클래스별 변조를 학습합니다.

```python
# models/cGAN.py — Conditional BatchNorm1d
class CBN1d(nn.Module):
    def __init__(self, num_features, n_classes, emb_dim=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.fc_gamma = nn.Linear(emb_dim, num_features)
        self.fc_beta  = nn.Linear(emb_dim, num_features)
        # 항등 초기화: 학습 초기 안정성 확보
        nn.init.ones_(self.fc_gamma.weight.data[:, 0])
        nn.init.zeros_(self.fc_beta.weight.data)

    def forward(self, x, y):
        e = self.emb(y)
        gamma = self.fc_gamma(e).unsqueeze(-1)  # (B, C, 1)
        beta  = self.fc_beta(e).unsqueeze(-1)
        return gamma * self.bn(x) + beta
```

**둘째, Projection Discriminator로 클래스 정보를 효율적으로 활용했습니다.** 단순히 입력에 레이블을 연결하는 방식 대신, Discriminator의 최종 특징 벡터 h와 클래스 임베딩 e(y)의 내적(inner product)을 판별 스코어에 더하는 방식입니다. 이를 통해 "이 샘플이 진짜인가"와 "이 클래스에 부합하는가"를 동시에 판단합니다.

```python
# models/cGAN.py — Projection Discriminator 핵심 로직
class Disc1D(nn.Module):
    def forward(self, x, y):
        h = self.backbone(x)            # (B, 256) — 특징 추출
        h = self.gap(h).squeeze(-1)     # Global Average Pooling
        score = self.fc(h).squeeze(-1)  # 진위 판별 스코어
        proj  = (self.emb(y) * h).sum(dim=1)  # 클래스 프로젝션
        return score + proj             # 결합 출력
```

**셋째, Discriminator의 모든 Conv1d와 Linear 레이어에 Spectral Normalization을 적용하여 Lipschitz 상수를 1로 제약했습니다.** 이는 Discriminator의 급격한 기울기 변화를 방지하여 Generator가 안정적으로 그래디언트를 받을 수 있도록 합니다.

---

### 3.3 cVAE 사후 분포 붕괴(Posterior Collapse)와 진폭 소실 문제

**상황:**
조건부 VAE를 학습시켰을 때 두 가지 심각한 문제가 발생했습니다.

1. **사후 분포 붕괴**: KL divergence가 0에 수렴하면서 잠재 벡터 z가 사전 분포 N(0,I)에 고정되어, Decoder가 입력과 무관한 평균적인 신호만 출력하는 현상이 나타났습니다.
2. **채널별 진폭 소실**: 복원 손실(MSE/L1) 최적화 과정에서 생성 신호의 표준편차가 실제 EEG 대비 극도로 축소되어, 분류기가 합성 데이터를 전혀 인식하지 못하는 문제가 있었습니다.

**해결:**
세 가지 보완 메커니즘을 설계하여 적용했습니다.

**Beta Annealing** — KL 항의 가중치를 0에서 시작하여 10 에폭에 걸쳐 선형적으로 1.0까지 증가시킵니다. 학습 초기에는 복원 품질에 집중하고, 점진적으로 잠재 공간의 정규화를 강화합니다.

```python
# train/train_cVAE.py — Beta Annealing 스케줄
beta = beta0 + (beta1 - beta0) * min(1.0, epoch / warm)
# epoch=0 → beta=0.0 (KL 무시, 복원에 집중)
# epoch=10 → beta=1.0 (전체 ELBO 최적화)
```

**Free-bits 정규화** — 잠재 벡터의 각 차원이 최소한 0.2 nats 이상의 정보를 인코딩하도록 KL을 차원별로 클램핑합니다. 이를 통해 Encoder가 일부 차원만 활용하고 나머지를 죽이는 현상을 방지합니다.

```python
# train/train_cVAE.py — Free-bits: 차원별 KL 하한 보장
kl_per_dim = -0.5 * (1 + logv - mu.pow(2) - logv.exp())  # (B, z_dim)
kl_clamped = torch.clamp(kl_per_dim - fb_nats, min=0.0)   # 최소 fb_nats 이하는 패널티 면제
loss_kl = kl_clamped.sum(dim=1).mean()
```

**채널별 Scale Penalty** — 가장 핵심적인 기여로, 생성 신호와 실제 신호의 채널별 표준편차 차이를 직접 페널티로 부여합니다. 이는 VAE가 복원 손실 최소화를 위해 안전하게 저진폭 신호를 출력하려는 경향을 직접적으로 억제합니다.

```python
# train/train_cVAE.py — 채널별 표준편차 매칭 페널티
std_real = xb.std(dim=(0, 2), unbiased=False)   # 실제 EEG의 채널별 std (70,)
std_gen  = xhat.std(dim=(0, 2), unbiased=False)  # 생성 EEG의 채널별 std (70,)
loss_scale = F.l1_loss(std_gen, std_real)

# 최종 손실: 복원 + β·KL + α·Scale
loss = loss_recon + beta * loss_kl + alpha_scale * loss_scale
```

학습 과정을 실시간으로 모니터링하기 위해, 매 에폭마다 잠재 공간 통계(μ 절대 평균, logσ² 평균)와 실제/생성 신호의 표준편차를 출력하는 디버그 로그를 삽입했습니다. 이를 통해 사후 분포 붕괴나 진폭 소실의 조기 징후를 포착할 수 있었습니다.

---

### 3.4 cDDPM의 1D 시계열 적응 — UNet1D와 FiLM 조건 부여

**상황:**
Diffusion 모델(DDPM)은 이미지 생성에서 뛰어난 성능을 보여주었지만, 기존 구현체는 2D 이미지(UNet2D)에 최적화되어 있습니다. 70채널 × 1280 타임스텝의 1D EEG 시계열에 Diffusion 프레임워크를 적용하려면 아키텍처 전반을 재설계해야 했습니다. 또한 타임스텝 t와 감정 클래스 y라는 **두 가지 이질적인 조건 정보**를 동시에 주입하는 방법이 필요했습니다.

**해결:**
UNet2D를 1D로 전환하되, 모든 Conv2d를 Conv1d로 교체하고, 다운/업샘플링 비율을 EEG 시퀀스 길이(1280)에 맞춰 3단계(1280→640→320→160)로 설계했습니다.

조건 부여에는 **FiLM(Feature-wise Linear Modulation)** 기법을 채택했습니다. 타임스텝 t는 사인파 위치 인코딩으로, 클래스 y는 학습 가능한 임베딩으로 각각 벡터화한 후, 두 조건의 scale/shift를 가산적으로 결합하여 각 ResBlock의 중간 특징 맵을 변조합니다.

```python
# models/cDDPM.py — FiLM 조건 변조가 적용된 ResBlock1D
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, c_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        # 타임스텝 → scale, shift
        self.t_proj = nn.Linear(t_dim, out_ch * 2)
        # 클래스 → scale, shift
        self.c_proj = nn.Linear(c_dim, out_ch * 2)

    def forward(self, x, t_emb, c_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # FiLM: 두 조건의 변조를 가산적으로 결합
        ts, tb = self.t_proj(t_emb).chunk(2, dim=1)  # time scale, shift
        cs, cb = self.c_proj(c_emb).chunk(2, dim=1)  # class scale, shift
        scale = (ts + cs).unsqueeze(-1)
        shift = (tb + cb).unsqueeze(-1)
        h = h * (1 + scale) + shift  # Feature-wise 변조
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
```

노이즈 스케줄은 Cosine Schedule을 채택했습니다. 선형 스케줄 대비 초반과 후반의 노이즈 변화가 완만하여, EEG처럼 미세한 진폭 변화가 중요한 신호에서 디노이징 품질이 더 안정적이었습니다.

```python
# models/cDDPM.py — Cosine Noise Schedule
def cosine_schedule(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, max=0.999)
```

또한 **Classifier-Free Guidance(CFG)** 를 위해 null 임베딩(영 벡터)을 설계하여, 추론 시 조건부/비조건부 예측의 가중 결합으로 클래스 일관성을 강화할 수 있도록 했습니다.

---

### 3.5 생성 품질 평가 전략 — 분류기 기반 정량 평가 프레임워크

**상황:**
이미지 생성에서는 FID(Fréchet Inception Distance)나 IS(Inception Score)와 같은 표준 평가 지표가 존재하지만, EEG 합성 데이터에는 이러한 범용 지표가 없습니다. 생성된 EEG가 "감정 조건을 제대로 반영하고 있는가"를 객관적으로 측정할 방법이 필요했습니다.

**해결:**
사전 학습된 CNN 분류기를 평가 도구로 활용하는 프레임워크를 설계했습니다. 핵심 아이디어는 **"실제 EEG로 학습된 분류기가 합성 EEG에서도 올바른 감정 클래스를 인식할 수 있다면, 그 합성 데이터는 감정 조건부 패턴을 성공적으로 재현한 것"** 이라는 전제입니다.

```python
# eval/eval_cGAN.py — 분류기 기반 대량 평가
classifier.eval()
generator.eval()

all_preds_val, all_preds_aro = [], []
all_true_val, all_true_aro = [], []

for _ in range(num_samples // batch_size):
    z = torch.randn(batch_size, z_dim, device=device)
    y = torch.randint(0, 4, (batch_size,), device=device)
    with torch.no_grad():
        fake = generator(z, y)
        logit_val, logit_aro = classifier(fake)

    # 4클래스 레이블에서 Valence/Arousal 이진 레이블 추출
    y_val_true = (y >> 1) & 1   # 상위 비트: Valence
    y_aro_true = y & 1          # 하위 비트: Arousal
    # → Accuracy, Precision, Recall, F1, AUC 산출
```

평가 지표는 두 레벨로 설계했습니다:
- **차원별 지표**: Valence와 Arousal 각각에 대한 F1, AUC
- **결합 4클래스 지표**: `pred_4 = 2×pred_val + pred_aro`로 결합한 4클래스 정확도 및 macro F1

학습 중에도 매 에폭마다 클래스당 128개(총 512개)의 합성 샘플을 생성하여 분류기로 평가하고, **평균 F1 향상 + 양 차원 AUC 최소 임계값 충족**이라는 이중 조건 하에서만 체크포인트를 저장하는 전략을 적용했습니다.

```python
# train/train_cGAN.py — 이중 조건 체크포인트 저장
avg_f1 = (f1_val + f1_aro) / 2
if avg_f1 > best_f1 and auc_val > min_auc and auc_aro > min_auc:
    best_f1 = avg_f1
    torch.save({
        'epoch': ep, 'G': G.state_dict(), 'D': D.state_dict(),
        'avg_f1': avg_f1, 'z_dim': z_dim
    }, 'experiments/c_gan_best.pth')
```

cVAE의 경우 한 단계 더 나아가 **이중 체크포인트 전략**을 적용했습니다. 순수 VAE 손실(복원+KL) 기준 최적 모델과, 분류기 평가 기준 최적 모델을 별도로 저장하여, 학습 목적함수와 실제 하류 과제 성능 간의 괴리를 분석할 수 있도록 했습니다.

---

### 3.6 다중 디바이스 호환성과 MPS 환경 대응

**상황:**
Apple Silicon(M 시리즈) Mac에서 개발하면서 MPS(Metal Performance Shaders) 백엔드를 사용했는데, PyTorch의 MPS 지원이 CUDA 대비 불완전하여 특정 연산에서 오류가 발생했습니다. 특히 `ConvTranspose1d`와 일부 업샘플링 연산에서 MPS 호환성 이슈가 있었습니다.

**해결:**
디바이스 자동 선택 유틸리티를 구현하여 CUDA → MPS → CPU 순서로 폴백하도록 했습니다.

```python
# utils/device_selection.py
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Generator의 업샘플링에서 `ConvTranspose1d` 대신 **Nearest-neighbor interpolation + Conv1d** 조합을 채택했습니다. 이 방식은 MPS에서의 호환성 문제를 회피하는 동시에, 전치 합성곱에서 발생할 수 있는 체커보드 아티팩트(checkerboard artifacts)를 원천적으로 방지하는 이점이 있었습니다.

또한 DDPM의 ResBlock에서 BatchNorm 대신 **GroupNorm(8그룹)** 을 사용하여, 작은 배치 크기에서도 정규화가 안정적으로 작동하고 디바이스에 무관한 동작을 보장했습니다.

---

## 4. 핵심 성과

1. **세 가지 생성 패러다임의 체계적 비교 프레임워크 확립**
   - 동일한 데이터 전처리, 동일한 평가 체계 하에서 cGAN, cVAE, cDDPM을 공정하게 비교할 수 있는 실험 환경을 구축했습니다. cGAN이 4클래스 정확도 75.02%, Valence AUC 0.9128로 가장 높은 성능을 달성하여, EEG 조건부 생성에서의 적대적 학습의 효과를 실증했습니다.

2. **도메인 특화 전처리 파이프라인으로 생성 품질 향상**
   - 원시 14채널 EEG를 5대역 × 14채널 = 70채널로 분해하여, 생성 모델이 주파수 대역별 감정 패턴을 독립적으로 학습할 수 있는 표현을 확보했습니다. Butterworth 영위상 필터링, NPZ 캐싱, 데이터 누수 방지 정규화를 포함한 견고한 전처리 파이프라인을 설계했습니다.

3. **VAE 학습 안정화를 위한 다중 정규화 기법 설계**
   - Beta Annealing, Free-bits, 채널별 Scale Penalty라는 세 가지 보완 메커니즘을 조합하여 사후 분포 붕괴와 진폭 소실 문제를 해결했습니다. 특히 채널별 표준편차 매칭이라는 직관적이면서도 효과적인 정규화 항을 직접 설계하여 적용했습니다.

4. **분류기 기반 평가 및 이중 체크포인트 전략**
   - EEG 도메인에 적합한 생성 품질 평가 프레임워크를 구축했습니다. 순수 손실 기반과 하류 분류기 기반의 이중 체크포인트 저장을 통해, 학습 목적함수와 실제 과제 성능 간의 관계를 분석할 수 있는 실험 환경을 마련했습니다.
