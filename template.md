# 관광 식당 메뉴판 캡처 및 AI 번역 자동화 파이프라인

> **다국어 관광객을 위한 End-to-End 메뉴판 분석/번역 서비스**  
> Python · FastAPI · Docker · Flutter · Next.js · NextUI · Surya OCR · OpenNMT
> 개발 기간: 2024년 4월 ~ 2024년 6월 / 2인 프로젝트 (팀원 하차로 **1인 주도 완수**)

---

## 1. 프로젝트 도입 배경 (Background)

### 문제 인식
외국인 관광객이 한국의 로컬 식당을 방문했을 때 겪는 가장 큰 장벽 중 하나는 **메뉴판 해독**입니다. 특히 한국의 식당 메뉴판은 손글씨, 세로쓰기, 독특한 폰트, 다양한 배경 노이즈가 기입되어 있는 경우가 많아 기존의 상용 번역 앱이나 범용 OCR 앱으로는 인식률이 현저히 떨어집니다. 더불어, "제육볶음" 등 고유 한식 단어가 문맥 없이 오역되는 문제가 지적되어 왔습니다.

### 프로젝트 목표 및 나의 역할

* **전체 파이프라인 1인 주도**: 2인 프로젝트로 시작했으나 팀원의 중도 하차로 인해 AI 모델 서빙, 파이프라인 백엔드 구축, 프론트엔드(Web/App) 개발까지 전 과정을 1인 주도로 완수했습니다.
* **MLOps 환경 구축에 집중**: AI 모델 자체의 학습보다는, AI Hub에서 제공하는 Pretrained Inference 스크립트를 **실제 서비스에 안정적으로 연동하는 MLOps 파이프라인 구축**에 역량을 집중했습니다.
* **도메인 특화 AI 모델 편입**: 한국 AI 허브에 공개된 **'관광 음식메뉴판 데이터' 기반 사전학습 알고리즘**을 차용해, 식당 메뉴 특화 텍스트 검출/인식(OCR)과 번역(NMT)을 자동화했습니다.

---

## 2. 아키텍처 설계

```bash
┌────────────────────────┐
│   Flutter App          │  카메라 촬영 · 갤러리 접근
│   (image_picker)       │  권한 관리 (permission_handler)
└───────────┬────────────┘
            │ HTTP POST /upload  (이미지 바이너리)
┌───────────▼────────────┐
│   FastAPI Server       │  오케스트레이터 · WebSocket 진행률
│   (main.py)            │  비동기 이벤트 루프 (asyncio)
│   (docker_utils.py)    │  docker cp / docker exec 래핑
│   (ocr_crop.py)        │  Surya 모델 기반 Bounding Box 검출 · 크롭
└──┬──────────┬──────────┘
   │          │
  docker cp/exec
┌──▼────────┐  ┌──▼──────────┐
│ OCR       │  │ Translate   │
│ Container │  │ Container   │
│ TPS-VGG   │  │ OpenNMT     │
│ BiLSTM    │  │ ko→en       │
│ Attn      │  │ en_model.pt │
└───────────┘  └─────────────┘
            │ JSON (Base64 이미지 + 번역 텍스트)
            │ POST /api/receive
┌───────────▼────────────┐
│   Next.js 14 Web       │  반응형 결과 그리드 (NextUI)
│   DataContext 전역 상태  │  원본 · 검출 · 크롭 이미지 렌더링
└───────────┬────────────┘
            │ webview_flutter 로드
┌───────────▼────────────┐
│   Flutter App (결과)    │
└────────────────────────┘
```

**레이어 분리 원칙**
- `docker_utils.py` — Docker CLI 래핑 유틸리티 (`docker_cp`, `docker_exec`를 `asyncio`로 비동기화)
- `ocr_crop.py` — Surya 모델 기반 텍스트 검출 및 Bounding Box 크롭 로직
- `main.py` — FastAPI 엔드포인트, WebSocket 진행률 관리, 멀티 컨테이너 오케스트레이션
- `context/DataContext.jsx` — Next.js 전역 상태 관리 (Base64 이미지 + 번역 데이터)
- Flutter — 하드웨어 네이티브 제어 전담 (카메라, 갤러리, WebView 브릿지)

---

## 3. 주요 기술적 문제 해결 과정 (Problem Solving)

### 3.1 AI Inference 스크립트의 서비스화 및 Docker 통신

**상황:**  
머신러닝 환경과 Docker를 처음 접하는 상태에서, 격리된 컨테이너 내부의 무거운 AI 모델(OCR, OpenNMT)과 백엔드 서버 간의 데이터 입출력을 제어해야 했습니다. 컨테이너 내부의 파일 시스템 권한, 셸 명령어 비동기 제어 등에서 큰 삽질을 겪었습니다.

**해결:**  
- `subprocess.run`을 직접 호출하면 FastAPI의 이벤트 루프가 블로킹되므로, `asyncio.get_event_loop().run_in_executor()`로 감싸 **논블로킹 Docker CLI 유틸리티**(`docker_utils.py`)를 설계했습니다.

```python
# docker_utils.py - subprocess를 asyncio 이벤트 루프에서 논블로킹으로 실행
async def docker_exec(container: str, command: str) -> subprocess.CompletedProcess:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, docker_exec_sync, container, command)
```

- 하나의 `/upload` 엔드포인트 안에서 **10단계 파이프라인**을 순차적으로 실행했습니다:

```bash
① 이미지 업로드 수신 (aiofiles)
② Surya 모델로 텍스트 영역 검출 및 Bounding Box 크롭 (ocr_crop.py)
③ 크롭 이미지를 ZIP 압축
④ ZIP → OCR 컨테이너로 docker cp
⑤ 컨테이너 내부에서 unzip
⑥ OCR 추론 실행 (TPS-VGG-BiLSTM-Attn)
⑦ 결과 엑셀(.xlsx) → 호스트로 docker cp
⑧ 엑셀 → 번역 컨테이너로 docker cp
⑨ OpenNMT 번역 실행 (ko→en, en_model.pt)
⑩ 최종 결과 수합 → Base64 인코딩 → Next.js POST
```

```python
# main.py - 10단계 파이프라인의 핵심 흐름
copy_result = await docker_cp(zip_file_path, f"{container_ocr_name}:{container_ocr_zip_path}")
unzip_result = await docker_exec(container_ocr_name, unzip_command)

exec_command = (
    f"python {script_path} "
    f"--Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM "
    f"--Prediction Attn --image_folder {container_ocr_unzip_dir} "
    f"--saved_model saved_models/TPS-VGG-BiLSTM-Attn-final/best_accuracy.pth "
    f"--batch_max_length 50 --workers 16 --batch_size 64 --imgH 64 --imgW 300"
)
exec_result = await docker_exec(container_ocr_name, exec_command)
```

> 각 단계마다 `returncode`를 검사하여 실패 시 즉시 `JSONResponse(500)`을 반환하는 **단계별 에러 핸들링 체계**를 구축했습니다. 파이프라인의 어느 지점에서 실패했는지 프론트엔드에 정확히 전달할 수 있습니다.

---

### 3.2 텍스트 영역 검출 및 고해상도 이미지 크롭 좌표 처리

**상황:**  
모바일에서 넘어온 고해상도 원본 이미지에서 수십 개의 텍스트 영역을 정확히 검출하고, 각각을 개별 이미지로 잘라낸 뒤 번역 결과와 일대일 매핑해야 했습니다.

**해결:**  
- 텍스트 검출에는 **Surya** 라이브러리의 `batch_text_detection` 모델(SegFormer 기반)을 활용했습니다. Bounding Box 좌표(`[x1, y1, x2, y2]`)를 추출한 뒤 PIL의 `Image.crop()`으로 개별 텍스트 이미지를 잘라냈습니다.
- 검출 시각화를 위해 원본 이미지에 빨간 사각형과 인덱스 번호를 오버레이하는 `detected_image`도 함께 생성하여, 프론트엔드에서 "어디서 어떤 텍스트를 검출했는지" 시각적으로 확인할 수 있도록 했습니다.

```python
# ocr_crop.py - Surya SegFormer 모델로 텍스트 영역을 검출하고 PIL로 크롭
model, processor = load_model(), load_processor()
predictions = batch_text_detection([image], model, processor)

for i, result in enumerate(predictions):
    for j, box in enumerate(result.bboxes):
        bbox = box.bbox  # [x1, y1, x2, y2]
        draw.rectangle(bbox, outline='red', width=2)
        draw.text((bbox[0]-10, bbox[1]-10), str(j), fill='blue', font=font)
        cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        cropped_image.save(os.path.join(save_dir, f'cropped_{i}_{j}.jpg'))
```

---

### 3.3 데이터 통신 최적화 — Base64 JSON 직렬화

**상황:**  
수십 장의 크롭 이미지 + 원본 이미지 + 검출 오버레이 이미지를 프론트엔드에 전달해야 했습니다. 이미지 파일을 정적 저장 후 URL로 서빙하면 파일 관리 복잡도와 I/O 병목이 우려되었습니다.

**해결:**  
- 모든 이미지를 **Base64로 즉시 인코딩**하여 단일 JSON 페이로드 안에 번역 텍스트와 함께 패키징했습니다.
- Next.js의 `/api/receive` 라우트에서 이 JSON을 수신하고, `DataContext`(React Context API)를 통해 전역 상태로 관리하여, `/result` 페이지에서 그리드 렌더링에 바로 활용했습니다.

```python
# main.py - 최종 응답 데이터 구조 (원본 · 검출 · 크롭 · 번역 텍스트를 하나의 JSON으로 통합)
df = pd.read_excel(excel_path)
text_data = {}
for index, row in df.iterrows():
    row_data = {}
    for col in ['ko', 'prediction']:
        if pd.notna(row[col]):
            row_data[col] = row[col]
    text_data[str(index)] = row_data

response_data = {
    "original": original_image_base64,     # 원본 메뉴판 이미지
    "detect": detected_image_base64,       # Bounding Box 시각화 이미지
    "crop": cropped_images_base64,         # 크롭된 개별 텍스트 이미지 리스트
    "data": text_data                      # {"0": {"ko": "제육볶음", "prediction": "Stir-fried pork"}, ...}
}
send_status = send_data_to_next(response_data)  # Next.js /api/receive로 POST
```

---

### 3.4 메인 서버 블로킹(Blocking)과 시스템 자원(OOM) 문제 제어

**상황:**  
대용량 이미지를 처리하다 보니 10단계 파이프라인 전체에 평균 수십 초가 소요되었습니다. API가 호출된 동안 프론트엔드와 백엔드의 HTTP 연결이 타임아웃되거나, Docker 컨테이너가 OOM(Out of Memory)으로 죽는 현상이 발생했습니다.

**해결:**  
- **WebSocket 기반 실시간 진행률 UI**: 타임아웃이 발생하지 않는 WebSocket 채널(`/ws/progress`)을 열고, 파이프라인의 각 단계마다 `update_progress()`를 호출하여 클라이언트에 실시간 퍼센티지를 푸시했습니다. `ConnectionManager` 클래스를 통해 다중 클라이언트 브로드캐스팅을 지원했습니다.

```python
# main.py - WebSocket을 통한 실시간 진행률 브로드캐스팅
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# 파이프라인 각 단계에서 호출
await update_progress(10)   # ② 크롭 완료
await update_progress(30)   # ④ OCR 컨테이너 전송 완료
await update_progress(50)   # ⑥ OCR 추론 완료
await update_progress(70)   # ⑧ 번역 컨테이너 전송 완료
await update_progress(90)   # ⑨ 번역 완료
await update_progress(100)  # ⑩ 최종 결과 전송 완료
```

- **도커 메모리 최적화**: Docker 컨테이너 런타임에 `--shm-size=8GB` 설정을 주입하여 공유 IPC 메모리를 충분히 할당하고, 서버 환경에 따라 배치 사이즈를 유동적으로 하향 조정(`--batch_size 64`, `--workers 16`)하여 모델의 안정성을 확보했습니다.

---

### 3.5 다양한 실무 스택 경험을 위한 앱/웹 하이브리드 아키텍처

**상황:**  
모바일 카메라만 구동해서 결과를 띄우는 것이 목적이라면 Flutter 단일 앱으로도 구현이 가능했습니다. 하지만 딥러닝 연동 환경에서 **웹과 앱을 아우르는 하이브리드 통신 스택**을 직접 다뤄보기 위해, 의도적으로 의존성을 쪼개는 아키텍처를 선택했습니다.

**해결:**  
- **하드웨어 제어(Flutter)**: `image_picker`와 `permission_handler`를 활용하여 카메라 촬영 / 갤러리 접근 등의 네이티브 기능을 쾌적하게 구현했습니다.
- **결과 렌더링(Next.js)**: 수십 개의 Base64 이미지와 텍스트 쌍을 다이내믹한 반응형 그리드(NextUI)로 렌더링하는 결과 페이지를 Next.js 14로 분리 개발했습니다. `/api/receive` API 라우트에서 FastAPI의 JSON을 수신하고, `DataContext`(React Context API)를 통해 전역 상태를 관리했습니다.
- **브릿지(WebView)**: Flutter 앱 내에서 `webview_flutter` 모듈을 통해 Next.js 결과 페이지를 로드함으로써, 이원화된 두 생태계를 하나의 앱처럼 매끄럽게 동작시켰습니다.

---

## 4. 핵심 성과

1. **상용화 수준의 End-to-End 서비스 구현 경험**
   - 단순히 머신러닝 논문의 소스코드를 훈련시켜 로컬에서 돌려보는 것을 넘어서, **App(Flutter) → Web(Next.js) → Middleware Server(FastAPI) → Dockerized AI Models(OCR, OpenNMT)** 전체 파이프라인 아키텍처를 총괄 설계하며 Full-Stack 및 MLOps 역량을 키웠습니다.
2. **AI 허브 공공데이터의 프로덕션 레벨 서비스 융합**
   - AI 허브에 등재된 우수한 사전학습 메커니즘을 실제 소프트웨어 모형에 결합하여 "외국인들이 활용할 수 있는 한식 메뉴 자동 번역기"라는 실질적인 가치를 창출했습니다.
3. **비동기 모델 처리와 UX 중심의 상태 관리**
   - 무거운 모델 추론으로 인한 단점을 WebSocket 기반 실시간 피드백 시스템으로 감싸 사용자 편의성을 높임으로써 프론트엔드와 백엔드 간 상태 교환 인터페이스 설계에 대한 인사이트를 얻었습니다.
4. **단계별 에러 핸들링과 파이프라인 안정성**
   - 10단계 파이프라인의 각 지점에서 `returncode` 검사 및 `stderr` 전달을 통해 실패 원인을 즉시 특정할 수 있는 디버깅 친화적 구조를 확립했습니다.
