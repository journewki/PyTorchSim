# SANA + FALQON PyTorchSim Test

SANA Diffusion Transformer에 FALQON(FP8 LoRA) 학습을 적용한 코드를 PyTorchSim NPU 시뮬레이터에서 실행할 수 있도록 단순화한 테스트 코드.

## 배경

- **SANA**: Linear Attention 기반 Diffusion Transformer ([논문](https://huggingface.co/papers/2410.10629))
- **FALQON**: FP8 양자화 + LoRA fine-tuning 기법 (Choi et al., 2025)
- **PyTorchSim**: NPU 아키텍처 시뮬레이터 — `torch.compile(dynamic=False)`로 TOG 생성 후 cycle 측정

원본 학습 스크립트(`train_dreambooth_lora_sana_falqon.py`, 2626줄)를 `test_llama.py` / `test_diffusion.py` 패턴에 맞춰 블록별 분리 테스트로 재구성했다.

## 파일 구조

```
tests/Diffusion/
├── test_sana_falqon.py         # 메인 테스트 코드 (9개 테스트)
├── run_test_sana_falqon.sh     # 실행 스크립트
├── README_sana_falqon.md       # 이 문서
└── test_diffusion.py           # 기존 UNet 테스트 (참고용)
```

## 실행 환경

### Phase 1: CUDA 검증 (현재)

```bash
conda activate diffusers
```

필수 패키지: `torch>=2.0`, `diffusers>=0.37.0`, CUDA GPU

### Phase 2: PyTorchSim NPU 시뮬레이션 (추후)

PyTorchSim 환경 설정 후 `--device npu`로 전환.

## 사용법

### 전체 테스트 실행

```bash
# 셸 스크립트 사용
bash run_test_sana_falqon.sh

# 또는 직접 실행
python test_sana_falqon.py --device cuda --tests all
```

### 개별 테스트 실행

```bash
# FALQON Linear 레이어 단독 (forward + backward)
python test_sana_falqon.py --device cuda --tests falqon_linear

# SANA 블록별 테스트
python test_sana_falqon.py --device cuda --tests linear_attention
python test_sana_falqon.py --device cuda --tests cross_attention
python test_sana_falqon.py --device cuda --tests mixffn

# Transformer 블록 / 전체 모델
python test_sana_falqon.py --device cuda --tests transformer_block
python test_sana_falqon.py --device cuda --tests falqon_block
python test_sana_falqon.py --device cuda --tests transformer_forward

# 학습 테스트
python test_sana_falqon.py --device cuda --tests training_step
python test_sana_falqon.py --device cuda --tests flow_matching --num_steps 10
```

### 복수 테스트 조합

```bash
python test_sana_falqon.py --device cuda --tests "falqon_linear,training_step"
python test_sana_falqon.py --device cuda --tests "linear_attention,cross_attention,mixffn"
```

### PyTorchSim NPU 실행 (Phase 2)

```bash
python test_sana_falqon.py --device npu --tests all
```

## CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--device` | `cuda` | `cuda` (Phase 1), `npu` (Phase 2), `cpu` |
| `--tests` | `all` | 실행할 테스트 이름 (쉼표 구분) |
| `--rank` | `4` | FALQON LoRA rank |
| `--dtype` | `float32` | 데이터 타입 |
| `--rtol` | `1e-3` | 상대 허용 오차 |
| `--atol` | `1e-3` | 절대 허용 오차 |
| `--num_steps` | `5` | flow_matching 테스트 학습 스텝 수 |

## 테스트 목록

Bottom-up 순서로 작은 블록부터 전체 모델, 학습까지 검증한다.

| # | 이름 | 대상 | 유형 |
|---|------|------|------|
| 1 | `falqon_linear` | FalqonLinearEmulated (forward + backward) | 단일 레이어 |
| 2 | `linear_attention` | SanaLinearAttnProcessor2_0 (ReLU linear attention) | 서브 블록 |
| 3 | `cross_attention` | SanaAttnProcessor2_0 (SDPA cross attention) | 서브 블록 |
| 4 | `mixffn` | GLUMBConv (3x3 depthwise Conv + GLU) | 서브 블록 |
| 5 | `transformer_block` | SanaTransformerBlock (attn + cross_attn + FFN) | 블록 |
| 5b | `falqon_block` | SanaTransformerBlock + FALQON 변환 | 블록 |
| 6 | `transformer_forward` | SanaTransformer2DModel (전체 모델 forward) | 전체 모델 |
| 7 | `training_step` | Forward + Backward + Optimizer + Weight Fusion | 학습 |
| 8 | `flow_matching` | Flow Matching 노이즈 스케줄링 + 다중 스텝 학습 | 학습 |

## SANA 아키텍처 → 테스트 매핑

```
SanaTransformer2DModel                    [Test 6: transformer_forward]
├── patch_embed (Conv2d)
├── time_embed (AdaLayerNormSingle)
├── caption_projection + caption_norm
├── transformer_blocks[i]                 [Test 5: transformer_block]
│   ├── norm1 + attn1 (Linear Attention)  [Test 2: linear_attention]
│   │   └── to_q, to_k, to_v, to_out.0   [Test 1: falqon_linear] ← FALQON 대상
│   ├── norm2 + attn2 (Cross Attention)   [Test 3: cross_attention]
│   │   └── to_q, to_k, to_v, to_out.0   ← FALQON 대상
│   └── ff (GLUMBConv)                    [Test 4: mixffn]
├── norm_out (SanaModulatedNorm)
└── proj_out (Linear)
```

## FALQON 에뮬레이션 방식

PyTorchSim에 FP8 op이 등록되어 있지 않으므로, FALQON의 수학적 구조를 BF16/FP32로 에뮬레이션한다.

```
원본 FALQON:  output = FP8_matmul([W_fp8 | A_fp8], x) → split → W·x
에뮬레이션:   output = matmul([W_bf16 | A_bf16], x)    → split → W·x
```

보존되는 핵심 구조:
- **Weight**: `[Q(W) | A]` 결합된 단일 행렬 (frozen)
- **B**: `[out_features, rank]` 유일한 학습 파라미터
- **Forward**: 단일 matmul → split → main output
- **Backward**: `grad_B = grad_output^T @ (A·x)` (B에만 gradient)
- **Weight Fusion**: `W_new = W + B @ A^T`, B → zero 리셋

## 축소 모델 설정

시뮬레이션 속도를 위해 실제 SANA-0.6B보다 작은 설정을 사용한다.

| 파라미터 | 테스트 설정 | 실제 SANA-0.6B |
|---------|-----------|---------------|
| inner_dim | 64 | 2240 |
| num_attention_heads | 4 | 70 |
| num_layers | 2 | 20 |
| cross_attention_dim | 64 | 2240 |
| sample_size | 8x8 | 32x32 |
| FALQON rank | 4 | 16 |

## 검증 결과 (Phase 1, CUDA)

```
[Test 1]  FalqonLinear forward         PASS  (max diff: 4.8e-7)
[Test 1b] FalqonLinear B.grad          PASS  (max diff: 8.7e-11)
[Test 2]  SanaLinearAttn               PASS  (max diff: 6.4e-5)
[Test 3]  SanaCrossAttn                PASS  (max diff: 1.3e-7)
[Test 4]  GLUMBConv (Mix-FFN)          PASS  (max diff: 3.7e-8)
[Test 5]  SanaTransformerBlock         PASS  (max diff: 2.8e-4)
[Test 5b] SanaTransformerBlock+FALQON  PASS  (max diff: 3.1e-4)
[Test 6]  SanaTransformer2DModel       PASS  (max diff: 2.8e-5)
[Test 7]  FALQON Training Step         PASS  (loss: 2.43 → 2.41)
[Test 8]  Flow Matching Loop (5 steps) PASS  (loss: 2.66 → 2.22)
```