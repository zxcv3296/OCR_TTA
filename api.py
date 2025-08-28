import io
import os
import base64
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import onnx
import onnxruntime as ort


# -------------------------
# Configuration
# -------------------------

ONNX_MODEL_PATH: str = './model.onnx'
CHARACTER_TXT: str = './char_file.txt'
PREDICTION: str = 'CTC'  # 'CTC' or 'Attn'
IMG_H: int = 32
IMG_W: int = 100
RGB: bool = False
PAD: bool = False
BATCH_MAX_LENGTH: int = 25
NORMALIZE_TO_NEG1_POS1: bool = False


# -------------------------
# Utilities
# -------------------------

def select_providers() -> List[str]:
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'AzureExecutionProvider' in available and 'CPUExecutionProvider' in available:
        return ['AzureExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def read_character_file(char_path: str) -> str:
    if not os.path.isfile(char_path):
        raise FileNotFoundError(f"Character file not found: {char_path}")
    chars: List[str] = []
    with open(char_path, 'r', encoding='utf-8') as f:
        for line in f:
            c = line.strip('\n\r')
            if c:
                chars.append(c)
    return ''.join(chars)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")


def preprocess_image(pil_img: Image.Image, img_h: int, img_w: int, rgb: bool, pad: bool,
                     normalize_to_neg1_pos1: bool) -> np.ndarray:
    # Convert color space
    if not rgb:
        pil_img = pil_img.convert('L')
    else:
        pil_img = pil_img.convert('RGB')

    orig_w, orig_h = pil_img.size
    if pad:
        # Keep ratio by height-first resize
        scale = img_h / float(orig_h)
        resized_w = int(np.floor(orig_w * scale))
        resized_w = max(1, min(resized_w, img_w))
        resized = pil_img.resize((resized_w, img_h), Image.BILINEAR)

        # Convert resized to CHW np array
        resized_np = np.asarray(resized).astype(np.float32)
        if rgb:
            resized_np = np.transpose(resized_np, (2, 0, 1))  # C,H,W
            c = 3
        else:
            resized_np = np.expand_dims(resized_np, axis=0)   # 1,H,W
            c = 1

        # Normalize to [0,1]
        resized_np = resized_np / 255.0
        if normalize_to_neg1_pos1:
            resized_np = (resized_np - 0.5) / 0.5

        # Allocate target and copy, then replicate last column like NormalizePAD
        out_np = np.zeros((c, img_h, img_w), dtype=np.float32)
        out_np[:, :, :resized_w] = resized_np
        if resized_w < img_w:
            out_np[:, :, resized_w:] = np.expand_dims(resized_np[:, :, resized_w - 1], axis=2)

        np_img = out_np
    else:
        # Simple resize to target size
        pil_img = pil_img.resize((img_w, img_h), Image.BILINEAR)
        np_img = np.asarray(pil_img).astype(np.float32)
        if rgb:
            np_img = np.transpose(np_img, (2, 0, 1))
        else:
            np_img = np.expand_dims(np_img, axis=0)
        np_img = np_img / 255.0
        if normalize_to_neg1_pos1:
            np_img = (np_img - 0.5) / 0.5

    # Add batch dimension: 1, C, H, W
    np_img = np.expand_dims(np_img, axis=0).astype(np.float32)
    return np_img


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s


def decode_ctc(preds_logits: np.ndarray, charset: str) -> Tuple[str, float]:
    """
    preds_logits: (T, C) or (1, T, C)
    charset: characters, length = num_classes - 1 (blank is index 0)
    Return: (text, confidence)
    """
    if preds_logits.ndim == 3:
        # (1, T, C)
        preds_logits = preds_logits[0]

    probs = softmax_np(preds_logits, axis=1)  # (T, C)
    pred_indices: np.ndarray = np.argmax(probs, axis=1)  # (T,)
    pred_max_probs: np.ndarray = np.max(probs, axis=1)   # (T,)

    blank_index = 0
    result_chars: List[str] = []
    conf_list: List[float] = []

    prev_idx = None
    for t, idx in enumerate(pred_indices.tolist()):
        if idx == blank_index:
            prev_idx = idx
            continue
        if idx == prev_idx:
            prev_idx = idx
            continue
        # map: 1..len(charset) -> charset
        mapped_index = idx - 1
        if 0 <= mapped_index < len(charset):
            result_chars.append(charset[mapped_index])
            conf_list.append(float(pred_max_probs[t]))
        prev_idx = idx

    text = ''.join(result_chars)
    confidence = float(np.prod(conf_list)) if len(conf_list) > 0 else 0.0
    return text, confidence


def decode_attn(preds_logits: np.ndarray, charset: str) -> Tuple[str, float]:
    """
    preds_logits: (T, C) or (1, T, C)
    charset: characters; assumes indices:
      0: [GO], 1..N: charset, N+1: [s]
    Return: (text, confidence)
    """
    if preds_logits.ndim == 3:
        preds_logits = preds_logits[0]

    T = preds_logits.shape[0]
    probs = softmax_np(preds_logits, axis=1)  # (T, C)
    pred_indices: np.ndarray = np.argmax(probs, axis=1)
    pred_max_probs: np.ndarray = np.max(probs, axis=1)

    eos_index = len(charset) + 1
    text_chars: List[str] = []
    conf_list: List[float] = []

    for t in range(T):
        idx = int(pred_indices[t])
        if idx == 0:  # [GO]
            continue
        if idx == eos_index:  # [s]
            break
        mapped_index = idx - 1
        if 0 <= mapped_index < len(charset):
            text_chars.append(charset[mapped_index])
            conf_list.append(float(pred_max_probs[t]))

    text = ''.join(text_chars)
    confidence = float(np.prod(conf_list)) if len(conf_list) > 0 else 0.0
    return text, confidence


def run_onnx(model_sess: ort.InferenceSession, input_names: List[str], output_names: List[str],
             image_array_bchw: np.ndarray,
             need_text: bool,
             batch_max_length: int) -> np.ndarray:
    ort_inputs = {input_names[0]: image_array_bchw}
    if need_text:
        # Shape: (N, L) with zeros; assumes int64 input for text
        text_input = np.zeros((image_array_bchw.shape[0], batch_max_length + 1), dtype=np.int64)
        ort_inputs[input_names[1]] = text_input
    outputs = model_sess.run(output_names, ort_inputs)
    logits = outputs[0]  # (N, T, C)
    return logits


# -------------------------
# App and startup
# -------------------------

app = FastAPI(title="OCR ONNX Inference API", version="1.0.0")


class Base64Image(BaseModel):
    image_base64: str


class InferenceResponse(BaseModel):
    text: str


@app.on_event("startup")
def startup_event():
    if not ONNX_MODEL_PATH:
        raise RuntimeError("환경변수 ONNX_MODEL 이 설정되어야 합니다.")
    if not CHARACTER_TXT:
        raise RuntimeError("환경변수 CHARACTER_TXT 가 설정되어야 합니다.")
    if PREDICTION not in {"CTC", "Attn"}:
        raise RuntimeError("환경변수 PREDICTION 은 'CTC' 또는 'Attn' 이어야 합니다.")

    # Validate files
    if not os.path.isfile(ONNX_MODEL_PATH):
        raise RuntimeError(f"ONNX 모델 파일을 찾을 수 없습니다: {ONNX_MODEL_PATH}")
    if not os.path.isfile(CHARACTER_TXT):
        raise RuntimeError(f"문자 집합 파일을 찾을 수 없습니다: {CHARACTER_TXT}")

    # Basic ONNX check
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)

    # Cache objects on app.state
    app.state.charset = read_character_file(CHARACTER_TXT)
    app.state.providers = select_providers()
    app.state.sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=app.state.providers)
    app.state.input_names = [i.name for i in app.state.sess.get_inputs()]
    app.state.output_names = [o.name for o in app.state.sess.get_outputs()]
    app.state.need_text = len(app.state.input_names) >= 2


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': os.path.basename(ONNX_MODEL_PATH),
        'providers': app.state.providers,
        'inputs': app.state.input_names,
        'outputs': app.state.output_names,
        'prediction': PREDICTION,
        'imgH': IMG_H,
        'imgW': IMG_W,
        'rgb': RGB,
        'pad': PAD,
    }


def infer_image(pil_img: Image.Image) -> InferenceResponse:
    import time
    np_bchw = preprocess_image(
        pil_img, IMG_H, IMG_W, RGB, PAD, NORMALIZE_TO_NEG1_POS1
    )
    t0 = time.time()
    logits = run_onnx(
        app.state.sess,
        app.state.input_names,
        app.state.output_names,
        np_bchw,
        app.state.need_text,
        BATCH_MAX_LENGTH,
    )  # (N, T, C)
    elapsed_ms = (time.time() - t0) * 1000.0

    logits_single = logits[0]  # (T, C)
    if PREDICTION == 'CTC':
        text, conf = decode_ctc(logits_single, app.state.charset)
    else:
        text, conf = decode_attn(logits_single, app.state.charset)

    return InferenceResponse(text=text)


@app.post('/infer/file', response_model=InferenceResponse)
async def infer_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pil = load_image_from_bytes(content)
        return infer_image(pil)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")


@app.post('/infer/base64', response_model=InferenceResponse)
def infer_base64(payload: Base64Image = Body(...)):
    try:
        header_sep = payload.image_base64.find(',')
        if header_sep != -1:
            b64 = payload.image_base64[header_sep+1:]
        else:
            b64 = payload.image_base64
        img_bytes = base64.b64decode(b64, validate=True)
        pil = load_image_from_bytes(img_bytes)
        return infer_image(pil)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"BASE64 처리 실패: {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'api:app', host='0.0.0.0', port=int(os.environ.get('PORT', '8000')), reload=True
    )
