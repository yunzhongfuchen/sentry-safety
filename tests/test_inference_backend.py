import numpy as np
from unittest.mock import MagicMock, patch
from inference_backend import YoloCudaBackend


def test_yolo_cuda_backend_predict_batch():
    mock_model = MagicMock()
    mock_model.return_value = [MagicMock()] * 2

    with patch("inference_backend.YOLO", return_value=mock_model):
        backend = YoloCudaBackend(model_path="dummy.pt", device="cpu", confidence=0.5)
        frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = backend.predict_batch(frames, "helmet")

    assert len(results) == 2
    mock_model.assert_called_once()
    call_kwargs = mock_model.call_args[1]
    assert call_kwargs.get("half") is False


def test_yolo_cuda_backend_predict_batch_with_half():
    mock_model = MagicMock()
    mock_model.return_value = [MagicMock()] * 2

    with patch("inference_backend.YOLO", return_value=mock_model):
        backend = YoloCudaBackend(model_path="dummy.pt", device="cpu", confidence=0.5)
        frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = backend.predict_batch(frames, "helmet", half=True)

    assert len(results) == 2
    mock_model.assert_called_once()
    call_kwargs = mock_model.call_args[1]
    assert call_kwargs.get("half") is True
