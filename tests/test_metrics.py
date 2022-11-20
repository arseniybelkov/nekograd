import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryF1Score, BinaryPrecision,
                                         BinaryRecall)

from nekograd.metrics.binary import (accuracy, binary_cross_entropy, f1_score,
                                     precision, recall, roc_auc)


@pytest.fixture
def gray_image():
    return np.array([[0.4, 0.3, 0.2], [0.9, 0.1, 0.9], [0.8, 0.7, 0.6]]).astype(np.float32)


@pytest.fixture
def rgb_image(gray_image):
    return np.stack([gray_image for _ in range(3)]).transpose(1, 2, 0).astype(np.float32)


@pytest.fixture
def segmentation_masks(gray_image, rgb_image):
    gm_02 = gray_image > 0.2
    gm_05 = gray_image > 0.5
    
    rgbm_02 = rgb_image > 0.2
    rgbm_05 = rgb_image > 0.5
    
    return gm_02, gm_05, rgbm_02, rgbm_05


def test_binary_segmentation(gray_image, segmentation_masks):
    gm_02, gm_05, _, _ = segmentation_masks
    
    # accuracy
    g02_05 = accuracy(gm_02, gm_05)
    g02_05_thr = accuracy(gm_02, gray_image[None, ...])
    g02_05_t = BinaryAccuracy()(torch.from_numpy(gm_05[None, ...]),
                                     torch.from_numpy(gm_02[None, ...])).numpy() 
    assert np.allclose(g02_05, g02_05_t)
    assert np.allclose(g02_05_thr, g02_05_t)
    
    # precision
    g02_05 = precision(gm_02, gm_05)
    g02_05_thr = precision(gm_02, gray_image[None, ...])
    g02_05_t = BinaryPrecision()(torch.from_numpy(gm_05[None, ...]),
                                     torch.from_numpy(gm_02[None, ...])).numpy()
    assert np.allclose(g02_05, g02_05_t)
    assert np.allclose(g02_05_thr, g02_05_t)
    
    # recall
    g02_05 = recall(gm_02, gm_05)
    g02_05_thr = recall(gm_02, gray_image[None, ...])
    g02_05_t = BinaryRecall()(torch.from_numpy(gm_05[None, ...]),
                                     torch.from_numpy(gm_02[None, ...])).numpy()
    assert np.allclose(g02_05, g02_05_t)
    assert np.allclose(g02_05_thr, g02_05_t)
    
    # f1-score
    g02_05 = f1_score(gm_02, gm_05)
    g02_05_thr = f1_score(gm_02, gray_image[None, ...])
    g02_05_t = BinaryF1Score()(torch.from_numpy(gm_05[None, ...]),
                                     torch.from_numpy(gm_02[None, ...])).numpy()
    assert np.allclose(g02_05, g02_05_t)
    assert np.allclose(g02_05_thr, g02_05_t)
    
    # roc-auc
    g02_05 = roc_auc(gm_02, gray_image[None, ...])
    g02_05_t = BinaryAUROC()(torch.from_numpy(gray_image[None, ...]),
                                     torch.from_numpy(gm_02[None, ...])).numpy()
    assert np.allclose(g02_05, g02_05_t)

    # bce
    g02_05 = binary_cross_entropy(gm_02, gray_image)
    g02_05_t = F.binary_cross_entropy(torch.from_numpy(gray_image[None, ...]),
                        torch.from_numpy(gm_02[None, ...].astype(np.float32))).numpy()
    assert np.allclose(g02_05, g02_05_t)


def test_binary_classification():
    p = [0.1, 0.2, 0.6]
    t = [1, 0, 1]
    
    from sklearn.metrics import roc_auc_score, log_loss
    
    assert np.allclose(accuracy(t, p), 2 / 3)
    assert np.allclose(precision(t, p), 1 / 1)
    assert np.allclose(recall(t, p), 1 / (1 + 1))
    assert np.allclose(f1_score(t, p), 2 / (2 + 1))
    assert np.allclose(roc_auc(t, p), roc_auc_score(t, p))
    assert np.allclose(binary_cross_entropy(t, p), log_loss(t, p))