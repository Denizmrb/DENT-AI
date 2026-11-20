import numpy as np
import pytest

from dentai.imaging import enhance_contrast, fit_to_screen, imread, imwrite, letterbox


@pytest.mark.parametrize("hw", [(600, 1200), (1200, 600), (640, 640), (37, 981)])
@pytest.mark.parametrize("target", [(640, 640), (320, 1024), (720, 1280)])
def test_letterbox_shape_and_roundtrip(hw, target):
    img = np.random.randint(0, 255, (*hw, 3), np.uint8)
    out, info = letterbox(img, target)
    assert out.shape[:2] == target
    box = np.array([[10.0, 5.0, min(100.0, hw[1]), min(50.0, hw[0])]])
    back = info.to_original(info.to_letterbox(box))
    assert np.allclose(back, box, atol=1e-3)


def test_letterbox_preserves_aspect_ratio():
    img = np.full((500, 1000, 3), 255, np.uint8)
    out, info = letterbox(img, (640, 640), color=(0, 0, 0))
    content = np.argwhere(out[..., 0] == 255)
    h = content[:, 0].max() - content[:, 0].min() + 1
    w = content[:, 1].max() - content[:, 1].min() + 1
    assert w / h == pytest.approx(2.0, abs=0.02)
    assert info.top > 0 and info.left == 0


def test_letterbox_no_scale_up_and_grayscale():
    g = np.zeros((100, 200), np.uint8)
    out, info = letterbox(g, (400, 400), scale_up=False)
    assert info.scale == 1.0 and out.shape == (400, 400)


def test_fit_to_screen_never_upscales():
    small = np.zeros((100, 200, 3), np.uint8)
    assert fit_to_screen(small, 1600, 900).shape[:2] == (100, 200)
    big = np.zeros((3000, 6000, 3), np.uint8)
    out = fit_to_screen(big, 1600, 900)
    assert out.shape[1] <= 1600 and out.shape[0] <= 900


def test_unicode_path_roundtrip(tmp_path):
    p = tmp_path / "Masaüstü" / "röntgen.png"
    img = np.random.randint(0, 255, (20, 30, 3), np.uint8)
    imwrite(p, img)
    assert np.array_equal(imread(p), img)


def test_enhance_contrast_shape():
    assert enhance_contrast(np.zeros((50, 60, 3), np.uint8)).shape == (50, 60, 3)
