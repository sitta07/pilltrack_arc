import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from src.engine import AIEngine

@pytest.fixture
def fake_image():
    """สร้างภาพปลอมๆ (Noise) ขนาด 100x100 สี RGB"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def mock_engine_dependencies():
    """
    Fixture นี้จะ Mock ตัว YOLO และ torch.hub.load 
    ไม่ให้โหลดไฟล์จริงตอนสั่ง AIEngine(...)
    """
    with patch('src.engine.YOLO') as mock_yolo, \
         patch('src.engine.torch.hub.load') as mock_hub:
        
        # Setup Mock Return Values
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance
        
        # Setup DINO Mock
        mock_dino_model = MagicMock()
        mock_hub.return_value = mock_dino_model
        
        mock_dino_model.to.return_value = mock_dino_model
        mock_dino_model.eval.return_value = mock_dino_model
        
        yield mock_yolo_instance, mock_dino_model
# --- 2. TEST CASES ---

def test_engine_initialization(mock_engine_dependencies):
    """Test ว่า Init Class ผ่าน โดยไม่พังเรื่อง Path"""
    # Act
    engine = AIEngine("dummy_path.pt", dino_size=224)
    
    # Assert
    assert engine is not None
    assert engine.dino_size == 224
    assert hasattr(engine, 'sift')  # เช็คว่าสร้าง SIFT หรือยัง

def test_extract_features(mock_engine_dependencies, fake_image):
    """Test ฟังก์ชันหลัก extract_features"""
    _, mock_dino = mock_engine_dependencies
    
    # 1. Init Engine
    engine = AIEngine("dummy.pt", 224)
    
    # 2. Mock พฤติกรรมตอน DINO ถูกเรียกใช้งาน (Forward Pass)
    # สมมติว่า DINO คืนค่า Vector ขนาด (1, 768)
    mock_dino.return_value = torch.randn(1, 768)
    
    dino_vec, sift_des = engine.extract_features(fake_image)
    
    # 4. ตรวจสอบผลลัพธ์
    # DINO ต้องเป็น 1D Array (เพราะในโค้ดมี .flatten())
    assert isinstance(dino_vec, np.ndarray)
    assert len(dino_vec.shape) == 1 
    
    if sift_des is not None:
        assert isinstance(sift_des, np.ndarray)
        assert sift_des.shape[1] == 128  

def test_detect_and_crop_no_detection(mock_engine_dependencies, fake_image):
    """Test กรณี YOLO หาไม่เจอ (ต้องคืนรูปเดิมกลับมา)"""
    mock_yolo, _ = mock_engine_dependencies
    
    # 1. Setup Mock ให้ YOLO คืนค่าว่างเปล่า (No Masks)
    mock_result = MagicMock()
    mock_result.masks = None # จำลองว่าไม่มี Mask
    mock_yolo.return_value = [mock_result] # YOLO คืนค่าเป็น List
    
    # 2. Run
    engine = AIEngine("dummy.pt", 224)
    result_img = engine.detect_and_crop(fake_image, conf=0.5)
    
    # 3. Assert (ต้องได้รูปเดิมเป๊ะๆ)
    np.testing.assert_array_equal(result_img, fake_image)

def test_detect_and_crop_success(mock_engine_dependencies, fake_image):
    """Test กรณี YOLO เจอวัตถุ (ต้อง Crop ได้)"""
    mock_yolo, _ = mock_engine_dependencies
    
    # 1. สร้าง Mock Result ที่ซับซ้อนหน่อย (จำลอง Object YOLO)
    mock_result = MagicMock()
    
    # Mock Boxes (x1, y1, x2, y2) สมมติว่าเจอตรงกลางภาพ
    # box format: xyxy tensor
    mock_box = MagicMock()
    mock_box.xyxy = torch.tensor([[20, 20, 80, 80]]) 
    mock_box.conf = torch.tensor([0.9])
    mock_result.boxes = mock_box
    
    # Mock Masks (จำลอง Mask สี่เหลี่ยมตรงกลาง)
    mock_masks = MagicMock()
    # สร้าง Mask ขนาดเท่าภาพจริง (100x100)
    fake_mask_data = torch.zeros((1, 100, 100))
    fake_mask_data[0, 20:80, 20:80] = 1 # พื้นที่สีขาวตรงกลาง
    mock_masks.data = fake_mask_data
    mock_result.masks = mock_masks
    
    # สั่งให้ YOLO คืนค่า Mock นี้
    mock_yolo.return_value = [mock_result]
    
    # 2. Run
    engine = AIEngine("dummy.pt", 224)
    crop_img = engine.detect_and_crop(fake_image, conf=0.5)
    
    # 3. Assert
    # รูปที่ได้ต้องขนาดไม่เท่ารูปเดิม (เพราะโดน Crop)
    assert crop_img.shape != fake_image.shape
    # และต้องไม่ Empty
    assert crop_img.size > 0