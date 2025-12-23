import pytest
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError
from src.cloud_manager import CloudManager

# ==========================================
# 🛠️ FIXTURES: เตรียมของปลอม (Mock)
# ==========================================

@pytest.fixture
def mock_boto_client():
    """
    Mock ตัว boto3.client เพื่อไม่ให้ code ต่อไปยัง AWS จริงๆ
    """
    with patch('src.cloud_manager.boto3.client') as mock_boto:
        mock_s3_instance = MagicMock()
        mock_boto.return_value = mock_s3_instance
        yield mock_s3_instance

@pytest.fixture
def cloud_manager(mock_boto_client):
    """
    สร้าง Instance ของ CloudManager โดยใช้ S3 ปลอม
    """
    return CloudManager(bucket_name="test-bucket")

# ==========================================
# 🧪 TEST CASES
# ==========================================

def test_init(cloud_manager):
    """Test ว่า Init Class แล้วได้ค่าถูกต้อง"""
    assert cloud_manager.bucket_name == "test-bucket"
    assert cloud_manager.s3 is not None

def test_check_connection_success(cloud_manager, mock_boto_client):
    """Test กรณีต่อ S3 ติด (Connection OK)"""
    # Setup: ไม่ต้องทำอะไรเพิ่ม เพราะ Mock ปกติถือว่าผ่านอยู่แล้ว
    
    # Act
    is_connected, status_msg = cloud_manager.check_connection()
    
    # Assert
    assert is_connected is True
    assert "Online" in status_msg
    # เช็คว่ามีการเรียก head_bucket จริง
    mock_boto_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

def test_check_connection_failed_403(cloud_manager, mock_boto_client):
    """Test กรณีไม่มีสิทธิ์ (Forbidden 403)"""
    # Setup: จำลอง Error 403 จาก AWS
    error_response = {'Error': {'Code': '403', 'Message': 'Forbidden'}}
    mock_boto_client.head_bucket.side_effect = ClientError(error_response, 'HeadBucket')
    
    # Act
    is_connected, status_msg = cloud_manager.check_connection()
    
    # Assert
    assert is_connected is False
    assert "Forbidden" in status_msg

def test_check_connection_failed_404(cloud_manager, mock_boto_client):
    """Test กรณีหา Bucket ไม่เจอ (Not Found 404)"""
    # Setup: จำลอง Error 404
    error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
    mock_boto_client.head_bucket.side_effect = ClientError(error_response, 'HeadBucket')
    
    # Act
    is_connected, status_msg = cloud_manager.check_connection()
    
    # Assert
    assert is_connected is False
    assert "Not Found" in status_msg

def test_check_connection_no_credentials(cloud_manager, mock_boto_client):
    """Test กรณีลืมใส่ Key (No Credentials)"""
    # Setup: จำลอง Error NoCredentialsError
    mock_boto_client.head_bucket.side_effect = NoCredentialsError()
    
    # Act
    is_connected, status_msg = cloud_manager.check_connection()
    
    # Assert
    assert is_connected is False
    assert "No Credentials" in status_msg

def test_upload_file_success(cloud_manager, mock_boto_client):
    """Test ว่าคำสั่ง Upload ถูกเรียกถูกต้อง"""
    # Act
    local_path = "data/model.pkl"
    s3_path = "models/model.pkl"
    result = cloud_manager.upload_file(local_path, s3_path)
    
    # Assert
    assert result is True
    mock_boto_client.upload_file.assert_called_once_with(
        local_path, 
        "test-bucket", 
        s3_path
    )

def test_upload_file_failure(cloud_manager, mock_boto_client):
    """Test กรณี Upload พัง (เช่น เน็ตหลุด)"""
    # Setup: สั่งให้ upload_file ระเบิด
    mock_boto_client.upload_file.side_effect = Exception("Upload Failed")
    
    # Act
    result = cloud_manager.upload_file("data/test.pkl", "remote/test.pkl")
    
    # Assert
    assert result is False  # ต้อง return False ไม่ใช่ Crash

def test_download_file_success(cloud_manager, mock_boto_client):
    """Test ว่าคำสั่ง Download ถูกเรียกถูกต้อง"""
    # Act
    s3_path = "models/model.pkl"
    local_path = "data/model.pkl"
    result = cloud_manager.download_file(s3_path, local_path)
    
    # Assert
    assert result is True
    # เช็คว่า Method download_file ของ boto3 ถูกเรียกด้วย args ที่ถูกเป๊ะๆ
    mock_boto_client.download_file.assert_called_once_with(
        "test-bucket",
        s3_path,
        local_path
    )