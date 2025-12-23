import boto3
from botocore.exceptions import ClientError, NoCredentialsError

class CloudManager:
    def __init__(self, bucket_name):
        # FIX 1: แก้ชื่อตัวแปรให้ตรงกับ Test
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3')

    def check_connection(self):
        """
        เช็คสถานะ S3
        Return: (is_connected: bool, message: str)
        """
        if not self.bucket_name: 
            return False, "🔴 Config Missing"
            
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            return True, "🟢 Online"
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '403':
                return False, "🔴 Forbidden (Access Denied)"
            elif error_code == '404':
                return False, "🔴 Bucket Not Found"
            else:
                return False, f"🔴 Error: {error_code}"
        except NoCredentialsError:
            return False, "🔴 No Credentials Found"
        except Exception as e:
            return False, f"🔴 Connection Failed: {str(e)}"

    def upload_file(self, local_path, s3_path):
        """Upload file ไปยัง S3 (มี try-except ดัก Error)"""
        try:
            self.s3.upload_file(local_path, self.bucket_name, s3_path)
            return True
        except Exception as e:
            print(f"Upload Error: {e}")
            return False

    def download_file(self, s3_path, local_path):
        """
        FIX 4: เพิ่มฟังก์ชันนี้เข้ามา เพราะ Test และ App เรียกใช้
        """
        try:
            self.s3.download_file(self.bucket_name, s3_path, local_path)
            return True
        except Exception as e:
            print(f"Download Error: {e}")
            return False
            
    def get_inventory(self):
        try:
            res = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix='latest/')
            return [obj['Key'].replace('latest/', '') for obj in res.get('Contents', []) if obj['Key'] != 'latest/']
        except: return []