import boto3
from botocore.exceptions import ClientError
import os

class CloudManager:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name

    def check_connection(self):
        if not self.bucket: return False, "🔴 Config Missing"
        try:
            self.s3.get_bucket_location(Bucket=self.bucket)
            return True, "🟢 Ready"
        except:
            return False, "🔴 Offline/Access Denied"

    def get_inventory(self):
        try:
            res = self.s3.list_objects_v2(Bucket=self.bucket, Prefix='latest/')
            return [obj['Key'].replace('latest/', '') for obj in res.get('Contents', []) if obj['Key'] != 'latest/']
        except: return []

    def upload_file(self, local_path, s3_path):
        self.s3.upload_file(local_path, self.bucket, s3_path)