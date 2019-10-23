import boto3
import os
import json
BUCKET = "amazon-rekognition2"
s3 = boto3.resource('s3')
my_bucket = s3.Bucket(BUCKET)
s3client = boto3.client('s3')
print(my_bucket.objects.all())
for s3_object in my_bucket.objects.all():
    path, file1 = os.path.split(s3_object.key)
    obj = s3client.get_object(Bucket=BUCKET,Key=file1)
    print(obj)