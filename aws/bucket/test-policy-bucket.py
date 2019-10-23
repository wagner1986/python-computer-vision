import boto3


BUCKET = "amazon-rekognition2"
# Retrieve the policy of the specified bucket
s3 = boto3.client('s3')
result = s3.get_bucket_policy(Bucket=BUCKET)
print(result['Policy'])