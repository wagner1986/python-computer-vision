import boto3

BUCKET = "amazon-rekognition2"
KEY = "teste.jpg"
region="us-east-1"
def detect_labels(bucket, key, max_labels=10, min_confidence=90, region="us-east-1"):
	rekognition = boto3.client("rekognition", region)
	response = rekognition.detect_labels(
		Image={
			"S3Object": {
				"Bucket": bucket,
				"Name": key,
			}
		},
		MaxLabels=max_labels,
		MinConfidence=min_confidence,
	)
	return response['Labels']

print(BUCKET, KEY)
for label in detect_labels(BUCKET, KEY,region=region):
	print("{Name} - {Confidence}%".format(**label))