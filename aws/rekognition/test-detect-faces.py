import boto3

BUCKET = "amazon-rekognition2"
KEY = "teste.jpg"
region = "us-east-1"
FEATURES_BLACKLIST = ("Landmarks", "Emotions", "Pose",
                      "Quality", "BoundingBox", "Confidence")


def detect_faces(bucket, key, attributes=['ALL'], region="us-east-1"):
    rekognition = boto3.client("rekognition", region)
    response = rekognition.detect_faces(
        Image={
            "S3Object": {
                "Bucket": bucket,
                "Name": key,
            }
        },
        Attributes=attributes,
    )
    return response['FaceDetails']


for face in detect_faces(BUCKET, KEY, region=region):
    print("Face ({Confidence}%)".format(**face))
    # emotions
    for emotion in face['Emotions']:
        print("  {Type} : {Confidence}%".format(**emotion))
    # quality
    for quality_key in face['Quality']:
    	print("  {quality} : {value}".format(quality=quality_key, value=face['Quality'][quality_key]))
    # facial features
    for feature in face:
        if feature not in FEATURES_BLACKLIST:
            print("  {feature} {data}".format(feature=feature, data=face[feature]))
