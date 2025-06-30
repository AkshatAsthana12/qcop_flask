import boto3

s3 = boto3.client("s3")
response = s3.get_bucket_location(Bucket="qcopbucket1")
print("✅ Bucket region:", response["LocationConstraint"])