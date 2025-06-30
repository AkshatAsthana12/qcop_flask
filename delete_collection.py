import boto3
from botocore.exceptions import ClientError

def delete_collection(collection_id):
    client = boto3.client('rekognition')
    try:
        response = client.delete_collection(CollectionId=collection_id)
        if response['StatusCode'] == 200:
            print(f'Collection {collection_id} deleted successfully.')
        else:
            print(f'Failed to delete collection {collection_id}.')
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f'Collection {collection_id} does not exist.')
        else:
            raise

def main():
    collection_id = "my_face_collection"  
    delete_collection(collection_id)

if __name__ == "__main__":
    main()