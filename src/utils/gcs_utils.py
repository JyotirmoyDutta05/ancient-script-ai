from google.cloud import storage
import yaml

def load_config():
    with open("config/gcs.yaml", "r") as f:
        return yaml.safe_load(f)

def get_bucket():
    config = load_config()
    client = storage.Client()
    return client.bucket(config["bucket_name"])
