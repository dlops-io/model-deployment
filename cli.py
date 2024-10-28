"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --upload
        python cli.py --deploy
        python cli.py --predict
"""

import os
import requests
import zipfile
import tarfile
import argparse
from glob import glob
import numpy as np
import base64
from google.cloud import storage
from google.cloud import aiplatform
import tensorflow as tf

# # W&B
# import wandb

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_MODELS_BUCKET_NAME = os.environ["GCS_MODELS_BUCKET_NAME"]
BEST_MODEL = "model-mobilenetv2_train_base_True.v1"
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}/{BEST_MODEL}"

data_details = {
    "image_width": 224, 
    "image_height": 224, 
    "num_channels": 3, 
    "num_classes": 4, 
    "label2index": {"parmigiano": 0, "gruyere": 1, "brie": 2, "gouda": 3}, 
    "index2label": {"0": "parmigiano", "1": "gruyere", "2": "brie", "3": "gouda"}
}


def download_file(packet_url, base_path="", extract=False, headers=None):
    if base_path != "":
        if not os.path.exists(base_path):
            os.mkdir(base_path)
    packet_file = os.path.basename(packet_url)
    with requests.get(packet_url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(os.path.join(base_path, packet_file), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if extract:
        if packet_file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(base_path, packet_file)) as zfile:
                zfile.extractall(base_path)
        else:
            packet_name = packet_file.split(".")[0]
            with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
                tfile.extractall(base_path)


def prepare():
    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.get_bucket(GCS_MODELS_BUCKET_NAME)

    # Use this code if you want to pull your model directly from WandB
    # WANDB_KEY = os.environ["WANDB_KEY"]
    # # Login into wandb
    # wandb.login(key=WANDB_KEY)

    # # Download model artifact from wandb
    # run = wandb.init()
    # artifact = run.use_artifact(
    #     "ac215-harvard/cheese-app-demo/model-mobilenetv2_train_base_True",
    #     type="model",
    # )
    # artifact_dir = artifact.download()
    # print("artifact_dir", artifact_dir)

    # Download model
    download_file(
        "https://github.com/dlops-io/models/releases/download/v3.0/mobilenetv2_train_base_True.zip",
        base_path="artifacts",
        extract=True,
    )
    prediction_model_path = "./artifacts/mobilenetv2_train_base_True/mobilenetv2_train_base_True.keras"

    # Load model
    prediction_model = tf.keras.models.load_model(prediction_model_path)
    #print(prediction_model.summary())

    # Preprocess Image
    def preprocess_image(bytes_input):
        decoded = tf.io.decode_jpeg(bytes_input, channels=3)
        decoded = tf.image.convert_image_dtype(decoded, tf.float32)
        resized = tf.image.resize(decoded, size=(224, 224))
        return resized

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def preprocess_function(bytes_inputs):
        decoded_images = tf.map_fn(
            preprocess_image, bytes_inputs, dtype=tf.float32, back_prop=False
        )
        return {"model_input": decoded_images}

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_function(bytes_inputs):
        images = preprocess_function(bytes_inputs)
        results = model_call(**images)
        return results

    model_call = tf.function(prediction_model.call).get_concrete_function(
        [
            tf.TensorSpec(
                shape=[None, 224, 224, 3], dtype=tf.float32, name="model_input"
            )
        ]
    )

    # Save updated model to GCS
    tf.saved_model.save(
        prediction_model,
        "ARTIFACT_URI",
        signatures={"serving_default": serving_function},
    )
    # Save locally for local testing
    tf.saved_model.save(
        prediction_model,
        "saved_model",
        signatures={"serving_default": serving_function},
    )



def deploy():
    # List of prebuilt containers for prediction
    # https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    serving_container_image_uri = (
        "us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-16:latest"
    )

    # Upload and Deploy model to Vertex AI
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_upload
    deployed_model = aiplatform.Model.upload(
        display_name=BEST_MODEL,
        artifact_uri=ARTIFACT_URI,
        serving_container_image_uri=serving_container_image_uri,
    )
    print("deployed_model:", deployed_model)
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy
    endpoint = deployed_model.deploy(
        deployed_model_display_name=BEST_MODEL,
        traffic_split={"0": 100},
        # machine_type="n1-standard-4",
        accelerator_count=0,
        min_replica_count=1,
        max_replica_count=1,
        sync=False,
    )
    print("endpoint:", endpoint)


def predict():
    # Get the endpoint
    # Endpoint format: endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}"
    endpoint = aiplatform.Endpoint(
        "projects/129349313346/locations/us-central1/endpoints/6690889994342498304"
    )

    # Get a sample image to predict
    image_files = glob(os.path.join("data", "*.jpg"))
    print("image_files:", image_files[:5])

    image_samples = np.random.randint(0, high=len(image_files) - 1, size=5)
    for img_idx in image_samples:
        print("Image:", image_files[img_idx])

        with open(image_files[img_idx], "rb") as f:
            data = f.read()
        b64str = base64.b64encode(data).decode("utf-8")
        # The format of each instance should conform to the deployed model's prediction input schema.
        instances = [{"bytes_inputs": {"b64": b64str}}]

        result = endpoint.predict(instances=instances)

        print("Result:", result)
        prediction = result.predictions[0]
        prediction_index = prediction.index(max(prediction))
        print(prediction, prediction_index)
        print(
            "Label:   ",
            data_details["index2label"][str(prediction_index)],
            "\n",
        )


def main(args=None):

    if args.prepare:
        print("Prepare model and save model to GCS Bucket")
        prepare()

    elif args.deploy:
        print("Deploy model")
        deploy()

    elif args.predict:
        print("Predict using endpoint")
        predict()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare model and save model to GCS Bucket",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy saved model to Vertex AI",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Make prediction using the endpoint from Vertex AI",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test deployment to Vertex AI",
    )

    args = parser.parse_args()

    main(args)
