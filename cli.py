"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --prepare
        python cli.py --test
        python cli.py --deploy
        python cli.py --predict
        python cli.py --cleanup
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
GCP_REGION = "us-central1"
BEST_MODEL = "model-mobilenetv2_train_base_True.v1"
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}/{BEST_MODEL}"
ENDPOINT_NAME = "cheese-app-endpoint"
# Smallest general-purpose machine; it is billed every hour while deployed
MACHINE_TYPE = "n1-standard-2"

data_details = {
    "image_width": 224,
    "image_height": 224,
    "num_channels": 3,
    "num_classes": 4,
    "label2index": {"parmigiano": 0, "gruyere": 1, "brie": 2, "gouda": 3},
    "index2label": {"0": "parmigiano", "1": "gruyere", "2": "brie", "3": "gouda"},
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
            with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
                tfile.extractall(base_path)


def prepare():
    # Fail early if the bucket does not exist or we cannot access it
    storage_client = storage.Client(project=GCP_PROJECT)
    storage_client.get_bucket(GCS_MODELS_BUCKET_NAME)

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

    # Download model (skipped if it was already downloaded)
    prediction_model_path = (
        "./artifacts/mobilenetv2_train_base_True/mobilenetv2_train_base_True.keras"
    )
    if not os.path.exists(prediction_model_path):
        download_file(
            "https://github.com/dlops-io/models/releases/download/v3.0/mobilenetv2_train_base_True.zip",
            base_path="artifacts",
            extract=True,
        )

    # Load model
    prediction_model = tf.keras.models.load_model(prediction_model_path)
    # print(prediction_model.summary())

    # Preprocess Image
    def preprocess_image(bytes_input):
        decoded = tf.io.decode_jpeg(bytes_input, channels=3)
        decoded = tf.image.convert_image_dtype(decoded, tf.float32)
        resized = tf.image.resize(decoded, size=(224, 224))
        return resized

    # Serving function: takes a batch of JPEG bytes, preprocesses each image, runs the model.
    # The argument name (bytes_inputs) is the key clients must use in prediction requests.
    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_function(bytes_inputs):
        images = tf.map_fn(
            preprocess_image, bytes_inputs, fn_output_signature=tf.float32
        )
        return prediction_model(images, training=False)

    # Save updated model to GCS, with the serving function as its default signature.
    # We use Keras' ExportArchive rather than tf.saved_model.save(): for Keras 3 models,
    # tf.saved_model.save() writes files that load in Python but fail in the TensorFlow
    # model server on Vertex AI ("Could not find variable ...").
    export_archive = tf.keras.export.ExportArchive()
    export_archive.track(prediction_model)
    export_archive.add_endpoint(name="serving_default", fn=serving_function)
    export_archive.write_out(ARTIFACT_URI, verbose=False)
    print("Saved model to:", ARTIFACT_URI)


def load_sample_images():
    # All sample images in data/ (any extension), as raw bytes
    image_files = sorted(glob(os.path.join("data", "*")))
    images = []
    for image_file in image_files:
        with open(image_file, "rb") as f:
            images.append(f.read())
    return image_files, images


def print_predictions(image_files, predictions):
    correct = 0
    for image_file, prediction in zip(image_files, predictions):
        prediction_index = int(np.argmax(prediction))
        label = data_details["index2label"][str(prediction_index)]
        # Image files are named after their cheese, e.g. data/brie-1.jpg
        correct += os.path.basename(image_file).startswith(label)
        print("Image:", image_file)
        print([round(float(p), 4) for p in prediction], prediction_index)
        print("Label:   ", label, "\n")
    print(f"Correct: {correct}/{len(image_files)}")


def test():
    # Load the packaged model from GCS (the same files Vertex AI will load) the way the
    # TensorFlow model server does: as a graph whose variables are restored by name.
    # The usual tf.saved_model.load() can succeed on a model the server cannot load.
    print("Loading model from:", ARTIFACT_URI)
    image_files, images = load_sample_images()
    with tf.Graph().as_default(), tf.compat.v1.Session() as session:
        meta_graph = tf.compat.v1.saved_model.load(session, ["serve"], ARTIFACT_URI)
        signature = meta_graph.signature_def["serving_default"]

        # Send raw image bytes, as the model server does after decoding the base64 in a request
        predictions = session.run(
            signature.outputs["output_0"].name,
            feed_dict={signature.inputs["bytes_inputs"].name: images},
        )
    print_predictions(image_files, predictions)


def deploy():
    # Prebuilt TensorFlow serving container (CPU). Vertex AI only accepts the images
    # on its supported list, and TF 2.15 is the newest one there. It serves the
    # SavedModel we export with TF 2.17 (the serving container only needs the graph,
    # not Keras). If the deploy fails with "does not have access to this container",
    # check the list for the currently supported images:
    # https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/pre-built-containers
    serving_container_image_uri = (
        "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"
    )

    # Upload the model to the Model Registry.
    # If a model with this name is already registered, add a new version of it
    # instead of creating a second model with the same name.
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_upload
    existing_models = aiplatform.Model.list(
        filter=f'display_name="{BEST_MODEL}"', order_by="create_time desc"
    )
    parent_model = existing_models[0].resource_name if existing_models else None
    model = aiplatform.Model.upload(
        display_name=BEST_MODEL,
        artifact_uri=ARTIFACT_URI,
        serving_container_image_uri=serving_container_image_uri,
        parent_model=parent_model,
    )
    print(f"Registered model: {model.resource_name} (version {model.version_id})")

    # Reuse our endpoint if it exists, so re-running --deploy never creates
    # a second endpoint (and a second machine to pay for)
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
    if endpoints:
        endpoint = endpoints[0]
        print("Using existing endpoint:", endpoint.resource_name)
    else:
        endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
        print("Created endpoint:", endpoint.resource_name)
    previously_deployed = endpoint.list_models()

    # Deploy the new model version and send it all the traffic.
    # This starts a machine and takes several minutes (often 10-20).
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy
    print(f"Deploying to the endpoint on a {MACHINE_TYPE} machine (this can take 10-20 minutes)...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=BEST_MODEL,
        traffic_percentage=100,
        machine_type=MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )

    # Undeploy older versions: they no longer get traffic but would still keep
    # their machines running (and billing)
    for deployed_model in previously_deployed:
        print("Undeploying previous model:", deployed_model.id)
        endpoint.undeploy(deployed_model_id=deployed_model.id)

    print("Endpoint ready:", endpoint.resource_name)


def predict(endpoint_name=None):
    # Use the endpoint given with --endpoint, or find the one created by --deploy
    # Endpoint format: "projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}" or just the ID
    if endpoint_name:
        endpoint = aiplatform.Endpoint(endpoint_name)
    else:
        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
        if not endpoints:
            raise SystemExit(
                f"No endpoint named '{ENDPOINT_NAME}' found. Run: python cli.py --deploy"
            )
        endpoint = endpoints[0]
    print("Endpoint:", endpoint.resource_name)
    if not endpoint.list_models():
        raise SystemExit("No model is deployed to this endpoint. Run: python cli.py --deploy")

    # Build one request with all sample images.
    # JSON cannot carry raw bytes, so each image is base64-encoded. The {"b64": ...}
    # wrapper tells the model server to decode it back to bytes, and the key
    # "bytes_inputs" must match the serving function's argument name (see prepare()).
    image_files, images = load_sample_images()
    instances = [
        {"bytes_inputs": {"b64": base64.b64encode(image).decode("utf-8")}}
        for image in images
    ]

    result = endpoint.predict(instances=instances)
    print_predictions(image_files, result.predictions)


def cleanup():
    # Undeploy the model and delete our endpoint so it stops billing.
    # The model versions in the Model Registry and the files in GCS are kept.
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
    if not endpoints:
        print(f"No endpoint named '{ENDPOINT_NAME}' found. Nothing to clean up.")
        return

    for endpoint in endpoints:
        print("Undeploying all models from:", endpoint.resource_name)
        endpoint.undeploy_all()
        endpoint.delete()
        print("Deleted endpoint:", endpoint.resource_name)


def main(args):
    # Tell the Vertex AI SDK which project and region to use for all calls
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)

    if args.prepare:
        print("Prepare model and save model to GCS Bucket")
        prepare()

    elif args.test:
        print("Test the packaged model locally")
        test()

    elif args.deploy:
        print("Deploy model")
        deploy()

    elif args.predict:
        print("Predict using endpoint")
        predict(args.endpoint)

    elif args.cleanup:
        print("Undeploy the model and delete the endpoint")
        cleanup()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Model Deployment CLI")
    parser = argparse.ArgumentParser(description="Model Deployment CLI")

    # Exactly one command must be given
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare model and save model to GCS Bucket",
    )
    command.add_argument(
        "--test",
        action="store_true",
        help="Test the packaged model from GCS locally before deploying",
    )
    command.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy saved model to Vertex AI",
    )
    command.add_argument(
        "--predict",
        action="store_true",
        help="Make prediction using the endpoint from Vertex AI",
    )
    command.add_argument(
        "--cleanup",
        action="store_true",
        help="Undeploy the model and delete the endpoint (stops billing)",
    )

    parser.add_argument(
        "--endpoint",
        help=f"Endpoint ID or resource name for --predict (default: the '{ENDPOINT_NAME}' endpoint)",
    )

    args = parser.parse_args()

    main(args)
