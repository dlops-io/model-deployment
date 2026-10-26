# Cheese App: Model Deployment

🧀 🏋️ → 📦 → ☁️ → 🔌 → 🔮

The goal of this tutorial is to take the **cheese classifier we already trained** and turn it into a **live prediction service in the cloud**: an HTTPS endpoint that anyone with the right credentials can send an image to and get back which cheese it is.

Training a model is only half the job. A model file sitting on a laptop (or in WandB) is not useful to an app. To be useful, it has to be **served**: loaded on a machine that is always on, behind an address the app can call. In this tutorial we let Google's managed ML platform do the serving for us, so we never have to manage a server.

> **Note on names:** In 2026 Google renamed Vertex AI to **Gemini Enterprise Agent Platform**. The console now says "Agent Platform", but the API (`aiplatform.googleapis.com`), the Python SDK (`google-cloud-aiplatform`), Model Registry and Online prediction work exactly as before. In this tutorial, "Vertex AI" and "Agent Platform" mean the same thing.

Here is the story, step by step:

1. In the previous tutorials we trained a **MobileNetV2** cheese classifier (brie, gouda, gruyere, parmigiano). We saved the trained weights; for this tutorial they are published as a GitHub release so everyone can download them.
2. We **prepare** the model for serving: we wrap it with a small function that accepts raw JPEG bytes, decodes and resizes them, and then calls the model. The packaged model is saved to a **GCS bucket**.
3. We **upload** the packaged model to the **Vertex AI Model Registry**, telling Vertex which ready-made TensorFlow serving container should run it.
4. We **deploy** the registered model to a **Vertex AI Endpoint**. Google starts a machine, loads the model, and gives us an endpoint address.
5. We **predict**: we send cheese images to the endpoint and get back class probabilities.
6. We **clean up**: we undeploy the model so the endpoint stops costing money.

The deployment flow is illustrated below:
<img src="images/serverless-model-deployment.png" width="800">

---

## Prerequisites

Complete **[Tutorial 0 - Setup & Installs](https://github.com/dlops-io/ac215-setup)** first. It covers a GCP account and project, VS Code, Git, Docker Desktop and `uv`.

> Unlike the mega-pipeline tutorial, you will create **your own** service account and bucket in **your own** GCP project. Deploying a model starts machines that are billed to that project, so do the **Clean Up** step at the end.

---

## Clone this Repository

```bash
git clone https://github.com/dlops-io/model-deployment.git
cd model-deployment
```

Everything runs from one container, driven by `cli.py`.

---

## What You'll Learn
By completing this tutorial, you'll gain experience with:
- **Packaging a trained model for serving**: adding a serving function so the model accepts raw image bytes, not preprocessed tensors.
- **The Model Registry**: keeping a named, versioned record of each model you might deploy.
- **Online prediction endpoints**: putting a model behind an address that apps can call.
- **Prebuilt serving containers**: letting Google supply the server that runs your model.
- **Cost awareness**: a deployed endpoint runs (and bills) until you undeploy it.

---

## Key Ideas Before You Start

### Model file vs. served model
A saved model is just files: weights plus the computation graph. To get a prediction you still need a program that loads those files, waits for requests, turns each request into model input, runs the model and sends the answer back. That program is the **model server**. Vertex AI provides one for TensorFlow models as a **prebuilt container**, so we only have to provide the model files.

### Why we add a serving function
Our trained model expects a batch of 224 × 224 × 3 float images. An app, however, has a JPEG file. In `prepare()` we save the model with a **serving signature** that takes JPEG bytes, decodes them, converts them to floats, resizes them to 224 × 224 and only then calls the model. The preprocessing travels with the model, so every client gets exactly the same preprocessing that was used in training.

### Model Registry vs. Endpoint
- **Model Registry**: a catalog. Uploading a model there records *where its files are* and *which container serves it*. Nothing is running yet and it costs nothing.
- **Endpoint**: a running service. Deploying a registered model to an endpoint starts at least one machine that loads the model and answers requests. This is the part that costs money.

### Where things live
- **GCS bucket** (`GCS_MODELS_BUCKET_NAME`): holds the packaged model files.
- **Vertex AI Model Registry**: holds the model record that points to those files.
- **Vertex AI Endpoint**: the address your app calls.

---

## GCP Setup

### APIs to enable in GCP for your project
Search for each of these in the GCP search bar and click **Enable**:
* Agent Platform API (formerly "Vertex AI API"; service name `aiplatform.googleapis.com`)
* Cloud Storage API (usually already enabled)

Or from a terminal with gcloud: `gcloud services enable aiplatform.googleapis.com storage.googleapis.com`

### Create a local **secrets** folder

We do not want any secrets in Git, so we keep them **outside** the repository. At the same level as the `model-deployment` folder, create a folder called **secrets**:
```
   |-model-deployment
   |-secrets
```

### Create a GCP service account
- Go to the [GCP Console](https://console.cloud.google.com/home/dashboard), search for "Service accounts" in the top search box (or go to "IAM & Admin" > "Service accounts") and create a service account called `model-deployment`.
- For "Service account permissions" select **Storage Admin** and **Vertex AI Administrator** (`roles/aiplatform.admin`; the console may show it under the Agent Platform name, so search for `aiplatform.admin` if you cannot find it).
- In the "Actions" column, click the vertical ⋮ and select "Manage keys". Click "Add key" > "Create new key", choose "JSON" and click "Create". A JSON key file downloads to your computer.
- Move the file into the **secrets** folder and rename it to `model-deployment.json`.
- If "Create new key" is greyed out or fails with a policy error, your organization blocks service-account key creation (the default for new Google Cloud organizations). Use a personal project, or ask your organization admin to allow key creation for this project.

### Create a GCS bucket
We need a bucket to hold the packaged model that Vertex AI will load.

- Go to `https://console.cloud.google.com/storage/browser`
- Create a bucket, for example `cheese-app-models-<your-name>` (bucket names are global, so make it unique), in region `us-central1`.

### Point the scripts at your project
Open `docker-shell.sh` and set:
```bash
export GCP_PROJECT="<your-project-id>"
export GCS_MODELS_BUCKET_NAME="<your-bucket-name>"
```

---

## 🚀 Time to Build

### Step 1: Start the container
From inside the `model-deployment` folder:
```bash
sh docker-shell.sh
```
The container authenticates with your service account key and drops you into a shell with all dependencies installed. The image is built for your machine's architecture, so the same script works on Apple-silicon and Intel Macs, Windows and Linux.

### Step 2: Prepare the model
```bash
python cli.py --prepare
```
This downloads the trained model, adds the serving function described above and saves the packaged model to `gs://<your-bucket>/model-mobilenetv2_train_base_True.v1`.

> The weights were originally tracked in WandB under Pavlos' account. So that everyone can run this, the model is downloaded from GitHub instead; the WandB code is still in `cli.py`, commented out, if you want to pull your own model.

### Step 3: Upload and deploy
```bash
python cli.py --deploy
```
This first uploads the model to the **Model Registry**, then deploys it to a new **Endpoint**. It takes several minutes (often 10–20) while Google provisions a machine. When it finishes, the endpoint name is printed, in the form:
```
projects/<PROJECT_NUMBER>/locations/us-central1/endpoints/<ENDPOINT_ID>
```
You can also find it in the console under Agent Platform (Vertex AI) > Online prediction > Endpoints.

### Step 4: Predict
- Open `cli.py` and replace the endpoint name in `predict()` with **your** endpoint.
- Run:
```bash
python cli.py --predict
```
It sends five random images from `data/` to the endpoint. For each one you should see the class probabilities and the predicted label, similar to this (your numbers will differ):
```
Image: data/brie-1.jpg
[0.0213, 0.0402, 0.9301, 0.0084] 2
Label:    brie
```
The four probabilities are in the order parmigiano, gruyere, brie, gouda.

### Step 5: Clean up (important)
A deployed endpoint keeps a machine running and **bills your project every hour**, even when nobody calls it. When you are done:
- In the console, go to Agent Platform (Vertex AI) > Online prediction > Endpoints, open your endpoint, **undeploy** the model, then **delete** the endpoint.
- Or from the container:
```bash
gcloud ai endpoints list --region=us-central1
gcloud ai endpoints undeploy-model <ENDPOINT_ID> --region=us-central1 --deployed-model-id=<DEPLOYED_MODEL_ID>
gcloud ai endpoints delete <ENDPOINT_ID> --region=us-central1
```
The model in the Model Registry and the files in your bucket do not cost compute, so you can keep them.

---

## Quick Reference

```bash
sh docker-shell.sh            # start the container
python cli.py --prepare       # package the model and save it to GCS
python cli.py --deploy        # upload to Model Registry and deploy to an Endpoint
python cli.py --predict       # send test images to the Endpoint (set your endpoint in cli.py first)
```
Then **undeploy and delete the endpoint** when you are done.

---

## Appendix

### Notes for running on Windows
* Run Docker commands from **Git BASH**.
* **Always quote `$(pwd)`**: Windows paths often contain spaces.
* If you see `the input device is not a TTY`, prefix the `docker run` command in `docker-shell.sh` with `winpty`.

### Learn more
- [Deploy a model to an endpoint](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/deployment)
- [Prebuilt containers for prediction](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers)
- [Model Registry](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/model-registry/versioning)
