# End-to-End Pipeline Setup Guide

This document outlines all prerequisites and step-by-step instructions required to run the complete Entity Resolution pipeline (`main.ipynb`) end-to-end on your local machine.

Because the pipeline is a hybrid architecture, lightweight phases (like Data Exploration, Feature Engineering, and LightGBM training) run locally on your CPU, while heavy phases (like BGE-m3 embedding, FAISS indexing, and Cross-Encoder fine-tuning) are dispatched to AWS SageMaker Training Jobs.

---

## 1. System & Account Prerequisites

1. **Local Machine:** 
   - **Python:** Python 3.10 or higher must be installed.
   - **Storage:** At least 16GB of free disk space is recommended to hold the downloaded embedding shards and temporary outputs.
   - **RAM:** A standard laptop (e.g., 8GB-16GB RAM) is sufficient. The memory-intensive FAISS index build (~64GB RAM) is offloaded to AWS.
   - **Internet Connection:** An active, unrestricted internet connection is required to download Python packages and communicate with AWS.

2. **AWS Account:** 
   - You need an active AWS account.
   - Your account must have access to `ml.g4dn` instance types (see Step 7 below).

---

## 2. Set up AWS IAM Credentials

You need explicit IAM credentials to authorize your laptop to launch SageMaker jobs and access S3.

1. Log into the [AWS Management Console](https://console.aws.amazon.com/).
2. Navigate to the **IAM** (Identity and Access Management) dashboard.
3. Click **Users** -> **Create user**. Name it something descriptive, like `er-pipeline-user`.
4. On the permissions page, choose **Attach policies directly**.
5. Search for and select the following two managed policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
6. Click **Next** and then **Create user**.
7. Click on your newly created user, navigate to the **Security credentials** tab, scroll down to **Access keys**, and click **Create access key**.
8. Choose **Command Line Interface (CLI)**, click **Next**, and click **Create access key**.
9. **CRITICAL:** Copy the **Access key ID** and **Secret access key** shown on the screen. Store them securely; you will need them in Step 4.

---

## 3. Install Local Dependencies

1. **Install the AWS CLI:** 
   - Download and install the AWS Command Line Interface for Windows from the [official AWS site](https://awscli.amazonaws.com/AWSCLIV2.msi).
   - Follow the standard installation prompts.

2. **Install Python Packages:** 
   - Open a fresh Command Prompt or PowerShell in the `Experiment_Day1/` folder.
   - Run the following command to install all required libraries (ensure your internet connection is active and not blocking `pypi.org`):
   ```bash
   pip install lightgbm>=4.0 faiss-cpu>=1.7.4 sentence-transformers>=2.7 transformers>=4.40 torch>=2.1 rapidfuzz>=3.0 pandas>=2.0 numpy>=1.24 scikit-learn>=1.3 ftfy>=6.1 indic-transliteration>=2.3 boto3>=1.34 sagemaker>=2.200 pyarrow fastparquet
   ```

---

## 4. Configure AWS Credentials Locally

Now, link your laptop to your AWS account using the AWS CLI.

1. Open Command Prompt or PowerShell and type:
   ```bash
   aws configure
   ```
2. You will be prompted for four pieces of information. Fill them out as follows:
   - **AWS Access Key ID:** (Paste the key copied in Step 2)
   - **AWS Secret Access Key:** (Paste the secret copied in Step 2)
   - **Default region name:** `us-east-1` *(Recommendation: use us-east-1 as it generally has the best GPU availability).*
   - **Default output format:** `json`

---

## 5. Create Your S3 Bucket

SageMaker requires an S3 bucket to store the dataset, model artifacts, and embedding shards.

1. Create a bucket using the AWS CLI. Run the following command, replacing `yourname123` with a globally unique identifier:
   ```bash
   aws s3 mb s3://er-challenge-yourname123 --region us-east-1
   ```

---

## 6. Update `main.ipynb` Configuration

You must update the main notebook to point to your newly created S3 bucket.

1. Open `main.ipynb` in your Jupyter environment.
2. Scroll to **Phase 3** (around cell 10).
3. Locate the configuration block:
   ```python
   # ---- Configure these once ------------------------------------------------
   S3_BUCKET = 'er-challenge-<your-name>'   # replace with your bucket name
   S3_PREFIX = 'er-challenge'
   REGION    = 'us-east-1'
   # --------------------------------------------------------------------------
   ```
4. Change the `S3_BUCKET` string to match the exact bucket name you created in Step 5 (e.g., `'er-challenge-yourname123'`).

---

## 7. Check AWS Service Quotas (Crucial for New Accounts)

New AWS accounts often have a quota of **0** for GPU training instances to prevent accidental spending. You must ensure you have quota for the required instances.

1. Go to the AWS Console and search for **Service Quotas**.
2. On the left sidebar, click **AWS Services** -> **Amazon SageMaker**.
3. Search for the following two quotas:
   - `ml.g4dn.xlarge for training job usage`
   - `ml.g4dn.4xlarge for training job usage`
4. Ensure the **Applied quota value** is at least `1` for both.
5. If either is `0`, select it and click **Request quota increase**. Request an increase to `1`. *(Note: AWS usually approves these requests within 15-30 minutes).*

---

## 8. Ensure Dataset Directory Structure

Ensure the raw `.tsv` files are extracted and located in the correct directory relative to the `Experiment_Day1/` folder. The structure must look exactly like this:

```
hackathon/Amazon_ML_Challenge/
├── Experiment_Day1/
│   ├── main.ipynb
│   ├── (other scripts...)
└── student_resource/
    └── dataset/
        ├── train/
        │   ├── train_source1.tsv
        │   ├── train_source2.tsv
        │   ├── train_source3.tsv
        │   └── train_ground_truth.tsv
        └── test/
            ├── test_source1.tsv
            ├── test_source2.tsv
            └── test_source3.tsv
```

---

## 9. Running the Pipeline

Once all the above steps are completed, you are ready to execute the pipeline.

1. Open `main.ipynb`.
2. Run the cells in order from top to bottom.
3. The notebook will automatically handle zipping your local code, uploading the datasets to your S3 bucket, dispatching the SageMaker jobs, and downloading the results back to your laptop's `output/` folder.

> [!WARNING] 
> **Cost Estimates:** 
> The AWS SageMaker jobs (Phases 3, 4, and 7) will incur charges against your AWS account. The total estimated GPU cost for a full run is **~$22.89**. Ensure you monitor your billing dashboard.
