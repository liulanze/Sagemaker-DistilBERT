# News Classification API

A complete end-to-end news classification system using DistilBERT, SageMaker, Lambda, and API Gateway.

## Architecture

```
Client → API Gateway → Lambda → SageMaker Endpoint → DistilBERT Model
```

## Components

1. **Training**: `script.py` - Fine-tunes DistilBERT on news data
2. **Inference**: `inference.py` - Handles model loading and prediction
3. **Deployment**: `Deployment.ipynb` - Deploys model to SageMaker endpoint
4. **Lambda**: `aws-lambda-llm-endpoint-invoke-function.py` - API handler
5. **API Gateway**: `template.yaml` - REST API infrastructure

## Setup

### Prerequisites
- AWS CLI configured (`aws configure`)
- AWS SAM CLI installed (`pip install aws-sam-cli`)
- SageMaker endpoint deployed and running

### Deploy the API

1. **Deploy SageMaker Model** (if not done):
   ```bash
   # Run the Deployment.ipynb notebook first
   ```

2. **Deploy API Gateway + Lambda**:
   ```bash
   ./deploy.sh
   ```

3. **Test the API**:
   ```bash
   # Get the API URL from deployment output, then:
   python test_api.py <API_URL>
   
   # Or test specific headline:
   python test_api.py <API_URL> "Stock market crashes due to inflation"
   ```

## API Usage

### Endpoint
```
POST https://{api-id}.execute-api.{region}.amazonaws.com/prod/classify
```

### Request Format
```json
{
  "query": {
    "headline": "Scientists discover new treatment for cancer"
  }
}
```

### Response Format
```json
{
  "predicted_label": "Health",
  "probabilities": [[0.05, 0.10, 0.05, 0.80]]
}
```

### Categories
- **Business** (index 0)
- **Science** (index 1) 
- **Entertainment** (index 2)
- **Health** (index 3)

## Example cURL
```bash
curl -X POST https://your-api-url/prod/classify \
     -H "Content-Type: application/json" \
     -d '{"query": {"headline": "New vaccine shows 95% effectiveness"}}'
```

## Files Overview

- `script.py` - Training script for SageMaker
- `inference.py` - Model inference logic
- `Deployment.ipynb` - SageMaker deployment notebook
- `aws-lambda-llm-endpoint-invoke-function.py` - Lambda function
- `template.yaml` - SAM infrastructure template
- `deploy.sh` - Deployment script
- `test_api.py` - API testing script

## Troubleshooting

1. **Lambda Timeout**: Increase timeout in `template.yaml`
2. **Permissions Error**: Check IAM roles for SageMaker access
3. **Endpoint Not Found**: Verify SageMaker endpoint name matches
4. **CORS Issues**: API Gateway CORS is pre-configured

## Cost Considerations

- **SageMaker Endpoint**: Runs continuously (~$100-200/month for ml.m5.xlarge)
- **Lambda**: Pay per request (~$0.0000002 per request)
- **API Gateway**: Pay per request (~$0.0000035 per request)

Consider using SageMaker Serverless Inference for lower costs with variable traffic.
