#!/bin/bash

# AWS SAM deployment script for News Classification API

echo "🚀 Deploying News Classification API..."

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo "❌ AWS SAM CLI is not installed. Please install it first:"
    echo "   pip install aws-sam-cli"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

# Build the SAM application
echo "📦 Building SAM application..."
sam build

# Deploy the application
echo "🚀 Deploying to AWS..."
sam deploy --guided

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Note the API endpoint URL from the outputs above"
echo "2. Test your API with:"
echo "   curl -X POST [API_URL] \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"query\": {\"headline\": \"Stock market crashes\"}}'"
echo ""
echo "3. Make sure your SageMaker endpoint is deployed and running!"
