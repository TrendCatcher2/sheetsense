#!/usr/bin/env bash
# Remove everything deploy.sh created.
set -uo pipefail
REGION="${AWS_REGION:-ap-south-1}"; FN="${FN:-sheetsense}"; ROLE="${FN}-lambda-role"
export AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${FN}-artifacts-${ACCOUNT}-${REGION}"
aws lambda delete-function-url-config --function-name "$FN" 2>/dev/null
aws lambda delete-function --function-name "$FN" 2>/dev/null
aws iam detach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null
aws iam delete-role --role-name "$ROLE" 2>/dev/null
aws cloudwatch delete-dashboards --dashboard-names "$FN" 2>/dev/null
aws logs delete-log-group --log-group-name "/aws/lambda/$FN" 2>/dev/null
aws s3 rb "s3://$BUCKET" --force 2>/dev/null
echo "removed $FN from $REGION"
