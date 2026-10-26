#!/usr/bin/env bash
# Build and deploy SheetSense to AWS Lambda (arm64 / Graviton) behind a public Function URL.
# Idempotent: re-running updates the code and configuration in place.
#
#   AWS_REGION=ap-south-1 ./deploy/deploy.sh
#
# Needs: aws CLI v2 or v1 with credentials, python3, pip, zip.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
FN="${FN:-sheetsense}"
ROLE="${FN}-lambda-role"
RUNTIME="${RUNTIME:-python3.14}"
PYVER="${RUNTIME#python}"
MEMORY="${MEMORY:-2048}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
export AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${FN}-artifacts-${ACCOUNT}-${REGION}"
echo "account=$ACCOUNT region=$REGION function=$FN runtime=$RUNTIME"

# ---------------------------------------------------------------- package
rm -rf "$BUILD" && mkdir -p "$BUILD/pkg"
python3 -m pip install --quiet --target "$BUILD/pkg" \
  --platform manylinux_2_28_aarch64 --implementation cp --python-version "$PYVER" \
  --only-binary=:all: -r "$ROOT/requirements-lambda.txt"
cp -r "$ROOT/sheetsense" "$ROOT/app" "$BUILD/pkg/"
rm -f "$BUILD/pkg/app/local_server.py"
find "$BUILD/pkg" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -rf "$BUILD/pkg/numpy/"*/tests "$BUILD/pkg/cv2/data"
(cd "$BUILD/pkg" && zip -q -r -9 "$BUILD/lambda.zip" .)
echo "package: $(du -h "$BUILD/lambda.zip" | cut -f1) zipped, $(du -sh "$BUILD/pkg" | cut -f1) unzipped"

# ---------------------------------------------------------------- artifact bucket (private, encrypted)
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
KEY="lambda/${FN}-$(date +%Y%m%d%H%M%S).zip"
aws s3 cp --quiet "$BUILD/lambda.zip" "s3://$BUCKET/$KEY"

# ---------------------------------------------------------------- execution role (logs only)
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "waiting for the new role to propagate..." && sleep 12
fi
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"

# ---------------------------------------------------------------- function
aws logs create-log-group --log-group-name "/aws/lambda/$FN" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "/aws/lambda/$FN" --retention-in-days 14

if aws lambda get-function --function-name "$FN" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FN" --s3-bucket "$BUCKET" --s3-key "$KEY" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FN"
  aws lambda update-function-configuration --function-name "$FN" --runtime "$RUNTIME" --memory-size "$MEMORY" --timeout 30 >/dev/null
else
  aws lambda create-function --function-name "$FN" --runtime "$RUNTIME" --architectures arm64 \
    --handler app.handler.lambda_handler --role "$ROLE_ARN" --memory-size "$MEMORY" --timeout 30 \
    --code "S3Bucket=$BUCKET,S3Key=$KEY" --description "SheetSense: OpenCV 5 MCQ grading agent" >/dev/null
fi
aws lambda wait function-updated-v2 --function-name "$FN" 2>/dev/null || aws lambda wait function-active-v2 --function-name "$FN"

# Cost / abuse cap. New accounts may not allow reserved concurrency; the account limit then applies.
aws lambda put-function-concurrency --function-name "$FN" --reserved-concurrent-executions 5 >/dev/null 2>&1 \
  && echo "reserved concurrency: 5" || echo "reserved concurrency not set (account limit applies)"

# ---------------------------------------------------------------- public Function URL
if ! aws lambda get-function-url-config --function-name "$FN" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FN" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url --action lambda:InvokeFunctionUrl \
    --principal "*" --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url-invoke --action lambda:InvokeFunction \
    --principal "*" --invoked-via-function-url >/dev/null 2>&1 \
    || echo "WARNING: could not add lambda:InvokeFunction for the URL; if the URL returns 403, update the AWS CLI and re-run"
fi
URL="$(aws lambda get-function-url-config --function-name "$FN" --query FunctionUrl --output text)"

# ---------------------------------------------------------------- dashboard
sed -e "s/__REGION__/$REGION/g" -e "s/__FN__/$FN/g" "$ROOT/deploy/dashboard.json" > "$BUILD/dashboard.json"
aws cloudwatch put-dashboard --dashboard-name "$FN" --dashboard-body "file://$BUILD/dashboard.json" >/dev/null

echo "health: $(curl -s -o /dev/null -w '%{http_code}' "${URL}api/health") $(curl -s "${URL}api/health")"
echo
echo "Deployed: $URL"
echo "Dashboard: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards/dashboard/$FN"
