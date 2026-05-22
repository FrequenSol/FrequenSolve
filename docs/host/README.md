# Documentation Hosting Infrastructure

> Deprecated: this legacy Terraform stack no longer owns the public Python docs
> deployment. The active publishing path is the `FrequenSol/cloud-amplify`
> `docs-site-app` and its manual `Publish Python Docs` workflow, which publishes
> versioned generated docs artifacts under `/python/<version>/` and
> `/python/latest/` on the main docs domain.

This directory contains the Terraform configuration for hosting the FrequenSolve Python documentation on AWS.

The hosted docs can be found at https://docs.frequensol.com/python/.

## Infrastructure Overview

The documentation is hosted using:
- Amazon S3 for storage
- CloudFront for content delivery and HTTPS
- ACM for SSL certificate
- IAM roles and OIDC for GitHub Actions deployment
- DNS managed through Bluehost

### Architecture

User → docs.frequensol.com/python/ → CloudFront → S3 bucket

## Files

- `main.tf`: Main Terraform configuration (S3, CloudFront, ACM)
- `iam.tf`: IAM roles and policies for GitHub Actions deployment
- `outputs.tf`: Output variables (domain names, bucket name, role ARN)
- `providers.tf`: AWS provider configuration
- `variables.tf`: Input variables
- `backend.tf`: Terraform state configuration

## Setup Instructions

1. **Prerequisites**:
   - AWS CLI configured
   - Terraform installed and initialized (`terraform init`)
   - Access to Bluehost DNS settings
   - GNU Make installed
   - GitHub repository access

2. **Apply Configuration**:
   ```bash
   make apply
   ```

3. **Configure GitHub Actions**:
   - Get the IAM role ARN from Terraform output:
     ```bash
     terraform output github_actions_role_arn
     ```
   - Add these secrets to your GitHub repository:
     - `AWS_ROLE_ARN`: The role ARN from above
     - `AWS_REGION`: "us-east-1" (or your configured region)

4. **DNS Configuration in Bluehost**:
   - Add ACM validation CNAME record (provided by Terraform output)
   - Add CNAME record pointing docs.frequensol.com to CloudFront domain

## Deployment

Documentation is automatically deployed when:
- Changes are pushed to the main branch
- The GitHub Actions workflow runs successfully

Manual deployment is also possible:
```bash
make deploy-all  # Deploys and invalidates cache
```

## Security

- HTTPS enabled by default
- S3 bucket accessed only through CloudFront
- GitHub Actions uses OIDC for secure AWS authentication
- Least-privilege IAM policies
- Content served via secure CloudFront endpoints

## IAM Configuration

The setup includes:
- OIDC provider for GitHub Actions
- IAM role with permissions for:
  - S3 operations (PutObject, GetObject, ListBucket, DeleteObject)
  - CloudFront invalidations
- Trust relationship limited to specific GitHub repository

## Troubleshooting

1. **Deployment Issues**:
   - Check GitHub Actions logs
   - Verify AWS role ARN and region are correctly set
   - Ensure IAM role has correct permissions

2. **Access Issues**:
   - Check DNS propagation: `dig docs.frequensol.com`
   - Verify CloudFront distribution status
   - Clear browser cache or try private window

3. **Content Not Updating**:
   - CloudFront cache might need invalidation
   - Verify S3 upload was successful
   - Check GitHub Actions workflow logs

## Cost Estimation

Typical monthly costs for low-traffic documentation:
- S3: ~$0.02 (storage)
- CloudFront: ~$0.17 (data transfer)
- Total: < $1/month

## Resources

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [CloudFront Documentation](https://docs.aws.amazon.com/cloudfront/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Terraform Documentation](https://www.terraform.io/docs)
