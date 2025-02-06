# Create S3 bucket for Terraform state
resource "aws_s3_bucket" "terraform_state" {
  bucket = "docs.frequensol.com-python-terraform-state"
}

# Enable versioning for state files
resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Create DynamoDB table for state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "docs-frequensol-com-python-terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
