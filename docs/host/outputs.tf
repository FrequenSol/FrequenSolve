output "cloudfront_domain" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.docs.domain_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.docs.id
}

output "website_endpoint" {
  description = "S3 static website hosting endpoint"
  value       = aws_s3_bucket_website_configuration.docs.website_endpoint
}

output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.docs.id
}
