# S3 bucket for hosting documentation
resource "aws_s3_bucket" "docs" {
  bucket = "${var.domain_name}-python-docs"
}

# Enable website hosting
resource "aws_s3_bucket_website_configuration" "docs" {
  bucket = aws_s3_bucket.docs.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "404.html"
  }
}

# Create ACM certificate
resource "aws_acm_certificate" "docs" {
  provider          = aws.us-east-1
  domain_name       = var.domain_name
  validation_method = "DNS"
}

# Output the validation details (you'll need these for Bluehost)
output "certificate_validation_records" {
  value = {
    for dvo in aws_acm_certificate.docs.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }
}

# Create CloudFront distribution
resource "aws_cloudfront_distribution" "docs" {
  depends_on = [aws_acm_certificate.docs]
  enabled             = true
  default_root_object = "index.html"

  origin {
    domain_name = aws_s3_bucket_website_configuration.docs.website_endpoint
    origin_id   = "S3-${aws_s3_bucket.docs.id}"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.docs.id}"
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.docs.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  aliases = [var.domain_name]
}

# Bucket policy to allow public read access
resource "aws_s3_bucket_policy" "docs" {
  bucket = aws_s3_bucket.docs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.docs.arn}/python/*"
      },
    ]
  })
}

# Enable static website hosting
resource "aws_s3_bucket_public_access_block" "docs" {
  bucket = aws_s3_bucket.docs.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}
