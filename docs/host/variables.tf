variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "domain_name" {
  description = "Domain name for the documentation"
  type        = string
  default     = "docs.frequensol.com"
}

variable "github_repository" {
  description = "GitHub repository in format 'owner/repo'"
  type        = string
  default     = "FrequenSol/FrequenSolve"
}
