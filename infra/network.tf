# The account's default VPC, and the default subnet whose id sorts first.
# The choice is stable for as long as the VPC's default subnets are the same
# ones; the data volume is created in that subnet's zone.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_subnet" "server" {
  id = sort(data.aws_subnets.default.ids)[0]
}

# The rules are written inside the group, so that Terraform owns the whole
# list and removes a rule added by any other means. Do not add a separate
# rule resource for this group: the tests read the list below, and the
# provider does not support mixing the two forms.
#
# No port 22: the machine is reached through Systems Manager. No IPv6: the
# proxy is published on IPv4 only (see `.env.example`).
resource "aws_security_group" "server" {
  name        = "causalpulse-server"
  description = "CausalPulse: HTTP and HTTPS in over IPv4, everything out"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP, for the certificate challenge and the redirect"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Everything out, over IPv4"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "causalpulse-server"
  }
}

# A fixed public address, which the site's A record names. It outlives a
# stop, a start and a replacement of the instance.
resource "aws_eip" "server" {
  domain = "vpc"

  tags = {
    Name = "causalpulse-server"
  }
}

resource "aws_eip_association" "server" {
  allocation_id = aws_eip.server.id
  instance_id   = aws_instance.server.id
}
