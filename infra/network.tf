# The subnet is named by the owner (`var.subnet_id`) and is not looked up, so
# the choice cannot move when the account's subnets change: a moved choice
# would have Terraform propose replacing the machine and its data volume. The
# VPC and the data volume's zone are the subnet's.
#
# The two conditions are what the rest of this directory assumes. The machine
# reaches the package archives at first boot through the public address the
# subnet gives it, before the fixed address is associated; and nothing here
# is to have an IPv6 address.
data "aws_subnet" "server" {
  id = var.subnet_id

  lifecycle {
    postcondition {
      condition     = self.map_public_ip_on_launch
      error_message = "The subnet must give a public IPv4 address at launch: the first boot needs the network before the fixed address is associated."
    }

    postcondition {
      condition     = !self.assign_ipv6_address_on_creation
      error_message = "The subnet must not assign IPv6 addresses."
    }
  }
}

# The rules are written inside the group, so that Terraform owns the whole
# list and removes a rule added by any other means. Do not add a separate
# rule resource for this group: the tests read the list below, the provider
# does not support mixing the two forms, and `tests/inventory.sh` refuses
# both kinds of rule resource and any second group.
#
# No port 22: the machine is reached through Systems Manager. No IPv6: the
# proxy is published on IPv4 only (see `.env.example`).
resource "aws_security_group" "server" {
  name        = "causalpulse-server"
  description = "CausalPulse: HTTP and HTTPS in over IPv4, everything out"
  vpc_id      = data.aws_subnet.server.vpc_id

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
