# A warning by e-mail at 80% of 50 US dollars in a month, once on what has
# been spent and once on what the month is forecast to cost.
#
# It counts the whole account, which this project shares, and so it is not a
# measure of this project. If the other project's spending alone passes 40
# dollars in a month, both warnings are sent whatever this project costs, and
# once sent they say nothing more that month. A filter by the `Project` tag
# would need the tag switched on for billing by hand, and would count nothing
# until then. Left as it is, knowingly.
resource "aws_budgets_budget" "monthly" {
  name         = "causalpulse-monthly"
  budget_type  = "COST"
  limit_amount = "50"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
