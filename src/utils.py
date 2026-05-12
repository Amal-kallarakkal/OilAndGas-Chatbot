from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Tuple


def resolve_date_range(expression: str) -> Tuple[date, date]:
    """
    Resolve a natural language date expression to a (start, end) tuple.

    NOTE:
    The dataset only contains data up to 2024, so we anchor "today"
    to 2024-12-31 instead of the system clock.

    Supported expressions:
        'last_N_days'   -> today - N days to today
        'last_month'    -> first to last day of previous calendar month
        'last_quarter'  -> first to last day of previous quarter
        'last_N_months' -> today - N months to today
        'ytd'           -> Jan 1 of current year to today
    """

    # Anchor "today" to the dataset end date
    today = date(2024, 12, 31)

    expr = expression.lower().strip()

    if expr.startswith('last_') and expr.endswith('_days'):
        n = int(expr.split('_')[1])
        return today - timedelta(days=n), today

    if expr.startswith('last_') and expr.endswith('_months'):
        n = int(expr.split('_')[1])
        return today - relativedelta(months=n), today

    if expr == 'last_month':
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, last_month_end

    if expr == 'last_quarter':
        q = (today.month - 1) // 3
        if q == 0:
            q, year = 4, today.year - 1
        else:
            year = today.year
        q_start = date(year, (q - 1) * 3 + 1, 1)
        q_end = date(year, q * 3, 1) + relativedelta(months=1) - timedelta(days=1)
        return q_start, q_end

    if expr == 'ytd':
        return date(today.year, 1, 1), today
    if expr == "all_time":
        # dataset covers 2023–2024
        return date(2023, 1, 1), date(2024, 12, 31)

    raise ValueError(f'Unrecognised date expression: "{expression}"')