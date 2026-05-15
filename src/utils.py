from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Tuple, Union, Optional, List
import json




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


def parse_list_param(value: Union[list, str, None],
                      param_name: str = 'parameter') -> list:
    """
    Coerce an LLM tool parameter to a Python list.

    Some NVIDIA (and other) models serialise list arguments as JSON
    strings rather than native JSON arrays. For example, the LLM may
    send '["A001-W01"]' (string) instead of ['A001-W01'] (list).
    This guard handles both cases safely.

    Args:
        value:      The raw value received from the LLM tool call.
        param_name: Name of the parameter, used in error messages.

    Returns:
        A Python list. Never None.

    Raises:
        ValueError: If value is None or empty after coercion.
    """
    if value is None:
        raise ValueError(f'{param_name} is required and cannot be None.')

    if isinstance(value, list):
        return [str(v).strip() for v in value if v]

    if isinstance(value, str):
        stripped = value.strip()
        # Try JSON parse first: handles '["A001-W01"]'
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
            return [str(parsed).strip()]
        except json.JSONDecodeError:
            pass
        # Fallback: comma-separated string 'A001-W01, A001-W02'
        return [v.strip() for v in stripped.split(',') if v.strip()]

    # Any other type: wrap it
    return [str(value)]


def parse_optional_list(value: Union[list, str, None]) -> Optional[list]:
    """Same as parse_list_param but returns None if value is None/empty."""
    if value is None or value == '' or value == [] or value == '[]':
        return None
    result = parse_list_param(value, 'optional_list')
    return result if result else None

