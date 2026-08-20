"""Self-financing accounting and transaction-cost utilities.

The finance layer converts pathwise hedge positions into discrete trading gains
and then combines those gains with the derivative liability to obtain terminal
PnL samples for evaluation or, later, risk minimization.
"""

from .costs import (
    compute_proportional_transaction_cost,
    compute_proportional_transaction_cost_from_config,
)
from .pnl import (
    compute_portfolio_pnl,
    compute_portfolio_pnl_from_config,
    compute_terminal_pnl,
    compute_terminal_pnl_from_config,
    compute_trading_gain,
)

__all__ = [
    "compute_portfolio_pnl",
    "compute_portfolio_pnl_from_config",
    "compute_proportional_transaction_cost",
    "compute_proportional_transaction_cost_from_config",
    "compute_terminal_pnl",
    "compute_terminal_pnl_from_config",
    "compute_trading_gain",
]