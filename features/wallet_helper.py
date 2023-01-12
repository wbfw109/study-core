"""A feature feature tests."""

from pytest_bdd import (
    given,
    scenario,
    then,
    when,
)


@scenario("wallet.feature", "Second wallet JPY amount stays constant")
def test_second_wallet_jpy_amount_stays_constant():
    """Second wallet JPY amount stays constant."""


@scenario("wallet.feature", "Wallet EUR amount stays constant")
def test_wallet_eur_amount_stays_constant():
    """Wallet EUR amount stays constant."""


@given("I have 10 EUR in my wallet")
def _():
    """I have 10 EUR in my wallet."""
    pass


@given("I have 100 JPY in my second wallet")
def _():
    """I have 100 JPY in my second wallet."""
    pass


@given("I have a second wallet")
def _():
    """I have a second wallet."""
    pass


@given("I have a wallet")
def _():
    """I have a wallet."""
    pass


@then("I should have 10 EUR in my wallet")
def _():
    """I should have 10 EUR in my wallet."""
    pass


@then("I should have 100 JPY in my second wallet")
def _():
    """I should have 100 JPY in my second wallet."""
    pass
