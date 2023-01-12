# contents of wallet.feature
Feature: A feature

  Scenario: Wallet EUR amount stays constant
    Given I have 10 EUR in my wallet
    And I have a wallet
    Then I should have 10 EUR in my wallet

  Scenario: Second wallet JPY amount stays constant
    Given I have 100 JPY in my second wallet
    And I have a second wallet
    Then I should have 100 JPY in my second wallet