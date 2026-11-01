CONSTRAINTS = [
    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE",
    "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
    "CREATE CONSTRAINT branch_id IF NOT EXISTS FOR (b:Branch) REQUIRE b.branch_id IS UNIQUE",
    "CREATE CONSTRAINT alert_id IF NOT EXISTS FOR (al:Alert) REQUIRE al.alert_id IS UNIQUE",
    "CREATE CONSTRAINT phone_number IF NOT EXISTS FOR (p:Phone) REQUIRE p.number IS UNIQUE",
    "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE",
    "CREATE CONSTRAINT address_text IF NOT EXISTS FOR (ad:Address) REQUIRE ad.text IS UNIQUE",
    "CREATE INDEX transfer_txn_id IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.txn_id)",
    "CREATE INDEX transfer_timestamp IF NOT EXISTS FOR ()-[t:TRANSFER]-() ON (t.timestamp)",
    "CREATE INDEX customer_name IF NOT EXISTS FOR (c:Customer) ON (c.name)",
]

# Served by the MCP server as a resource, so an LLM can learn the graph before querying it.
SCHEMA_DESCRIPTION = """# AML transaction graph (synthetic data)

## Nodes
- (:Customer {customer_id, name, customer_type: individual|business, occupation, state, country,
  risk_rating: low|medium|high, is_pep, onboarded_on, expected_monthly_volume})
- (:Account {account_id, account_type: checking|savings|business|external, opened_on, home_branch,
  bank, country, is_external, holder_name, tags})
  External accounts (IDs starting with X) belong to other banks: offshore companies, crypto
  exchanges (tags=crypto_exchange), title companies, foreign individuals.
- (:Branch {branch_id, city, state})
- (:Alert {alert_id, rule, score, priority: High|Medium|Low, status: open|escalated|closed,
  created_at, summary, txn_ids})
- (:Phone {number}), (:Device {device_id}), (:Address {text})
- (:Dataset {as_of, days, seed, transactions}) - one node; as_of is the "today" of the data

## Relationships
- (:Customer)-[:OWNS]->(:Account)
- (:Customer)-[:HAS_PHONE]->(:Phone), -[:USES_DEVICE]->(:Device), -[:REGISTERED_AT]->(:Address)
- (:Account|Branch)-[:TRANSFER {txn_id, amount, timestamp, channel, description}]->(:Account|Branch)
  channel: card | ach | p2p | wire | crypto | cash_deposit (Branch->Account) | cash_withdrawal (Account->Branch)
- (:Alert)-[:ON_ACCOUNT]->(:Account)

## Alert rules
STRUCTURING_CASH, FAN_IN_FAN_OUT, CIRCULAR_FLOW, MULTI_BRANCH_CASH, RAPID_MOVEMENT,
SHARED_IDENTIFIER, PEP_LARGE_WIRE, LARGE_WIRE
"""
