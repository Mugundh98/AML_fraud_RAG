"""Deterministic synthetic bank: customers, accounts, transfers and alerts, with planted AML typologies.

Normal activity (payroll, card spend, bills, P2P, cash) fills most of the transaction budget.
On top of that, a handful of scenarios plant the patterns described in the knowledge base:
structuring, money mules, circular flows, funnel accounts, pass-through wires, synthetic
identity rings and PEP wires. `scenarios.csv` is the answer key and is NOT loaded into Neo4j.
"""

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
    "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Carlos", "Karen",
    "Daniel", "Lisa", "Matthew", "Nancy", "Anthony", "Sandra", "Priya", "Ashley", "Wei", "Fatima",
    "Andre", "Olga", "Kenji", "Amara", "Diego", "Mei", "Omar", "Ingrid", "Raj", "Lucia",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Patel", "Nguyen", "Kim",
    "Okafor", "Novak", "Chen", "Haddad", "Schmidt", "Rossi", "Silva", "Ivanova", "Tanaka", "Mensah",
]
BUSINESS_WORDS = [
    "Blue", "Harbor", "Summit", "Pioneer", "Granite", "Maple", "Silver", "Northern", "Cedar", "Atlas",
    "Evergreen", "Golden", "Riverside", "Lakeside", "Union", "Keystone", "Beacon", "Prairie", "Coastal", "Metro",
]
OCCUPATIONS = [
    "Software engineer", "Nurse", "Teacher", "Accountant", "Electrician", "Retail associate", "Driver",
    "Pharmacist", "Marketing manager", "Mechanic", "Chef", "Graphic designer", "Sales representative",
    "Paralegal", "Retired", "Physician", "Construction worker", "Real estate agent",
]
# Business categories and the role they play in normal activity.
BUSINESS_CATEGORIES = {
    "employer": ["Hospital", "Software", "School District", "Logistics", "Manufacturing"],
    "merchant": ["Grocery", "Pharmacy", "Electronics", "Apparel", "Hardware", "Fuel"],
    "utility": ["Power & Light", "Water Authority", "Telecom", "Insurance"],
    "supplier": ["Wholesale", "Packaging", "Food Distribution", "Industrial Supply"],
    "cash_intensive": ["Restaurant", "Laundromat", "Car Wash", "Convenience Store", "Bar & Grill"],
}
BRANCHES = [
    ("BR-01", "New York", "NY"), ("BR-02", "Brooklyn", "NY"), ("BR-03", "Newark", "NJ"),
    ("BR-04", "Philadelphia", "PA"), ("BR-05", "Boston", "MA"), ("BR-06", "Chicago", "IL"),
    ("BR-07", "Detroit", "MI"), ("BR-08", "Atlanta", "GA"), ("BR-09", "Miami", "FL"),
    ("BR-10", "Orlando", "FL"), ("BR-11", "Houston", "TX"), ("BR-12", "Dallas", "TX"),
    ("BR-13", "El Paso", "TX"), ("BR-14", "Phoenix", "AZ"), ("BR-15", "Tucson", "AZ"),
    ("BR-16", "Denver", "CO"), ("BR-17", "Los Angeles", "CA"), ("BR-18", "San Diego", "CA"),
    ("BR-19", "Seattle", "WA"), ("BR-20", "Las Vegas", "NV"),
]
BORDER_BRANCHES = ["BR-13", "BR-15", "BR-18"]
OFFSHORE_COUNTRIES = ["PA", "CY", "AE", "HK", "SC", "VG"]
FOREIGN_BANK_COUNTRIES = ["GB", "DE", "SG", "MX", "CA", "IN", "JP", "FR"]
STREETS = ["Oak", "Pine", "Elm", "Main", "Park", "Lake", "Hill", "River", "Washington", "Lincoln"]

CTR_THRESHOLD = 10_000
LARGE_WIRE_FALSE_POSITIVES = 3


@dataclass
class Customer:
    customer_id: str
    name: str
    customer_type: str  # individual | business
    occupation: str  # occupation, or business category for businesses
    state: str
    country: str
    risk_rating: str  # low | medium | high
    is_pep: bool
    onboarded_on: str
    expected_monthly_volume: int
    phone: str
    device_id: str
    address: str
    salary: int = field(default=0, repr=False)  # generator-only, not exported


@dataclass
class Account:
    account_id: str
    customer_id: str  # empty for external accounts
    account_type: str  # checking | savings | business | external
    opened_on: str
    home_branch: str
    bank: str
    country: str
    is_external: bool
    holder_name: str
    tags: str = ""  # e.g. crypto_exchange, offshore


@dataclass
class Transfer:
    from_id: str
    from_kind: str  # account | branch
    to_id: str
    to_kind: str  # account | branch
    amount: float
    timestamp: datetime
    channel: str  # card | ach | p2p | wire | cash_deposit | cash_withdrawal | crypto
    description: str
    txn_id: str = ""


@dataclass
class Alert:
    alert_id: str
    account_id: str
    rule: str
    score: int
    priority: str
    status: str
    created_at: datetime
    summary: str
    transfers: list[Transfer] = field(repr=False)


@dataclass
class Scenario:
    scenario_id: str
    typology: str
    customer_ids: list[str]
    account_ids: list[str]
    alerted: bool
    description: str


class SyntheticBank:
    def __init__(self, seed: int = 42, n_transactions: int = 20_000, as_of: date = date(2026, 10, 31), days: int = 180):
        self.rng = random.Random(seed)
        self.seed = seed
        self.n_transactions = n_transactions
        self.as_of = as_of
        self.days = days
        self.start = datetime.combine(as_of - timedelta(days=days - 1), time(0))

        self.customers: list[Customer] = []
        self.accounts: list[Account] = []
        self.transfers: list[Transfer] = []
        self.alerts: list[Alert] = []
        self.scenarios: list[Scenario] = []

        self._accounts_by_customer: dict[str, list[Account]] = {}
        self._businesses: dict[str, list[Account]] = {k: [] for k in BUSINESS_CATEGORIES}
        self._individual_accounts: list[Account] = []
        self._employer_of: dict[str, Account] = {}
        self._externals: dict[str, list[Account]] = {}

    # ---------- public API ----------

    def generate(self) -> "SyntheticBank":
        n_individuals = max(200, self.n_transactions // 20)
        for _ in range(n_individuals):
            self._new_individual()
        for category in BUSINESS_CATEGORIES:
            for _ in range(max(6, n_individuals // 40)):
                self._new_business(category)
        self._new_externals()
        for customer in self.customers:
            if customer.customer_type == "individual" and customer.occupation != "Retired":
                self._employer_of[customer.customer_id] = self.rng.choice(self._businesses["employer"])

        self._plant_scenarios()
        self._normal_activity(self.n_transactions - len(self.transfers) - LARGE_WIRE_FALSE_POSITIVES)
        self._false_positive_alerts()

        self.transfers.sort(key=lambda t: t.timestamp)
        for i, t in enumerate(self.transfers, start=1):
            t.txn_id = f"T{i:07d}"
        self.alerts.sort(key=lambda a: a.created_at)
        for i, a in enumerate(self.alerts, start=1):
            a.alert_id = f"AL-{i:04d}"
        return self

    def write_csv(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        customers = [{k: v for k, v in asdict(c).items() if k != "salary"} for c in self.customers]
        _write(out_dir / "customers.csv", customers)
        _write(out_dir / "accounts.csv", [asdict(a) for a in self.accounts])
        _write(out_dir / "branches.csv", [{"branch_id": b, "city": c, "state": s} for b, c, s in BRANCHES])
        _write(
            out_dir / "transfers.csv",
            [
                {
                    "txn_id": t.txn_id, "from_id": t.from_id, "from_kind": t.from_kind, "to_id": t.to_id,
                    "to_kind": t.to_kind, "amount": t.amount, "timestamp": t.timestamp.isoformat(),
                    "channel": t.channel, "description": t.description,
                }
                for t in self.transfers
            ],
        )
        _write(
            out_dir / "alerts.csv",
            [
                {
                    "alert_id": a.alert_id, "account_id": a.account_id, "rule": a.rule, "score": a.score,
                    "priority": a.priority, "status": a.status, "created_at": a.created_at.isoformat(),
                    "summary": a.summary, "txn_ids": ";".join(t.txn_id for t in a.transfers),
                }
                for a in self.alerts
            ],
        )
        _write(
            out_dir / "scenarios.csv",
            [
                {**asdict(s), "customer_ids": ";".join(s.customer_ids), "account_ids": ";".join(s.account_ids)}
                for s in self.scenarios
            ],
        )
        (out_dir / "dataset.json").write_text(
            json.dumps(
                {"seed": self.seed, "as_of": self.as_of.isoformat(), "days": self.days,
                 "transactions": len(self.transfers), "customers": len(self.customers),
                 "accounts": len(self.accounts), "alerts": len(self.alerts)},
                indent=2,
            )
        )

    # ---------- entities ----------

    def _new_individual(self, **overrides) -> Customer:
        rng = self.rng
        _, city, state = rng.choice([b for b in BRANCHES if b[2] == overrides.get("state")] or BRANCHES)
        customer = Customer(
            customer_id=f"C{len(self.customers) + 1:05d}",
            name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            customer_type="individual",
            occupation=rng.choice(OCCUPATIONS),
            state=state,
            country="US",
            risk_rating=rng.choices(["low", "medium", "high"], [80, 17, 3])[0],
            is_pep=False,
            onboarded_on=self._random_date(years_back=12).isoformat(),
            expected_monthly_volume=0,
            phone=self._phone(),
            device_id=self._device(),
            address=self._address(city, state),
        )
        customer.salary = rng.randrange(2_500, 9_500, 50)
        customer.expected_monthly_volume = int(customer.salary * 1.3)
        for key, value in overrides.items():
            setattr(customer, key, value)
        self.customers.append(customer)

        branch = self._branch_in(customer.state)
        accounts = [self._new_account(customer, "checking", branch)]
        if rng.random() < 0.35 and not overrides:
            accounts.append(self._new_account(customer, "savings", branch))
        self._individual_accounts.append(accounts[0])
        return customer

    def _new_business(self, category: str, **overrides) -> Customer:
        rng = self.rng
        _, city, state = rng.choice(BRANCHES)
        kind = rng.choice(BUSINESS_CATEGORIES.get(category, ["Trading"]))
        customer = Customer(
            customer_id=f"C{len(self.customers) + 1:05d}",
            name=f"{rng.choice(BUSINESS_WORDS)} {rng.choice(BUSINESS_WORDS)} {kind} {rng.choice(['LLC', 'Inc', 'Co'])}",
            customer_type="business",
            occupation=category,
            state=state,
            country="US",
            risk_rating="high" if category == "cash_intensive" else rng.choices(["low", "medium"], [75, 25])[0],
            is_pep=False,
            onboarded_on=self._random_date(years_back=15).isoformat(),
            expected_monthly_volume=rng.randrange(20_000, 400_000, 1_000),
            phone=self._phone(),
            device_id=self._device(),
            address=self._address(city, state),
        )
        for key, value in overrides.items():
            setattr(customer, key, value)
        self.customers.append(customer)
        account = self._new_account(customer, "business", self._branch_in(customer.state))
        if category in self._businesses:
            self._businesses[category].append(account)
        return customer

    def _new_account(self, customer: Customer, account_type: str, branch: str, opened_on: str | None = None) -> Account:
        account = Account(
            account_id=f"A{len(self.accounts) + 1:06d}",
            customer_id=customer.customer_id,
            account_type=account_type,
            opened_on=opened_on or customer.onboarded_on,
            home_branch=branch,
            bank="Example Bank",
            country="US",
            is_external=False,
            holder_name=customer.name,
        )
        self.accounts.append(account)
        self._accounts_by_customer.setdefault(customer.customer_id, []).append(account)
        return account

    def _new_external(self, group: str, holder: str, bank: str, country: str, tags: str = "") -> Account:
        account = Account(
            account_id=f"X{len(self.accounts) + 1:06d}",
            customer_id="",
            account_type="external",
            opened_on="",
            home_branch="",
            bank=bank,
            country=country,
            is_external=True,
            holder_name=holder,
            tags=tags,
        )
        self.accounts.append(account)
        self._externals.setdefault(group, []).append(account)
        return account

    def _new_externals(self) -> None:
        rng = self.rng
        for name in ["CoinVault Exchange", "BitHarbor Markets", "Ledgerline Crypto"]:
            self._new_external("crypto", name, f"{name} Custody", "US", "crypto_exchange")
        for _ in range(12):
            country = rng.choice(OFFSHORE_COUNTRIES)
            holder = f"{rng.choice(BUSINESS_WORDS)} {rng.choice(['Holdings', 'Trading', 'Ventures', 'Capital'])} Ltd"
            self._new_external("offshore", holder, f"{country} International Bank", country, "offshore")
        for _ in range(10):
            country = rng.choice(FOREIGN_BANK_COUNTRIES)
            holder = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            self._new_external("foreign_person", holder, f"{country} National Bank", country)
        for name in ["Secure Title & Escrow", "First Closing Services"]:
            self._new_external("title", name, "Domestic Partner Bank", "US", "title_company")

    # ---------- normal activity ----------

    def _normal_activity(self, budget: int) -> None:
        kinds = ["card", "p2p", "bill", "payroll", "cash_withdrawal", "cash_deposit", "business_cash", "supplier_wire"]
        weights = [44, 12, 12, 12, 7, 4, 5, 4]
        for _ in range(max(0, budget)):
            getattr(self, f"_normal_{self.rng.choices(kinds, weights)[0]}")()

    def _normal_card(self):
        payer = self.rng.choice(self._individual_accounts)
        merchant = self.rng.choice(self._businesses["merchant"] + self._businesses["cash_intensive"])
        amount = round(min(900, self.rng.lognormvariate(3.6, 0.8)), 2)
        self._add(payer, merchant, amount, self._ts(hours=(7, 22)), "card", f"POS PURCHASE {merchant.holder_name}")

    def _normal_p2p(self):
        a, b = self.rng.sample(self._individual_accounts, 2)
        self._add(a, b, round(self.rng.uniform(15, 600), 2), self._ts(), "p2p", "P2P TRANSFER")

    def _normal_bill(self):
        payer = self.rng.choice(self._individual_accounts)
        utility = self.rng.choice(self._businesses["utility"])
        self._add(payer, utility, round(self.rng.uniform(40, 380), 2), self._ts(hours=(0, 6)), "ach", f"BILL PAY {utility.holder_name}")

    def _normal_payroll(self):
        account = self.rng.choice(self._individual_accounts)
        employer = self._employer_of.get(account.customer_id)
        if employer is None:
            return self._normal_card()
        customer = self._customer(account.customer_id)
        day = self.rng.choice([d for d in range(self.days) if (self.start + timedelta(days=d)).day in (1, 15)])
        ts = self.start + timedelta(days=day, hours=6)
        self._add(employer, account, round(customer.salary / 2, 2), ts, "ach", f"PAYROLL {employer.holder_name}")

    def _normal_cash_withdrawal(self):
        account = self.rng.choice(self._individual_accounts)
        self._add_cash_out(account, account.home_branch, self.rng.randrange(40, 800, 20), self._ts(hours=(9, 17)))

    def _normal_cash_deposit(self):
        account = self.rng.choice(self._individual_accounts)
        self._add_cash_in(account.home_branch, account, round(self.rng.uniform(50, 2_000), 2), self._ts(hours=(9, 17)))

    def _normal_business_cash(self):
        business = self.rng.choice(self._businesses["cash_intensive"])
        # Mostly modest takings; occasionally a big weekend that lands near the CTR threshold.
        amount = self.rng.uniform(8_000, 9_800) if self.rng.random() < 0.04 else self.rng.uniform(500, 6_000)
        self._add_cash_in(business.home_branch, business, round(amount, 2), self._ts(hours=(9, 17)))

    def _normal_supplier_wire(self):
        payer = self.rng.choice(self._businesses["merchant"] + self._businesses["cash_intensive"])
        supplier = self.rng.choice(self._businesses["supplier"])
        amount = round(self.rng.uniform(2_000, 40_000), 2)
        self._add(payer, supplier, amount, self._ts(hours=(8, 18)), "wire", f"INVOICE PAYMENT {supplier.holder_name}")

    # ---------- planted scenarios ----------

    def _plant_scenarios(self) -> None:
        for i in range(4):
            self._scenario_structuring(alerted=i != 3)
        for i in range(3):
            self._scenario_mule(alerted=i != 2)
        for i in range(3):
            self._scenario_cycle(alerted=i != 2)
        for _ in range(2):
            self._scenario_funnel(alerted=True)
        for i in range(3):
            self._scenario_pass_through(alerted=i != 1)
        for i in range(2):
            self._scenario_identity_ring(alerted=i == 0)
        for _ in range(2):
            self._scenario_pep(alerted=True)

    def _scenario_structuring(self, alerted: bool):
        rng = self.rng
        customer = self._new_individual(occupation="Self-employed contractor")
        account = self._accounts_by_customer[customer.customer_id][0]
        state_branches = [b for b, _, s in BRANCHES if s == customer.state] or [account.home_branch]
        nearby = sorted({*state_branches, *rng.sample([b for b, _, _ in BRANCHES], 2)})
        day = rng.randrange(self.days - 60, self.days - 10)
        deposits = []
        for _ in range(rng.randint(5, 8)):
            ts = self.start + timedelta(days=day + rng.randint(0, 7), hours=rng.randint(9, 16), minutes=rng.randint(0, 59))
            amount = rng.randrange(9_000, 9_901, 50)
            deposits.append(self._add_cash_in(rng.choice(nearby), account, amount, ts))
        total = sum(t.amount for t in deposits)
        last = max(t.timestamp for t in deposits)
        self._add(account, rng.choice(self._externals["offshore"]), round(total * 0.9, 2), last + timedelta(days=2), "wire", "INTL WIRE OUT")
        self._record(
            "structuring", [customer], [account], alerted,
            f"{len(deposits)} cash deposits of $9,000-$9,900 across {len(nearby)} branches within about a week, then an offshore wire.",
            alert=("STRUCTURING_CASH", rng.randint(72, 92), deposits,
                   f"{len(deposits)} cash deposits just below $10,000 totalling ${total:,.0f}"),
        )

    def _scenario_mule(self, alerted: bool):
        rng = self.rng
        opened = self.as_of - timedelta(days=rng.randint(20, 45))
        customer = self._new_individual(occupation="Student", onboarded_on=opened.isoformat(), salary=0)
        self._employer_of.pop(customer.customer_id, None)
        mule = self._accounts_by_customer[customer.customer_id][0]
        exchange = rng.choice(self._externals["crypto"])
        inbound = []
        for _ in range(rng.randint(2, 3)):  # a few waves of victim payments
            wave_day = rng.randint((opened - self.start.date()).days + 2, self.days - 3)
            wave = []
            for sender in rng.sample(self._individual_accounts, rng.randint(4, 6)):
                ts = self.start + timedelta(days=wave_day, hours=rng.randint(8, 20), minutes=rng.randint(0, 59))
                wave.append(self._add(sender, mule, rng.randrange(400, 2_500, 25), ts, "p2p", "P2P TRANSFER"))
            inbound += wave
            out_ts = max(t.timestamp for t in wave) + timedelta(hours=rng.randint(3, 20))
            self._add(mule, exchange, round(sum(t.amount for t in wave) * rng.uniform(0.88, 0.95), 2), out_ts, "crypto", "CRYPTO PURCHASE")
        senders = len({t.from_id for t in inbound})
        self._record(
            "money_mule", [customer], [mule], alerted,
            f"New account receives P2P payments from {senders} unrelated people and forwards ~90% to a crypto exchange within hours.",
            alert=("FAN_IN_FAN_OUT", rng.randint(60, 85), inbound,
                   f"{senders} distinct senders into a {(self.as_of - opened).days}-day-old account, rapid outflow to crypto"),
        )

    def _scenario_cycle(self, alerted: bool):
        rng = self.rng
        _, city, state = rng.choice(BRANCHES)
        registered_agent = self._address(city, state)  # shell companies share a registered address
        companies = [
            self._new_business("shell", occupation="Consulting", address=registered_agent, risk_rating="medium",
                               onboarded_on=self._random_date(years_back=2).isoformat())
            for _ in range(rng.randint(3, 5))
        ]
        accounts = [self._accounts_by_customer[c.customer_id][0] for c in companies]
        transfers = []
        for round_start in rng.sample(range(10, self.days - 20), 2):
            amount = rng.randrange(80_000, 250_000, 500)
            ts = self.start + timedelta(days=round_start, hours=10)
            for i, src in enumerate(accounts):
                dst = accounts[(i + 1) % len(accounts)]
                transfers.append(self._add(src, dst, amount, ts, "wire", "CONSULTING FEES"))
                amount = round(amount * rng.uniform(0.98, 0.995), 2)
                ts += timedelta(days=rng.randint(1, 3), hours=rng.randint(0, 5))
        self._record(
            "circular_flow", companies, accounts, alerted,
            f"{len(accounts)} consulting companies sharing one registered address pass the same funds around a cycle twice.",
            alert=("CIRCULAR_FLOW", rng.randint(75, 95), transfers[: len(accounts)],
                   f"Funds returned to originator via {len(accounts) - 1} intermediaries"),
        )

    def _scenario_funnel(self, alerted: bool):
        rng = self.rng
        customer = self._new_individual(occupation="Warehouse worker", state="IL")
        account = self._accounts_by_customer[customer.customer_id][0]
        day = rng.randrange(20, self.days - 20)
        deposit_branches = rng.sample([b for b, _, _ in BRANCHES if b not in BORDER_BRANCHES], 6)
        deposits = []
        for _ in range(rng.randint(10, 14)):
            ts = self.start + timedelta(days=day + rng.randint(0, 13), hours=rng.randint(9, 16))
            deposits.append(self._add_cash_in(rng.choice(deposit_branches), account, rng.randrange(2_000, 8_000, 100), ts))
        last = max(t.timestamp for t in deposits)
        border = rng.choice(BORDER_BRANCHES)
        remaining = sum(t.amount for t in deposits) * 0.95
        withdrawals = []
        while remaining > 1_000:
            amount = min(remaining, rng.randrange(5_000, 9_000, 100))
            withdrawals.append(self._add_cash_out(account, border, round(amount, 2), last + timedelta(days=1 + len(withdrawals))))
            remaining -= amount
        self._record(
            "funnel_account", [customer], [account], alerted,
            f"Cash deposited at {len(deposit_branches)} branches in different states, withdrawn in cash at border branch {border}.",
            alert=("MULTI_BRANCH_CASH", rng.randint(55, 78), deposits + withdrawals,
                   f"Cash in at {len(set(t.from_id for t in deposits))} branches, cash out at {border}"),
        )

    def _scenario_pass_through(self, alerted: bool):
        rng = self.rng
        customer = self._new_business("import_export", occupation="Import/export", risk_rating="medium")
        account = self._accounts_by_customer[customer.customer_id][0]
        transfers = []
        for day in sorted(rng.sample(range(5, self.days - 5), rng.randint(2, 3))):
            source, target = rng.sample(self._externals["offshore"], 2)
            amount = rng.randrange(60_000, 300_000, 1_000)
            ts = self.start + timedelta(days=day, hours=rng.randint(9, 15))
            transfers.append(self._add(source, account, amount, ts, "wire", "INTL WIRE IN"))
            out_ts = ts + timedelta(hours=rng.randint(6, 40))
            transfers.append(self._add(account, target, round(amount * rng.uniform(0.92, 0.98), 2), out_ts, "wire", "INTL WIRE OUT"))
        self._record(
            "pass_through", [customer], [account], alerted,
            "Large offshore wires in, 92-98% wired out to a different offshore account within two days.",
            alert=("RAPID_MOVEMENT", rng.randint(65, 90), transfers[:2],
                   f"${transfers[0].amount:,.0f} in and out within 48 hours"),
        )

    def _scenario_identity_ring(self, alerted: bool):
        rng = self.rng
        phone, device = self._phone(), self._device()
        _, city, state = rng.choice(BRANCHES)
        address = self._address(city, state)
        ring = []
        for _ in range(rng.randint(4, 6)):
            opened = self.as_of - timedelta(days=rng.randint(30, 90))
            ring.append(self._new_individual(phone=phone, device_id=device, address=address if rng.random() < 0.6 else self._address(city, state),
                                             onboarded_on=opened.isoformat(), occupation="Sales representative"))
        accounts = [self._accounts_by_customer[c.customer_id][0] for c in ring]
        collector = accounts[0]
        transfers = []
        for account in accounts[1:]:
            ts = self.start + timedelta(days=self.days - rng.randint(3, 15), hours=rng.randint(8, 22))
            transfers.append(self._add(account, collector, rng.randrange(1_500, 4_500, 50), ts, "p2p", "P2P TRANSFER"))
        self._record(
            "synthetic_identity_ring", ring, accounts, alerted,
            f"{len(ring)} recently onboarded customers share a phone number and device; funds pooled into one account.",
            alert=("SHARED_IDENTIFIER", rng.randint(50, 70), transfers,
                   f"{len(ring)} customers share phone {phone} and one device"),
        )

    def _scenario_pep(self, alerted: bool):
        rng = self.rng
        customer = self._new_individual(occupation="Government official", is_pep=True, risk_rating="high")
        account = self._accounts_by_customer[customer.customer_id][0]
        wires = []
        for day in sorted(rng.sample(range(10, self.days - 5), rng.randint(2, 4))):
            source = rng.choice(self._externals["offshore"])
            ts = self.start + timedelta(days=day, hours=rng.randint(9, 16))
            wires.append(self._add(source, account, rng.randrange(40_000, 180_000, 1_000), ts, "wire", "CONSULTANCY FEE"))
        total = sum(t.amount for t in wires)
        self._record(
            "pep_offshore_wires", [customer], [account], alerted,
            f"PEP with ${customer.salary:,}/month declared income receives ${total:,.0f} in offshore wires.",
            alert=("PEP_LARGE_WIRE", rng.randint(40, 60), wires,
                   f"PEP received {len(wires)} offshore wires totalling ${total:,.0f}"),
        )

    def _false_positive_alerts(self) -> None:
        """Alerts on legitimate activity, so triage has something to close."""
        near_threshold = [t for t in self.transfers if t.channel == "cash_deposit" and t.amount >= 8_000
                          and self._is_business(t.to_id)]
        for t in self.rng.sample(near_threshold, min(4, len(near_threshold))):
            self._add_alert(t.to_id, "STRUCTURING_CASH", self.rng.randint(35, 55), [t],
                            f"Cash deposit of ${t.amount:,.0f} by a cash-intensive business", false_positive=True)
        for _ in range(LARGE_WIRE_FALSE_POSITIVES):
            account = self.rng.choice(self._individual_accounts)
            ts = self._ts(hours=(9, 15))
            t = self._add(self.rng.choice(self._externals["title"]), account, self.rng.randrange(150_000, 420_000, 1_000),
                          ts, "wire", "HOME SALE PROCEEDS")
            self._add_alert(account.account_id, "LARGE_WIRE", self.rng.randint(30, 50), [t],
                            f"Incoming wire of ${t.amount:,.0f} from a title company", false_positive=True)

    # ---------- helpers ----------

    def _record(self, typology, customers, accounts, alerted, description, alert):
        rule, score, transfers, summary = alert
        if alerted:
            self._add_alert(accounts[0].account_id, rule, score, transfers, summary)
        self.scenarios.append(Scenario(
            scenario_id=f"S{len(self.scenarios) + 1:02d}", typology=typology,
            customer_ids=[c.customer_id for c in customers], account_ids=[a.account_id for a in accounts],
            alerted=alerted, description=description,
        ))

    def _add_alert(self, account_id, rule, score, transfers, summary, false_positive=False):
        owner = self._customer(next(a.customer_id for a in self.accounts if a.account_id == account_id))
        # Priority follows the (fictional) Example Bank policy in the knowledge base.
        priority = "High" if score >= 80 or owner.is_pep else "Medium" if score >= 50 else "Low"
        if false_positive:
            status = self.rng.choices(["closed", "open"], [70, 30])[0]
        else:
            status = self.rng.choices(["open", "escalated", "closed"], [55, 35, 10])[0]
        created = max(t.timestamp for t in transfers) + timedelta(days=1)
        self.alerts.append(Alert("", account_id, rule, score, priority, status, created, summary, transfers))

    def _add(self, src: Account, dst: Account, amount, ts, channel, description) -> Transfer:
        t = Transfer(src.account_id, "account", dst.account_id, "account", float(amount), ts, channel, description)
        self.transfers.append(t)
        return t

    def _add_cash_in(self, branch_id: str, dst: Account, amount, ts) -> Transfer:
        t = Transfer(branch_id, "branch", dst.account_id, "account", float(amount), ts, "cash_deposit", "CASH DEPOSIT")
        self.transfers.append(t)
        return t

    def _add_cash_out(self, src: Account, branch_id: str, amount, ts) -> Transfer:
        t = Transfer(src.account_id, "account", branch_id, "branch", float(amount), ts, "cash_withdrawal", "CASH WITHDRAWAL")
        self.transfers.append(t)
        return t

    def _customer(self, customer_id: str) -> Customer:
        # Customer IDs are sequential, so the index is the numeric part minus one.
        return self.customers[int(customer_id[1:]) - 1]

    def _is_business(self, account_id: str) -> bool:
        account = next(a for a in self.accounts if a.account_id == account_id)
        return bool(account.customer_id) and self._customer(account.customer_id).customer_type == "business"

    def _ts(self, hours=(0, 23)) -> datetime:
        day = self.rng.randrange(self.days)
        return self.start + timedelta(days=day, hours=self.rng.randint(*hours), minutes=self.rng.randint(0, 59))

    def _random_date(self, years_back: int) -> date:
        return self.as_of - timedelta(days=self.rng.randint(self.days + 30, 365 * years_back))

    def _branch_in(self, state: str) -> str:
        matches = [b for b, _, s in BRANCHES if s == state]
        return self.rng.choice(matches) if matches else self.rng.choice(BRANCHES)[0]

    def _phone(self) -> str:
        # 555-01xx numbers are reserved for fiction.
        return f"+1-{self.rng.randint(200, 989)}-555-01{self.rng.randint(0, 99):02d}"

    def _device(self) -> str:
        return f"dev-{self.rng.getrandbits(40):010x}"

    def _address(self, city: str, state: str) -> str:
        return f"{self.rng.randint(10, 9999)} {self.rng.choice(STREETS)} St, {city}, {state}"


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
