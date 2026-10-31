# Fraud Typologies and Red Flags

Summary written for this project. Fraud and money laundering overlap: fraud proceeds usually need to be laundered, and mule accounts serve both.

## Account takeover (ATO)

A criminal gains control of a genuine customer's account, usually through phishing, credential stuffing, SIM swapping or malware.

Red flags:
- Login from a new device, browser or country, followed quickly by high-risk changes
- Change of email, phone number or mailing address shortly before a large transfer
- New payee added and paid within minutes
- Several failed login or one-time-passcode attempts before a successful one
- Customer's typical session behavior (typing speed, navigation) suddenly changes

## Authorized push payment (APP) fraud

The genuine customer is tricked into sending money themselves, so the payment looks authorized. Variants include romance scams, investment scams, impersonation of the bank or police, and invoice redirection (business email compromise).

Red flags:
- First-ever payment to a new payee for an unusually large amount
- Customer is under time pressure or is being coached on the phone during the transaction
- Payee account is newly opened or has a history of receiving many inbound payments from unrelated people
- Payment reference doesn't fit the stated purpose

## Business email compromise (BEC)

Attackers compromise or spoof a supplier's or executive's email and ask for payment to a new bank account.

Red flags:
- Last-minute change of bank details for a known supplier
- Request marked urgent or confidential, bypassing normal approval
- Sender domain that is slightly different from the real one

## Synthetic identity fraud

A fraudster combines real and fabricated information (for example, a real Social Security number with a fake name and date of birth) to create a new identity. The identity is "nursed" over months to build credit, then used to max out credit lines and disappear ("bust-out").

Red flags:
- SSN issued recently relative to the stated date of birth, or associated with multiple names
- Thin credit file with rapid credit-limit growth
- Contact details (phone, address, device) shared with other applicants
- Authorized-user tradelines added from unrelated people

## First-party fraud and bust-out

The account holder themselves intends never to repay. Typical signs are a sudden spike in credit utilization, cash advances, and balance transfers just before default, often after a period of good behavior.

## Card fraud

- **Card-not-present (CNP):** stolen card details used online. Watch for many small test transactions followed by a large purchase, mismatched billing and shipping addresses, and high-risk merchant categories.
- **Card testing:** bots run many small authorizations to find valid cards. Look for bursts of low-value declines from one merchant or IP.
- **Skimming:** card data copied at ATMs or point-of-sale terminals. Clusters of fraud on cards that share a common point of purchase indicate a compromised terminal.

## Elder financial exploitation

Older customers are targeted by scammers, caregivers or relatives. Red flags include sudden large withdrawals, new joint account holders or powers of attorney, changes to beneficiaries, and the customer appearing confused or accompanied by someone who speaks for them.

## Check fraud

Includes counterfeit, altered and stolen checks, and check kiting (exploiting float time between accounts at different banks). Red flags are deposits of checks followed by immediate withdrawals, mobile deposit of the same check more than once, and checks with altered payee names.
