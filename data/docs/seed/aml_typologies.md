# Money Laundering Typologies Overview

Summary written for this project. For authoritative detail, see the FATF typology reports and FinCEN advisories listed in data/SOURCES.md.

## The three stages of money laundering

### Placement
Placement is the point where illicit cash first enters the financial system. It is the stage where launderers are most exposed, because large cash amounts attract attention. Common methods include cash deposits split across branches, purchases of monetary instruments such as money orders or cashier's checks, and mixing illicit cash into the takings of a cash-intensive business.

### Layering
Layering is a series of transactions designed to distance funds from their source and make the audit trail hard to follow. Typical techniques are rapid wire transfers between accounts and jurisdictions, movement through shell companies, conversion between currencies or into crypto assets, and back-to-back loans.

### Integration
Integration returns the laundered funds to the launderer as apparently legitimate wealth: real estate purchases, luxury goods, investments in businesses, or fake loan repayments and inflated invoices.

## Structuring (smurfing)

Structuring means breaking cash transactions into amounts below a reporting threshold to avoid a Currency Transaction Report (CTR). In the United States the CTR threshold is cash over $10,000 in a business day, so structured deposits often cluster just below that amount, for example several deposits of $9,000 to $9,900. Structuring is a crime in itself in the US (31 U.S.C. 5324), even when the underlying money is legitimate.

Red flags:
- Repeated cash deposits just under $10,000
- Deposits of the same customer spread across several branches or ATMs on the same day
- Multiple people ("smurfs") depositing into one account
- Customer asks about reporting thresholds or changes the amount after being told a form is required

## Funnel accounts

A funnel account receives many small cash deposits in different geographic locations, and the funds are then withdrawn quickly in another location, often near a border. FinCEN has highlighted funnel accounts as a common method for moving drug proceeds.

Red flags:
- Deposits in many cities or states that don't match where the customer lives or does business
- Balance stays low because money leaves almost as fast as it arrives
- Out-of-area cash withdrawals or wires shortly after deposits

## Rapid movement of funds (pass-through activity)

Funds arrive by wire or ACH and leave within hours or days, with little or no balance retained. On its own this can be legitimate for some businesses, but it is a classic layering pattern when the account profile doesn't justify it.

## Round-tripping and circular transactions

Money leaves an account, passes through one or more intermediaries, and comes back to the originator or a related party. In a transaction graph this shows up as a cycle (A to B to C to A). It is used to create fake revenue, disguise the origin of funds, or inflate turnover.

## Trade-based money laundering (TBML)

TBML moves value through trade transactions rather than direct transfers. Techniques include:
- Over-invoicing or under-invoicing goods
- Multiple invoicing for the same shipment
- Over- or under-shipping relative to the invoice
- "Phantom shipments" where no goods move at all

Red flags include prices far from market value, inconsistent shipping documents, and payments from third parties unrelated to the trade.

## Money mules

A money mule receives and moves funds on behalf of others, knowingly or not. Mules are often recruited through fake job ads ("payment processing agent"), romance scams or social media. Mule accounts are frequently new, receive funds from many unrelated senders and forward them quickly, often to crypto exchanges or abroad.

## Shell and front companies

Shell companies have no real operations; front companies run a genuine business that is used to mix in illicit funds. Red flags include no web presence or employees, a registered address shared with many other companies, nominee directors, and transaction volumes out of line with stated business.

## Shared attributes and networks

Seemingly unrelated customers who share a phone number, email, device, IP address or residential address can indicate a coordinated network, synthetic identities or mule rings. Graph analysis is especially effective at detecting these links.
