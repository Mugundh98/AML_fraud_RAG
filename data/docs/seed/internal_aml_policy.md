# Example Bank AML Alert Handling Policy (Fictional)

This is a fictional internal policy written for this demo project. "Example Bank" does not exist, and the thresholds and timelines below are illustrative choices, not regulatory requirements.

## Scope

This policy covers how transaction-monitoring alerts are triaged, investigated, escalated and closed by the Financial Crimes Unit (FCU) of Example Bank.

## Roles

- **Level 1 Analyst:** reviews new alerts, gathers customer and transaction information, and either closes the alert with a rationale or escalates it to a case.
- **Level 2 Investigator:** investigates escalated cases, decides whether a SAR is warranted, and drafts the SAR narrative.
- **BSA Officer:** approves or rejects SAR filing decisions and owns the overall AML program.
- **Quality Assurance:** samples closed alerts and cases each month to check decision quality.

## Alert risk scoring

Every alert gets a risk score from 0 to 100 based on the rule that fired, customer risk rating, amounts and counterparties.

| Score | Priority | Triage deadline |
|-------|----------|-----------------|
| 80-100 | High | 2 business days |
| 50-79 | Medium | 5 business days |
| 0-49 | Low | 10 business days |

Alerts involving a possible sanctions match, terrorist financing or a customer who is a politically exposed person (PEP) are always treated as High priority regardless of score.

## Triage steps (Level 1)

1. Review the customer profile: occupation or business type, expected activity declared at onboarding, risk rating and prior alerts.
2. Review 90 days of transactions around the alert, looking for the typologies in the AML Typologies Overview.
3. Check counterparties against internal watchlists and the sanctions screening results.
4. Search for negative news about the customer.
5. Decide: close with a written rationale, or escalate to a case.

An alert can be closed only if the activity is consistent with the customer's known profile and the rationale says why. "No suspicious activity found" on its own is not an acceptable rationale.

## Escalation criteria

Escalate to a Level 2 case when any of the following applies:
- Activity matches a known typology and cannot be explained by the customer profile
- The customer has had three or more alerts in the past 12 months
- There is a confirmed or likely sanctions match
- Negative news links the customer to financial crime
- The analyst cannot get enough information to rule out suspicion

Urgent matters (ongoing fraud with funds still recoverable, sanctions hits, terrorist financing) must also be raised immediately in the #fcu-escalations Slack channel and by phone to the on-call Level 2 Investigator.

## Investigation and SAR decision (Level 2)

- Cases must reach a SAR decision within 20 calendar days of escalation, leaving at least 10 days before the regulatory 30-day SAR deadline counted from initial detection.
- The investigator documents the decision and, if filing, drafts the SAR narrative using the five Ws and how.
- The BSA Officer reviews and approves all SAR filings.
- After a SAR is filed, the customer relationship is reviewed for exit within 30 days if two SARs have been filed on the same customer within 12 months.

## Case review meetings

A weekly case review meeting is held with Level 2 investigators and the BSA Officer to discuss complex cases. Analysts can request an ad hoc review meeting for a case that needs input from several teams, such as Fraud Operations, Sanctions or Legal.

## Record keeping

All alert decisions, case files, SAR drafts and approvals are retained for at least five years. Notes must never be shared with the customer, in line with SAR confidentiality rules.

## Enhanced due diligence (EDD)

EDD applies to high-risk customers, including PEPs, cash-intensive businesses, money services businesses, and customers with links to high-risk jurisdictions. EDD requires source of funds and source of wealth verification, senior management approval to open or keep the relationship, and a full review at least every 12 months.
