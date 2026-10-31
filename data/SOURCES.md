# Public sources to add to the knowledge base

The files in `data/docs/seed/` are short summaries written for this project. To make the RAG more realistic, download any of these public documents (PDF or saved text) into `data/docs/public/` and re-index.

Keep the total small at first (5–10 documents). Large PDFs are fine, but indexing runs on your CPU.

## Regulatory and guidance (US)
- FFIEC BSA/AML Examination Manual: https://bsaaml.ffiec.gov/manual
- FinCEN advisories and alerts: https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets/advisories
- FinCEN SAR narrative guidance ("Guidance on Preparing a Complete & Sufficient SAR Narrative"): https://www.fincen.gov
- OFAC sanctions programs and the SDN list: https://ofac.treasury.gov

## International
- FATF Recommendations: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html
- FATF money laundering and terrorist financing typologies: https://www.fatf-gafi.org/en/topics/methods-and-trends.html

## Datasets for phase 2 (structured data, Neo4j)
- IBM Transactions for Anti Money Laundering (synthetic): search Kaggle for "IBM Transactions for Anti Money Laundering". Use the smallest "HI-Small" file and sample at most MAX_TRANSACTIONS rows.
- PaySim mobile money fraud (synthetic): search Kaggle for "PaySim".

Only use synthetic data. Never load real customer data into this project.
