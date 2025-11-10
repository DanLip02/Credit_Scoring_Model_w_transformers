# German Credit Risk Dataset Documentation

## Target Variable
- **target**: Credit risk classification
  - `1` = Good client (low risk)
  - `2` = Bad client (high risk)

## Attributes Description

### Categorical Attributes

| Attribute | Code | Description |
|-----------|------|-------------|
| **status_existing_checking** | A11 | < 0 DM (negative balance) |
| | A12 | 0 ≤ ... < 200 DM |
| | A13 | ≥ 200 DM |
| | A14 | No checking account |
| **credit_history** | A30 | No credits/all paid back duly |
| | A31 | All credits at this bank paid back |
| | A32 | Existing credits paid duly till now |
| | A33 | Delay in paying in the past |
| | A34 | Critical account/other credits |
| **purpose** | A40 | Car (new) |
| | A41 | Car (used) |
| | A42 | Furniture/equipment |
| | A43 | Radio/television |
| | A44 | Domestic appliances |
| | A45 | Repairs |
| | A46 | Education |
| | A47 | Vacation |
| | A48 | Retraining |
| | A49 | Business |
| | A410 | Other |
| **savings** | A61 | < 100 DM |
| | A62 | 100 ≤ ... < 500 DM |
| | A63 | 500 ≤ ... < 1000 DM |
| | A64 | ≥ 1000 DM |
| | A65 | Unknown/no savings |
| **employment** | A71 | Unemployed |
| | A72 | < 1 year |
| | A73 | 1 ≤ ... < 4 years |
| | A74 | 4 ≤ ... < 7 years |
| | A75 | ≥ 7 years |
| **personal_status_sex** | A91 | Male: divorced/separated |
| | A92 | Female: divorced/separated/married |
| | A93 | Male: single |
| | A94 | Male: married/widowed |
| **other_debtors** | A101 | None |
| | A102 | Co-applicant |
| | A103 | Guarantor |
| **property** | A121 | Real estate |
| | A122 | Building society savings/life insurance |
| | A123 | Car or other |
| | A124 | Unknown/no property |
| **other_installment_plans** | A141 | Bank |
| | A142 | Stores |
| | A143 | None |
| **housing** | A151 | Rent |
| | A152 | Own |
| | A153 | For free |
| **job** | A171 | Unemployed/unskilled non-resident |
| | A172 | Unskilled resident |
| | A173 | Skilled employee/official |
| | A174 | Management/self-employed |
| **telephone** | A191 | No |
| | A192 | Yes |
| **foreign_worker** | A201 | Yes |
| | A202 | No |

### Numerical Attributes

| Attribute | Description | Range |
|-----------|-------------|-------|
| **duration_month** | Loan duration in months | - |
| **credit_amount** | Credit amount in DM | - |
| **installment_rate** | Installment rate percentage | 1-4 |
| **residence_since** | Years at current residence | - |
| **age** | Age in years | - |
| **existing_credits** | Number of existing credits | - |
| **people_liable** | Number of dependents | - |