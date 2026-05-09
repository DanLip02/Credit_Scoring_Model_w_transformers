# feature_descriptions.py
# Feature descriptions for all datasets used in FeatureSemanticMatcher
# Used to enrich feature names before BERT encoding for better semantic matching

FEATURE_DESCRIPTIONS = {
    "GiveMeCredit": {
        "RevolvingUtilizationOfUnsecuredLines": "ratio of revolving credit balance used to total unsecured credit limit excluding real estate",
        "age": "age of the borrower in years",
        "NumberOfTime30-59DaysPastDueNotWorse": "number of times borrower was 30 to 59 days past due in last 2 years",
        "DebtRatio": "ratio of monthly debt payments including alimony and living costs to monthly gross income",
        "MonthlyIncome": "monthly gross income of the borrower in US dollars",
        "NumberOfOpenCreditLinesAndLoans": "total number of open loans and lines of credit including installment loans and revolving credit",
        "NumberOfTimes90DaysLate": "number of times borrower was 90 days or more past due",
        "NumberRealEstateLoansOrLines": "number of mortgage and real estate loans including home equity lines",
        "NumberOfTime60-89DaysPastDueNotWorse": "number of times borrower was 60 to 89 days past due in last 2 years",
        "NumberOfDependents": "number of dependents in family excluding borrower spouse partner children",
        "MonthlyIncome_missing": "binary flag indicating monthly income value is missing",
        "NumberOfDependents_missing": "binary flag indicating number of dependents value is missing",
        "HasHiddenLatePayments": "binary flag indicating borrower has hidden or undisclosed late payments",
        "TotalLatePayments_weighted": "weighted sum of all late payment counts by severity of payment delay",
    },

    "HomeCredit": {
        "CNT_CHILDREN": "number of children the client has",
        "AMT_INCOME_TOTAL": "total annual income of the applicant in local currency",
        "AMT_CREDIT": "credit amount of the loan requested by applicant",
        "AMT_ANNUITY": "monthly annuity payment amount for the requested loan",
        "AMT_GOODS_PRICE": "price of goods for which the loan is given",
        "REGION_POPULATION_RELATIVE": "normalized population of region where client lives relative to whole population",
        "DAYS_BIRTH": "client age expressed as negative number of days relative to application date",
        "DAYS_EMPLOYED": "number of days before application when person started current employment negative means currently employed",
        "DAYS_REGISTRATION": "number of days before application when client changed registration",
        "DAYS_ID_PUBLISH": "number of days before application when client changed identity document",
        "FLAG_WORK_PHONE": "binary flag indicating if client provided work phone number",
        "FLAG_PHONE": "binary flag indicating if client provided home phone number",
        "FLAG_EMAIL": "binary flag indicating if client provided email address",
        "CNT_FAM_MEMBERS": "total number of family members of the client",
        "REGION_RATING_CLIENT": "credit rating of the region where client lives",
        "REGION_RATING_CLIENT_W_CITY": "credit rating of the region where client lives including city rating",
        "HOUR_APPR_PROCESS_START": "hour of the day when the loan application was submitted",
        "REG_REGION_NOT_LIVE_REGION": "binary flag if client registration region differs from living region",
        "REG_REGION_NOT_WORK_REGION": "binary flag if client registration region differs from work region",
        "LIVE_REGION_NOT_WORK_REGION": "binary flag if client living region differs from work region",
        "REG_CITY_NOT_LIVE_CITY": "binary flag if client registration city differs from living city",
        "REG_CITY_NOT_WORK_CITY": "binary flag if client registration city differs from work city",
        "LIVE_CITY_NOT_WORK_CITY": "binary flag if client living city differs from work city",
        "EXT_SOURCE_1": "normalized score from external data source 1 representing creditworthiness",
        "EXT_SOURCE_2": "normalized score from external data source 2 representing creditworthiness",
        "EXT_SOURCE_3": "normalized score from external data source 3 representing creditworthiness",
        "OBS_30_CNT_SOCIAL_CIRCLE": "number of clients social circle observations with 30 day past due",
        "DEF_30_CNT_SOCIAL_CIRCLE": "number of clients social circle defaulters with 30 day past due",
        "DEF_60_CNT_SOCIAL_CIRCLE": "number of clients social circle defaulters with 60 day past due",
        "DAYS_LAST_PHONE_CHANGE": "number of days before application when client last changed phone number",
        "FLAG_DOCUMENT_3": "binary flag if client provided document 3",
        "FLAG_DOCUMENT_5": "binary flag if client provided document 5",
        "FLAG_DOCUMENT_6": "binary flag if client provided document 6",
        "FLAG_DOCUMENT_8": "binary flag if client provided document 8",
        "FLAG_DOCUMENT_9": "binary flag if client provided document 9",
        "FLAG_DOCUMENT_11": "binary flag if client provided document 11",
        "FLAG_DOCUMENT_13": "binary flag if client provided document 13",
        "FLAG_DOCUMENT_14": "binary flag if client provided document 14",
        "FLAG_DOCUMENT_15": "binary flag if client provided document 15",
        "FLAG_DOCUMENT_16": "binary flag if client provided document 16",
        "FLAG_DOCUMENT_18": "binary flag if client provided document 18",
        "AMT_REQ_CREDIT_BUREAU_HOUR": "number of credit bureau enquiries in last hour before application",
        "AMT_REQ_CREDIT_BUREAU_DAY": "number of credit bureau enquiries in last day before application",
        "AMT_REQ_CREDIT_BUREAU_WEEK": "number of credit bureau enquiries in last week before application",
        "AMT_REQ_CREDIT_BUREAU_MON": "number of credit bureau enquiries in last month before application",
        "AMT_REQ_CREDIT_BUREAU_QRT": "number of credit bureau enquiries in last quarter before application",
        "AMT_REQ_CREDIT_BUREAU_YEAR": "number of credit bureau enquiries in last year before application",
        "NAME_CONTRACT_TYPE": "type of loan contract cash or revolving",
        "CODE_GENDER": "gender of the client male or female",
        "FLAG_OWN_CAR": "binary flag indicating if client owns a car",
        "FLAG_OWN_REALTY": "binary flag indicating if client owns real estate property",
        "NAME_TYPE_SUITE": "who accompanied client during application",
        "NAME_INCOME_TYPE": "type of income source of the client",
        "NAME_EDUCATION_TYPE": "highest education level of the client",
        "NAME_FAMILY_STATUS": "family status of the client married single divorced",
        "NAME_HOUSING_TYPE": "type of housing where client lives",
        "OCCUPATION_TYPE": "occupation or job type of the client",
        "WEEKDAY_APPR_PROCESS_START": "day of the week when application was submitted",
        "ORGANIZATION_TYPE": "type of organization where client works",
        "bureau_loan_count": "total number of previous loans registered in credit bureau",
        "bureau_active_count": "number of currently active loans in credit bureau",
        "bureau_closed_count": "number of closed loans in credit bureau history",
        "bureau_active_ratio": "ratio of active loans to total loans in credit bureau",
        "bureau_amt_credit_sum": "total credit amount across all credit bureau loans",
        "bureau_amt_credit_mean": "average credit amount across all credit bureau loans",
        "bureau_amt_credit_max": "maximum credit amount across all credit bureau loans",
        "bureau_amt_debt_sum": "total outstanding debt amount from all credit bureau records",
        "bureau_amt_debt_mean": "average outstanding debt amount from credit bureau records",
        "bureau_amt_overdue_sum": "total overdue amount across all credit bureau loan records",
        "bureau_overdue_sum": "total overdue amount from credit bureau balance records",
        "bureau_days_credit_mean": "average number of days since credit bureau record was opened",
        "bureau_days_credit_min": "minimum number of days since credit bureau record was opened",
        "bureau_days_credit_max": "maximum number of days since credit bureau record was opened",
        "bureau_days_update_max": "maximum number of days since last credit bureau record update",
        "bureau_annuity_sum": "total annuity payments from all credit bureau loan records",
        "bureau_bb_dpd_max": "maximum days past due from credit bureau balance records",
        "bureau_bb_months_min": "minimum number of months in credit bureau balance history",
        "bureau_debt_credit_ratio": "ratio of total debt to total credit from credit bureau",
        # Previous applications aggregated features
        "prev_app_count": "total number of previous loan applications",
        "prev_approved_count": "number of previously approved loan applications",
        "prev_refused_count": "number of previously refused loan applications",
        "prev_amt_credit_sum": "total credit amount across all previous applications",
        "prev_amt_credit_mean": "average credit amount across all previous applications",
        "prev_amt_credit_max": "maximum credit amount across all previous applications",
        "prev_amt_annuity_sum": "total annuity amount across all previous applications",
        "prev_amt_goods_mean": "average goods price across all previous applications",
        "prev_amt_down_payment_mean": "average down payment amount across all previous applications",
        "prev_days_last_due_mean": "average number of days to last due date across previous applications",
        "prev_cnt_payment_mean": "average number of payments across all previous applications",
        "prev_cnt_payment_max": "maximum number of payments across all previous applications",
        "prev_credit_ratio_mean": "average ratio of credit amount to goods price in previous applications",
        "prev_credit_ratio_min": "minimum ratio of credit amount to goods price in previous applications",
        "prev_down_payment_mean": "average down payment ratio in previous applications",
        "down_payment_mean": "average down payment amount in previous loan applications",
        "app_count": "total number of loan applications submitted by client",
        "approved_ratio": "ratio of approved to total loan applications",
        "prev_amt_credit_ratio": "ratio of previous credit amount to current application credit amount",
    },

    "CreditScoring": {
        "Age": "age of the customer in years",
        "Annual_Income": "annual income of the customer in local currency",
        "Monthly_Inhand_Salary": "monthly take home salary after tax deductions",
        "Num_Bank_Accounts": "number of bank accounts held by the customer",
        "Num_Credit_Card": "number of credit cards owned by the customer",
        "Interest_Rate": "interest rate on the customers credit in percent per year",
        "Num_of_Loan": "total number of loans taken by the customer",
        "Delay_from_due_date": "average number of days delayed from payment due date",
        "Num_of_Delayed_Payment": "number of delayed payments made by the customer",
        "Changed_Credit_Limit": "percentage change in credit card limit",
        "Num_Credit_Inquiries": "number of credit inquiries made in recent period",
        "Outstanding_Debt": "total outstanding debt amount of the customer",
        "Credit_Utilization_Ratio": "ratio of credit used to total available credit limit",
        "Total_EMI_per_month": "total monthly equated monthly installment payments",
        "Amount_invested_monthly": "monthly amount invested by the customer",
        "Monthly_Balance": "monthly balance amount remaining after expenses",
        "credit_history_months": "length of credit history in months",
        "num_loan_types": "number of different types of loans held by customer",
        "month_num": "month number in the observation period",
        "emi_to_income": "ratio of total EMI payments to monthly income",
        "debt_to_income": "ratio of outstanding debt to annual income",
        "invest_to_income": "ratio of monthly investment to monthly income",
        "balance_to_income": "ratio of monthly balance to monthly income",
        "log_annual_income": "natural logarithm of annual income",
        "log_outstanding_debt": "natural logarithm of outstanding debt amount",
        "credit_cards_per_bank": "ratio of credit cards to number of bank accounts",
        "Occupation": "occupation or profession of the customer",
        "Credit_Mix": "mix of credit types held by the customer",
        "Payment_of_Min_Amount": "whether customer pays only minimum amount due on credit card",
        "Payment_Behaviour": "spending and payment behaviour pattern of the customer",
        "has_auto_loan": "binary flag indicating customer has an auto loan",
        "has_personal_loan": "binary flag indicating customer has a personal loan",
        "has_student_loan": "binary flag indicating customer has a student loan",
        "has_home_equity_loan": "binary flag indicating customer has a home equity loan",
        "has_mortgage_loan": "binary flag indicating customer has a mortgage loan",
        "has_payday_loan": "binary flag indicating customer has a payday loan",
        "has_debt_consolidation_loan": "binary flag indicating customer has a debt consolidation loan",
        "has_credit_builder_loan": "binary flag indicating customer has a credit builder loan",
        "REGION_RATING_CLIENT": "credit rating of the region where customer lives",
        "REGION_RATING_CLIENT_W_CITY": "credit rating of the region including city where customer lives",
    },

    "Application": {
        "id": "unique application identifier number",
        "age": "age of the applicant in years at time of application",
        "Score_bki": "credit bureau score of the applicant",
        "log_income": "natural logarithm of applicant monthly income",
        "log_appl_rej": "natural logarithm of number of previous application rejections",
        "log_out_req": "natural logarithm of number of outstanding credit requests",
        "region_rating_ord": "ordinal credit rating of applicant region from 1 to 3",
        "home_address_cd": "code indicating type of home address registration",
        "work_address_cd": "code indicating type of work address registration",
        "SNA": "social network analysis score of the applicant",
        "first_time_cd": "binary flag indicating first time credit applicant",
        "good_work_flg": "binary flag indicating applicant has stable good employment",
        "app_month": "month of loan application submission",
        "app_quarter": "quarter of loan application submission",
        "education_cd": "education level code of the applicant",
        "gender_cd": "gender code of the applicant male or female",
        "car_own_flg": "binary flag indicating applicant owns a car",
        "car_type_flg": "binary flag indicating type of car owned by applicant",
        "Air_flg": "binary flag indicating applicant has air travel history",
    },

    "TaiwanCC": {
        "LIMIT_BAL": "credit limit amount in new taiwan dollar given to client",
        "AGE": "age of the credit card holder in years",
        "PAY_0": "repayment status in september most recent month minus 1 delay plus paid",
        "PAY_2": "repayment status in august two months ago",
        "PAY_3": "repayment status in july three months ago",
        "PAY_4": "repayment status in june four months ago",
        "PAY_5": "repayment status in may five months ago",
        "PAY_6": "repayment status in april six months ago oldest",
        "BILL_AMT1": "bill statement amount in september most recent month",
        "BILL_AMT2": "bill statement amount in august two months ago",
        "BILL_AMT3": "bill statement amount in july three months ago",
        "BILL_AMT4": "bill statement amount in june four months ago",
        "BILL_AMT5": "bill statement amount in may five months ago",
        "BILL_AMT6": "bill statement amount in april six months ago oldest",
        "PAY_AMT1": "amount of previous payment made in september",
        "PAY_AMT2": "amount of previous payment made in august",
        "PAY_AMT3": "amount of previous payment made in july",
        "PAY_AMT4": "amount of previous payment made in june",
        "PAY_AMT5": "amount of previous payment made in may",
        "PAY_AMT6": "amount of previous payment made in april",
        "SEX": "gender of the credit card holder male or female",
        "EDUCATION": "education level graduate school university high school others",
        "MARRIAGE": "marital status married single others",
        "max_pay_delay": "maximum repayment delay in months across all six observation months",
        "mean_pay_delay": "average repayment delay in months across all six observation months",
        "n_delayed": "number of months with positive payment delay out of six months",
        "mean_bill": "average monthly bill statement amount across six months",
        "max_bill": "maximum monthly bill statement amount across six months",
        "bill_trend": "difference between most recent and oldest bill amount indicating debt trend",
        "mean_pay_amt": "average monthly payment amount made by card holder across six months",
        "total_pay_amt": "total payment amount made across all six observation months",
        "util_ratio": "ratio of most recent bill amount to credit limit utilization",
        "pay_ratio_1": "ratio of payment made to bill amount in most recent month",
        "log_limit_bal": "natural logarithm of credit limit balance",
        "log_mean_bill": "natural logarithm of average bill statement amount",
        "log_mean_pay": "natural logarithm of average payment amount made",
    },

    "LendingClub": {
        "loan_amnt": "requested loan amount in US dollars",
        "funded_amnt": "total amount committed to the loan by investors",
        "int_rate": "interest rate on the loan in percent per year",
        "installment": "monthly payment owed by the borrower if loan originates",
        "annual_inc": "annual income of the borrower in US dollars self reported",
        "dti": "debt to income ratio calculated using all monthly debt obligations",
        "delinq_2yrs": "number of 30 plus days delinquencies in past 2 years",
        "inq_last_6mths": "number of credit inquiries in past 6 months",
        "open_acc": "number of open credit lines in borrowers credit file",
        "pub_rec": "number of derogatory public records bankruptcies tax liens",
        "revol_bal": "total credit revolving balance outstanding",
        "revol_util": "revolving line utilization rate amount of credit used relative to total revolving credit available",
        "total_acc": "total number of credit lines currently in borrowers credit file",
        "out_prncp": "remaining outstanding principal for portion of total amount funded",
        "total_pymnt": "payments received to date for total amount funded",
        "total_rec_prncp": "principal received to date from borrower",
        "total_rec_int": "interest received to date from borrower",
        "last_pymnt_amnt": "last total payment amount received from borrower",
        "grade": "loan grade assigned by lender indicating credit risk level A through G",
        "sub_grade": "loan subgrade providing finer credit risk classification",
        "purpose": "category provided by the borrower for the loan request purpose",
    },
}


def get_enriched_name(feature_name: str, dataset_name: str = None) -> str:
    """
    Returns enriched feature description for better semantic matching via BERT.
    Falls back to normalized feature name if no description found.
    
    Args:
        feature_name: original feature column name
        dataset_name: dataset key from FEATURE_DESCRIPTIONS dict
    
    Returns:
        Enriched string: "feature name: description" or normalized name
    
    Example:
        get_enriched_name("AGE", "TaiwanCC")
        -> "age: age of the credit card holder in years"
        
        get_enriched_name("unknown_feature", "TaiwanCC")
        -> "unknown feature"
    """
    if dataset_name:
        desc = FEATURE_DESCRIPTIONS.get(dataset_name, {}).get(feature_name)
        if desc:
            return f"{feature_name.lower().replace('_', ' ')}: {desc}"
    return feature_name.lower().replace("_", " ")


def get_enriched_names(col_names: list, dataset_name: str = None) -> list:
    """
    Batch version of get_enriched_name for list of features.
    
    Args:
        col_names: list of feature column names
        dataset_name: dataset key from FEATURE_DESCRIPTIONS dict
    
    Returns:
        List of enriched strings for BERT encoding
    """
    return [get_enriched_name(name, dataset_name) for name in col_names]


if __name__ == "__main__":
    print("Feature descriptions loaded successfully!")
    print(f"Datasets covered: {list(FEATURE_DESCRIPTIONS.keys())}")
    for ds, feats in FEATURE_DESCRIPTIONS.items():
        print(f"  {ds}: {len(feats)} features")
    
    print()
    print("Example enriched names:")
    examples = [
        ("AGE", "TaiwanCC"),
        ("AMT_INCOME_TOTAL", "HomeCredit"),
        ("DebtRatio", "GiveMeCredit"),
        ("Age", "CreditScoring"),
        ("log_income", "Application"),
    ]
    for feat, ds in examples:
        print(f"  {feat} ({ds}): {get_enriched_name(feat, ds)}")
