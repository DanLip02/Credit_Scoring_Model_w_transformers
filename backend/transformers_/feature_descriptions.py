# feature_descriptions.py
# Feature descriptions for all datasets used in FeatureSemanticMatcher
# Used to enrich feature names before BERT encoding for better semantic matching

FEATURE_DESCRIPTIONS = {
    "GiveMeCredit": {
        "RevolvingUtilizationOfUnsecuredLines": "ratio of revolving credit balance used to total unsecured credit limit excluding real estate",
        "age": "age of borrower in years old person birth date days born",
        "NumberOfTime30-59DaysPastDueNotWorse": "THIS borrower FIRST STAGE 30 thirty to 59 days late one month personal 30 to 59 range NOT sixty NOT external",
        "DebtRatio": "ratio percentage of monthly debt payments to monthly income debt burden ratio",
        "MonthlyIncome": "monthly gross income salary earnings of the borrower NOT annuity payments NOT debt amount",
        "NumberOfOpenCreditLinesAndLoans": "total number of open loans and lines of credit including installment loans and revolving credit",
        "NumberOfTimes90DaysLate": "THIS borrower 90 ninety days past due severely delinquent third stage 3 months late maximum worst delinquency",
        "NumberRealEstateLoansOrLines": "number of mortgage real estate property home equity loans specific to property",
        "NumberOfTime60-89DaysPastDueNotWorse": "THIS borrower 60 sixty to 89 eighty nine days late on own personal payment second stage mid delinquency two months past due NOT external",
        "NumberOfDependents": "number of dependents in family including borrower spouse partner children",
        "MonthlyIncome_missing": "flag indicating monthly income value is missing unknown NOT related to city location address",
        "NumberOfDependents_missing": "flag indicating number of dependents value is missing unknown NOT related to documents",
        "HasHiddenLatePayments": "binary flag indicating borrower has hidden or undisclosed late payments",
         "TotalLatePayments_weighted": "weighted sum of borrower late payments weighted by severity of delay",
    },

    "HomeCredit": {
    "CNT_CHILDREN": "number of children the client has",
    "AMT_INCOME_TOTAL": "total annual income of the applicant in local currency",
    "AMT_CREDIT": "credit amount of the loan requested by applicant",
    "AMT_ANNUITY": "monthly annuity payment amount for the requested loan",
    "AMT_GOODS_PRICE": "price of goods for which the loan is given",
    "REGION_POPULATION_RELATIVE": "normalized population of region where client lives relative to whole population",
    "DAYS_BIRTH": "age of client in years how old person is birth date days old age borrower",
    "DAYS_EMPLOYED": "number of days before application when person started current employment negative means currently employed",
    "DAYS_REGISTRATION": "number of days before application when client changed registration",
    "DAYS_ID_PUBLISH": "number of days before application when client changed identity document",
    "FLAG_WORK_PHONE": "binary flag indicating if client provided work phone number",
    "FLAG_PHONE": "binary flag client provided home phone telephone contact number NOT mortgage NOT loan NOT property",
    "FLAG_EMAIL": "binary flag indicating if client provided email address",
    "CNT_FAM_MEMBERS": "count number family members children dependents household size persons living with client NOT income NOT salary NOT money",
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
    "DEF_30_CNT_SOCIAL_CIRCLE": "number of social circle clients 30 days past due delinquent overdue default late payment missed installment",
    "DEF_60_CNT_SOCIAL_CIRCLE": "number of social circle clients 60 days past due delinquent overdue default late payment missed installment",
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
    "CODE_GENDER": "CODE GENDER gender male female sex CODE GENDER client borrower",
    "FLAG_OWN_CAR": "binary flag indicating if client owns a car",
    "FLAG_OWN_REALTY": "binary flag client owns real estate property house apartment building home mortgage",
    "NAME_TYPE_SUITE": "who accompanied client during application",
    "NAME_INCOME_TYPE": "type of income source of the client",
    "NAME_EDUCATION_TYPE": "education level graduate school university high school others borrower academic degree diploma qualification",
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
    "bureau_amt_overdue_sum": "total overdue amount delinquent late payment default missed payment credit bureau",
    "bureau_overdue_sum": "total overdue amount from credit bureau balance records",
    "bureau_days_credit_mean": "average number of days since credit bureau record was opened",
    "bureau_days_credit_min": "minimum number of days since credit bureau record was opened",
    "bureau_days_credit_max": "maximum number of days since credit bureau record was opened",
    "bureau_days_update_max": "maximum number of days since last credit bureau record update",
    "bureau_annuity_sum": "total annuity payments from all credit bureau loan records",
    "bureau_bb_dpd_max": "maximum days past due delinquent overdue late payment default missed payment credit bureau",
    "bureau_bb_months_min": "minimum number of months in credit bureau balance history",
    "bureau_debt_credit_ratio": "ratio of total debt to total credit from credit bureau",
    "prev_app_count": "total number of previous loan applications",
    "prev_approved_count": "number of previously approved loan applications",
    "prev_refused_count": "number of previously refused loan applications credit risk rejection denied delinquent default history",
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
    "bureau_credit_enddate_mean": "average end date of credit loans days overdue delinquent past due payment history",
    }   ,

    "CreditScoring": {
    # → DAYS_BIRTH
    "Age": "age of the customer in years old person how old birth date days born borrower years since birth DAYS BIRTH",

    # → AMT_INCOME_TOTAL (а не CNT_FAM_MEMBERS!) — УБИРАЕМ ВСЕ НЕГАТИВЫ
    "Annual_Income": "yearly annual income salary earnings total sum money amount per year customer AMT INCOME TOTAL revenue",

    # → AMT_INCOME_TOTAL (а не prev_cnt_payment_mean!)
    "Monthly_Inhand_Salary": "monthly income salary take home pay after tax net earnings customer AMT INCOME money per month NOT payment count NOT mean NOT installment",

    # → AMT_INCOME_TOTAL (слабо, но closest)
    "Num_Bank_Accounts": "number count of bank accounts held by the customer how many banking accounts open AMT INCOME financial accounts",

    # → bureau_amt_credit_sum
    "Num_Credit_Card": "number count of credit cards owned by the customer how many cards bureau amt credit sum card count",

    # → HOUR_APPR_PROCESS_START (единственное числовое про время)
    "Interest_Rate": "interest rate percent per year annual percentage rate APR cost of borrowing money loan price HOUR time",

    # → bureau_loan_count или DAYS_ID_PUBLISH (closest available)
    "Num_of_Loan": "total count number of loans taken by customer how many loans bureau loan count borrower has",

    # → prev_days_last_due_mean (closest про даты и просрочки)
    "Delay_from_due_date": "average days delayed past due date customer late payment days overdue delinquency delay prev days last due mean",

    # → DEF_30_CNT_SOCIAL_CIRCLE (closest про просрочки)
    "Num_of_Delayed_Payment": "count number of delayed payments made by customer delinquent late missed payments DEF SOCIAL CIRCLE delinquency count",

    # → prev_cnt_payment_max (closest про изменения)
    "Changed_Credit_Limit": "percentage change in credit card limit increase decrease prev cnt payment max modification borrowing limit",

    # → AMT_REQ_CREDIT_BUREAU_HOUR/DAY
    "Num_Credit_Inquiries": "number count of credit inquiries applications AMT REQ CREDIT BUREAU HOUR DAY made by customer recent period pull requests",

    # → AMT_CREDIT (прямой матч по сумме)
    "Outstanding_Debt": "total outstanding debt amount unpaid balance AMT CREDIT money owed debt remaining sum",

    # → bureau_debt_credit_ratio
    "Credit_Utilization_Ratio": "ratio percent of credit used to total available credit limit utilization revolving bureau debt credit ratio usage percentage",

    # → bureau_annuity_sum
    "Total_EMI_per_month": "total monthly equated installment payments EMI per month bureau annuity sum regular loan payment monthly payment amount",

    # → prev_amt_annuity_sum
    "Amount_invested_monthly": "monthly amount invested savings deposit portfolio prev amt annuity sum customer money put aside investments per month",

    # → AMT_ANNUITY (closest)
    "Monthly_Balance": "monthly balance amount remaining after expenses customer leftover money surplus cash AMT ANNUITY per month",

    # → bureau_days_credit_mean
    "credit_history_months": "length duration of credit history in months how long customer has credit file bureau days credit mean months since first credit line",

    # → DAYS_REGISTRATION (closest по сроку)
    "num_loan_types": "number count of different types kinds of loans held by customer variety loan products DAYS REGISTRATION duration",

    # → prev_days_decision_min (closest по времени)
    "month_num": "month number in observation period calendar month index time point prev days decision min date sequence",

    # → bureau_debt_credit_ratio
    "emi_to_income": "ratio of total EMI payments to monthly income percent of income going to loan payments bureau debt credit ratio proportion",

    # → prev_amt_credit_sum
    "debt_to_income": "ratio percent of total outstanding debt to annual income debt burden income ratio prev amt credit sum",

    # → bureau_active_ratio (closest по ratio)
    "invest_to_income": "ratio percent of monthly investments to monthly income savings rate investment bureau active ratio proportion",

    # → prev_cnt_payment_mean (closest по среднему)
    "balance_to_income": "ratio percent of monthly balance surplus to monthly income leftover income prev cnt payment mean ratio proportion",

    # → bureau_days_credit_mean (closest по log)
    "log_annual_income": "natural logarithm of annual income log transform of yearly income salary bureau days credit mean log earnings",

    # → bureau_amt_debt_sum
    "log_outstanding_debt": "natural logarithm log transform of outstanding debt amount bureau amt debt sum log debt",

    # → bureau_amt_credit_max
    "credit_cards_per_bank": "ratio of credit cards to number of bank accounts cards per account density bureau amt credit max",

    # → OCCUPATION_TYPE
    "Occupation": "occupation profession job type work employment OCCUPATION TYPE what customer does for work career profession employment job title",

    # → Нет аналога (уникальные для CreditScoring)
    "Credit_Mix": "mix diversity of credit types portfolio loans credit cards mortgage revolving installment variety mix types different credit products",
    "Payment_of_Min_Amount": "whether customer pays only minimum amount due on credit card minimum payment flag behavior pay pattern",
    "Payment_Behaviour": "spending payment behaviour pattern customer how pays high low spent medium purchases payment behavior spending habits",

    # → FLAG_OWN_CAR (а не FLAG_PHONE!) — УБИРАЕМ НЕГАТИВЫ
    "has_auto_loan": "binary flag customer has auto car vehicle loan automobile financing FLAG OWN CAR vehicle automobile motor transport owner car ownership",

    # → prev_insured_mean (closest страховка)
    "has_personal_loan": "binary flag customer has personal unsecured loan borrowed money consumer loan prev insured mean personal individual",

    # → bureau_active_ratio (closest)
    "has_student_loan": "binary flag customer has student education loan university college tuition loan bureau active ratio education study",

    # → FLAG_OWN_REALTY
    "has_home_equity_loan": "binary flag customer has home equity loan property equity line of credit HELOC FLAG OWN REALTY home house property",

    # → FLAG_OWN_REALTY (а не FLAG_PHONE!) — МЕНЯЕМ
    "has_mortgage_loan": "binary flag customer has mortgage home loan real estate property FLAG OWN REALTY house apartment building",

    # → bureau_days_credit_min (closest)
    "has_payday_loan": "binary flag customer has payday loan short term high interest small cash advance loan bureau days credit min short small",

    # → FLAG_DOCUMENT_3 (closest)
    "has_debt_consolidation_loan": "binary flag customer has debt consolidation loan combine debts restructure FLAG DOCUMENT merge consolidate",

    # → FLAG_DOCUMENT_3 или REGION_RATING
    "has_credit_builder_loan": "binary flag customer has credit builder loan to build credit history secured small loan FLAG DOCUMENT build construct",

    # → REGION_RATING_CLIENT (прямой матч)
    "REGION_RATING_CLIENT": "credit rating of the region where customer lives REGION RATING CLIENT area location score",

    # → REGION_RATING_CLIENT_W_CITY (прямой матч)
    "REGION_RATING_CLIENT_W_CITY": "credit rating of the region including city where customer lives REGION RATING CLIENT CITY urban area score",
    },

    "Application": {
        "id": "unique application identifier number",
        "age": "age of the applicant in years at time of application",
        "Score_bki": "credit bureau risk score creditworthiness rating financial history assessment",
        "log_income": "natural logarithm of applicant monthly income salary earnings AMT INCOME TOTAL log transform money per month",
        "log_appl_rej": "natural logarithm of number of previous application rejections",
        "log_out_req": "natural logarithm of number of outstanding credit requests",
        "region_rating_ord": "ordinal credit rating of applicant region from 1 to 3",
        "home_address_cd": "code type of home address registration residence FLAG OWN REALTY REG REGION home house apartment where lives address location",
        "work_address_cd": "code type of work address registration employment DAYS EMPLOYED job office work location career",
        "SNA": "social network analysis score of the applicant",
        "first_time_cd": "binary flag first time credit applicant new borrower no previous loans prev app count first application ever",
        "good_work_flg": "binary flag indicating applicant has stable good employment",
        "app_month": "month of loan application submission",
        "app_quarter": "quarter of loan application submission",
        "education_cd": "education level code graduate university high school NAME EDUCATION TYPE academic degree diploma",
        "gender_cd": "CODE GENDER gender male or female of the applicant sex",
        "car_own_flg": "binary flag applicant owns a car vehicle FLAG OWN CAR automobile owner car",
        "car_type_flg": "binary flag indicating type of car owned by applicant",
        "Air_flg": "binary flag indicating applicant has air travel history",
    },

    "TaiwanCC": {
    "LIMIT_BAL": "credit limit amount credit card limit maximum allowed credit balance outstanding LIMIT BAL",
    "AGE": "age of the credit card holder in years old person DAYS BIRTH birth date days born how old",

    # PAY_* — СТАТУСЫ (не суммы!), каждый с уникальным месяцем
    "PAY_0": "SEPTEMBER month zero repayment status code delay category NOT amount NOT sum NOT money recent first",
    "PAY_2": "AUGUST month two repayment status code delay category NOT amount NOT sum two months ago",
    "PAY_3": "JULY month three repayment status code delay category NOT amount NOT sum three months ago",
    "PAY_4": "JUNE month four repayment status code delay category NOT amount NOT sum four months ago",
    "PAY_5": "MAY month five repayment status code delay category NOT amount NOT sum five months ago",
    "PAY_6": "APRIL month six repayment status code delay category NOT amount NOT sum oldest six months ago",

    # BILL_AMT* — СУММЫ счетов
    "BILL_AMT1": "SEPTEMBER month one bill statement amount dollars money debt balance recent first",
    "BILL_AMT2": "AUGUST month two bill statement amount dollars money debt balance two months ago",
    "BILL_AMT3": "JULY month three bill statement amount dollars money debt balance three months ago",
    "BILL_AMT4": "JUNE month four bill statement amount dollars money debt balance four months ago",
    "BILL_AMT5": "MAY month five bill statement amount dollars money debt balance five months ago",
    "BILL_AMT6": "APRIL month six bill statement amount dollars money debt balance oldest six months ago",

    # PAY_AMT* — СУММЫ платежей
    "PAY_AMT1": "SEPTEMBER month one payment amount dollars money paid previous month recent first",
    "PAY_AMT2": "AUGUST month two payment amount dollars money paid two months ago",
    "PAY_AMT3": "JULY month three payment amount dollars money paid three months ago",
    "PAY_AMT4": "JUNE month four payment amount dollars money paid four months ago",
    "PAY_AMT5": "MAY month five payment amount dollars money paid five months ago",
    "PAY_AMT6": "APRIL month six payment amount dollars money paid oldest six months ago",

    # Категориальные
    "SEX": "CODE GENDER gender male female borrower",
    "EDUCATION": "NAME EDUCATION TYPE education level graduate school university high school academic degree diploma",
    "MARRIAGE": "NAME FAMILY STATUS marital status married single divorced family status",

    # Агрегированные
    "max_pay_delay": "maximum repayment delay months overdue delinquent past due worst bureau bb dpd max",
    "mean_pay_delay": "average repayment delay months overdue delinquent past due payment history mean",
    "n_delayed": "number of months with payment delay delinquent overdue missed installment count DEF SOCIAL CIRCLE",
    "mean_bill": "average monthly bill statement amount outstanding balance credit card bureau amt credit mean",
    "max_bill": "maximum monthly bill statement amount outstanding balance credit card bureau amt credit max",
    "bill_trend": "difference most recent oldest bill amount debt trend increasing decreasing balance change",
    "mean_pay_amt": "average monthly payment amount made credit card bureau amt credit sum mean payment",
    "total_pay_amt": "total payment amount made all months credit card bureau amt credit sum total payments",
    "util_ratio": "ratio bill amount to credit limit utilization credit usage revolving bureau debt credit ratio",
    "pay_ratio_1": "ratio payment made to bill amount most recent month payment behavior prev credit ratio",
    "log_limit_bal": "logarithm credit limit balance log transformed bureau days credit mean limit",
    "log_mean_bill": "logarithm average bill statement amount log transformed outstanding debt bureau mean",
    "log_mean_pay": "logarithm average payment amount made log transformed monthly payment bureau mean pay",
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
