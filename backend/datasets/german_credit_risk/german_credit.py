from ucimlrepo import fetch_ucirepo
import pandas as pd
import ssl
import certifi
import io
import urllib

def load_german_credit_risk_UCI_old():

    # fetch dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)

    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets

    # metadata
    print(statlog_german_credit_data.metadata)

    # variable information
    print(statlog_german_credit_data.variables)

    column_rename = {
        'Attribute1': 'status_checking_account',
        'Attribute2': 'duration_months',
        'Attribute3': 'credit_history',
        'Attribute4': 'purpose',
        'Attribute5': 'credit_amount',
        'Attribute6': 'savings_account',
        'Attribute7': 'employment_since',
        'Attribute8': 'installment_rate',
        'Attribute9': 'personal_status_sex',
        'Attribute10': 'other_debtors',
        'Attribute11': 'present_residence',
        'Attribute12': 'property',
        'Attribute13': 'age',
        'Attribute14': 'other_installment_plans',
        'Attribute15': 'housing',
        'Attribute16': 'existing_credits',
        'Attribute17': 'job',
        'Attribute18': 'num_dependents',
        'Attribute19': 'telephone',
        'Attribute20': 'foreign_worker'
    }

    df = X.rename(columns=column_rename)
    #
    # print(df.columns.tolist())
    # print(f"\nРазмер данных: {df.shape}")
    finall = pd.concat([df, y], axis=1)

    return finall

def load_german_credit_risk_UCI():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    columns = [
        "status_existing_checking", "duration_month", "credit_history", "purpose",
        "credit_amount", "savings", "employment", "installment_rate",
        "personal_status_sex", "other_debtors", "residence_since", "property",
        "age", "other_installment_plans", "housing", "existing_credits",
        "job", "people_liable", "telephone", "foreign_worker", "target"
    ]

    ctx = ssl.create_default_context(cafile=certifi.where())

    raw = urllib.request.urlopen(url, context=ctx).read().decode("utf-8")

    df = pd.read_csv(io.StringIO(raw), sep=" ", header=None, names=columns)

    df.head()

    # df.to_csv("german_credit_risk.csv")

    # return df

if __name__ == '__main__':

    # load_german_credit_risk_UCI_old()

    load_german_credit_risk_UCI()