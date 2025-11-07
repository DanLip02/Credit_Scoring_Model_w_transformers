from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_german_credit_risk_UCI():

    # fetch dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)

    # data (as pandas dataframes)
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

    # Переименовываем колонки
    df = X.rename(columns=column_rename)
    #
    # print("Переименованные колонки:")
    # print(df.columns.tolist())
    # print(f"\nРазмер данных: {df.shape}")
    finall = pd.concat([df, y], axis=1)

    return finall

if __name__ == '__main__':
    load_german_credit_risk_UCI()