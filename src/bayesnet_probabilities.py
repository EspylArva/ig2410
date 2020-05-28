import pandas as pd
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split

"""
    input: dataset
    output: probabilities expected by part 2: Bayes Nets
    --
    The following probabilities are computed:
    - P(Symptoms_discovered = True | Visited_Wuhan = True)
    - P(CoVid_confirmed = True | (Symptoms_discovered = True & Visited_Wuhan = True))
    - P(Outcome = Death | Visited_Wuhan = True)
"""


def computeProbabilities(df):
    n_total = len(df.index)
    n_symptom_wuhan = len(df[(df['date_onset_symptoms'] == 1) & (df['travel_history_location'] == 1)])
    n_wuhan = len(df[df['travel_history_location'] == 1])
    n_truepatient_symptom_wuhan = len(df[(df['date_confirmation'].isnull() == False) & (df['date_onset_symptoms'] == 1) & (df['travel_history_location'] == 1)])
    n_dead_wuhan = len(df[(df['outcome'] == 1) & (df['travel_history_location'] == 1)])
    print("=== Probabilities ===")
    print("=== Units - N(X): [case] - P(X): [%]  ===")
    print("N_total:", n_total)
    print("=== Q1 ===")
    print("N(S=1,W=1):", n_symptom_wuhan)
    print("N(W=1):", n_wuhan)
    print("----------")
    print("P(S=1,W=1):", 100*n_symptom_wuhan/n_total)
    print("P(W=1):", 100*n_wuhan/n_total)
    print("P(S=1|W=1) = P(S=1,W=1)/P(W=1):", 100*n_symptom_wuhan/n_wuhan)
    print("=== Q2 ===")
    print("N(C=1,S=1,W=1):", n_truepatient_symptom_wuhan)
    print("N(S=1,W=1):", n_symptom_wuhan)
    print("----------")
    print("P(C=1,S=1,W=1):", 100*n_truepatient_symptom_wuhan/n_total)
    print("P(S=1,W=1):", 100*n_symptom_wuhan/n_total)
    print("P(C=1|S=1,W=1) = P(C=1,S=1,W=1)/P(S=1,W=1):", 100*n_truepatient_symptom_wuhan/n_symptom_wuhan)
    print("=== Q3 ===")
    print("N(O=1,W=1):", n_dead_wuhan)
    print("N(W=1):", n_wuhan)
    print("----------")
    print("P(O=1,W=1):", 100*n_dead_wuhan/n_total)
    print("P(W=1):", 100*n_wuhan/n_total)
    print("P(O=1|W=1) = P(O=1,W=1)/P(W=1):", 100*n_dead_wuhan/n_wuhan)

    dead = df['outcome'] == 0  # didnt die
    wuhan = df['travel_history_location'] == 1  # visited wuhan
    filter_ = dead & wuhan
    recoveries = df.loc[filter_,['date_death_or_discharge', 'date_confirmation']]
    recoveries['date_death_or_discharge'] = pd.to_datetime(recoveries['date_death_or_discharge'], format='%d.%m.%Y')
    recoveries['date_confirmation'] = pd.to_datetime(recoveries['date_confirmation'], format='%d.%m.%Y')
    rangeRecovery = (recoveries['date_death_or_discharge'] - recoveries['date_confirmation']).dt.days
    print(type(rangeRecovery), rangeRecovery)
    print("Recovery for a person who visited Wuhan:")
    print("Range (in days): [{min}, {max}]. Average time for a recovery: {average} days".format(min=rangeRecovery.min(), max=rangeRecovery.max(), average=rangeRecovery.mean()))


def bayesPrediction(X, y, prediction, typePrediction):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    BernNB = naive_bayes.BernoulliNB()
    BernNB.fit(X_train,y_train)
    if typePrediction == 'proba':
        return (BernNB.predict_proba([prediction]))
    else:
        return (BernNB.predict([prediction]))