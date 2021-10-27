import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("aviation-flight-information.csv")

types_dict = {'Arrive or Depart': str, 'Schedule': str, 'Airline': str, 'Flight Number': str, 'Gate': str,
              'Terminal': float}
for col, col_type in types_dict.items():
    df[col] = df[col].astype(col_type)

le = preprocessing.LabelEncoder()

arriveOrDepart_enc = le.fit_transform(df["Arrive or Depart"].values)
# print(arriveOrDepart_enc)

airline_enc = le.fit_transform(df["Airline"].values)
# print(airline_enc)

gate_enc = le.fit_transform(df["Gate"].values)
# print(gate_enc)

terminal_enc = le.fit_transform(df["Terminal"].values)
# print(terminal_enc)

y = terminal_enc

X = list(zip(gate_enc, arriveOrDepart_enc, airline_enc))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=22)

# print("X_train", X_train)
# print("X_test", X_test)
# print("y_train", y_train)
# print("y_test", y_test)

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))
df["Terminal Prediction Numerical"] = rfc.predict(X)

# df["Terminal Prediction Numerical"].to_csv("prediction.csv", encoding='utf-8', index=False)

df.assign(TerminalPrediction=pd.cut(df["Terminal Prediction Numerical"], bins=[-1, 0, 1, 2, 3],
                                    labels=['1', '2', '3', '4'])).to_csv("prediction.csv",
                                                                         encoding='utf-8', index=False)

rfc_imp = pd.DataFrame(rfc.feature_importances_, columns=['Importance'])
rfc_imp["Importance"] = rfc_imp["Importance"] * 100
print(rfc_imp["Importance"])
