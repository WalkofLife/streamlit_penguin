import pandas as pd
penguins = pd.read_excel('penguins_cleaned.xlsx')

df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix= col)
    df = pd.concat([df, dummy], axis = 1)
    del df[col]

target_mapper = {'Adelie' : 0, 'Chinstrap' : 1, 'Gentoo' : 2}
def target_encode(val):
    return target_mapper[val]

df[target] =df[target].apply(target_encode)

# Seperating X & y
X = df.drop(target, axis = 1)
y = df[target]

# Build Random Forest Model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Save the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))