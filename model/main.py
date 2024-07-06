
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib
## creating the model and running it 
def create_model(cancer_df):
    Y = cancer_df['diagnosis']
    X = cancer_df.drop(['diagnosis'], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    regressor = LogisticRegression()
    regressor.fit(X_train, Y_train)

    ## test the accuracy of the model
    Y_pred_test = regressor.predict(X_test)
    model_acc_test = accuracy_score(Y_test, Y_pred_test)
    print("The Model Accuracy for the X_Test data is: ", model_acc_test)

    return regressor, scaler

## cleaning the data to be used in prediction
def get_clean_data():
    cancer_df = pd.read_csv('../Data/breast_cancer_data.csv')
    cancer_df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    cancer_df.replace({'diagnosis':{'M':1, 'B':0}}, inplace=True)
    return cancer_df

# the main function 
def main():
    cancer_df = get_clean_data()

    regressor, scaler = create_model(cancer_df)


    filename = 'regressor.pkl'
    joblib.dump(regressor, filename)

    filename2 = 'scaler.pkl'
    joblib.dump(scaler, filename2)

    ## using the pickle5 instead of using joblib, but the outcome will be the same
    # with open('regressor.pkl', 'wb') as f:
    #     pickle.dump(regressor, f)
    
    # with open('scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)


if __name__ == '__main__':
    main()


## for exporting the fully trained model, you can use either the pickle or the joblib to transform the file into a .sav for easier training