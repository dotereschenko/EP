from prefect import flow,task
import zipfile
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


@task
def get_data():
    command = "kaggle competitions download -c spaceship-titanic"
    subprocess.run(command, shell=True)
    with zipfile.ZipFile('spaceship-titanic.zip', 'r') as zip_ref:
        zip_ref.extractall('spaceship-titanic')
        print("Unzipped!")

@task
def load_pd_process():
    df_train = pd.read_csv("spaceship-titanic/train.csv")
    df_test = pd.read_csv("spaceship-titanic/test.csv")
    df_train = df_train.drop(['PassengerId', 'Name'], axis=1)
    df_test = df_test.drop(['PassengerId', 'Name'], axis=1)
    df_train[df_train.isnull().any(axis=1)]

    rows_with_missing = df_train[df_train.isnull().any(axis=1)]
    num_resamples = len(rows_with_missing)
    resampled_rows = [df_train.dropna().sample(n=1).values.flatten() for _ in range(num_resamples)]
    resampled_df = pd.DataFrame(resampled_rows, columns=df_train.columns)
    df_train.loc[rows_with_missing.index] = resampled_df.values

    df_train["Transported"] = df_train["Transported"].astype(int)
    df_train['VIP'] = df_train['VIP'].astype(int)
    df_train['CryoSleep'] = df_train['CryoSleep'].astype(int)

    columns_to_replace = ['VIP', 'CryoSleep']
    df_test[columns_to_replace] = df_test[columns_to_replace].fillna(0)
    df_test['VIP'] = df_test['VIP'].astype(int)
    df_test['CryoSleep'] = df_test['CryoSleep'].astype(int)

    df_train[["Deck", "Cabin_num", "Side"]] = df_train["Cabin"].str.split("/", expand=True)
    df_test[["Deck", "Cabin_num", "Side"]] = df_test["Cabin"].str.split("/", expand=True)
    
    try:
        df_train = df_train.drop('Cabin', axis=1)
        df_test = df_test.drop('Cabin', axis=1)
    except KeyError:
        print("Field does not exist")
    
    enc = OrdinalEncoder()

    df_train["Destination"] = enc.fit_transform(df_train[["Destination"]])
    df_train["Deck"] = enc.fit_transform(df_train[["Deck"]])
    df_train["Side"] = enc.fit_transform(df_train[["Side"]])

    df_test["Destination"] = enc.fit_transform(df_test[["Destination"]])
    df_test["Deck"] = enc.fit_transform(df_test[["Deck"]])
    df_test["Side"] = enc.fit_transform(df_test[["Side"]])
    df_test["HomePlanet"] = enc.fit_transform(df_test[["HomePlanet"]])
    df_train["HomePlanet"] = enc.fit_transform(df_train[["HomePlanet"]])

    return df_train, df_test

@task
def training(df_train, df_test):
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop("Transported", axis=1), df_train["Transported"], test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]  

    }
    

    clf = RandomForestClassifier(random_state=0)
    

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    return X_train,X_test,y_train,y_test

@task
def save_data(X_train,X_test,y_train,y_test):
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

@flow(log_prints=True)
def hello_world(name: str = "world", goodbye: bool = False):
    print(f"Hello {name} from Prefect! 🤗")
    get_data()
    df_train,df_test = load_pd_process()
    X_train,X_test,y_train,y_test = training(df_train,df_test)
    save_data(X_train,X_test,y_train,y_test)

if __name__ == "__main__":
    hello_world.serve(name="my-first-deployment",
                      tags=["onboarding"],
                      parameters={"goodbye": True},
                      interval=60)

