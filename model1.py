# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# %matplotlib inline

# import os
# print(os.listdir())

# import warnings
# warnings.filterwarnings('ignore')
# dataset = pd.read_csv("heart.csv")

# type(dataset)

# dataset.shape

# dataset.head(5)


# dataset.sample(5)

# dataset.describe()

# dataset.info()

# info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



# for i in range(len(info)):
#     print(dataset.columns[i]+":\t\t\t"+info[i])
    
    
# dataset["target"].describe()




# dataset["target"].unique()

# print(dataset.corr()["target"].abs().sort_values(ascending=False))


# y = dataset["target"]

# sns.countplot(y)


# target_temp = dataset.target.value_counts()

# print(target_temp)


# print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
# print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))

# dataset["sex"].unique()

# sns.barplot(dataset["sex"],y)

# dataset["cp"].unique()
# sns.barplot(dataset["cp"],y)
# dataset["fbs"].describe()
# dataset["fbs"].unique()
# sns.barplot(dataset["fbs"],y)
# dataset["restecg"].unique()
# sns.barplot(dataset["restecg"],y)
# dataset["exang"].unique()
# sns.barplot(dataset["exang"],y)
# dataset["slope"].unique()
# sns.barplot(dataset["slope"],y)
# dataset["ca"].unique()
# sns.countplot(dataset["ca"])
# sns.barplot(dataset["ca"],y)
# dataset["thal"].unique()
# sns.barplot(dataset["thal"],y)
# sns.distplot(dataset["thal"])


# from sklearn.model_selection import train_test_split

# predictors = dataset.drop("target",axis=1)
# target = dataset["target"]

# X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
# X_train.shape


# X_test.shape

# Y_train.shape


# Y_test.shape



# from sklearn.metrics import accuracy_score



# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression()

# lr.fit(X_train,Y_train)

# Y_pred_lr = lr.predict(X_test)


# score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

# print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")




# from sklearn.naive_bayes import GaussianNB

# nb = GaussianNB()

# nb.fit(X_train,Y_train)

# Y_pred_nb = nb.predict(X_test)

# Y_pred_nb.shape

# score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

# print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

# from sklearn import svm

# sv = svm.SVC(kernel='linear')

# sv.fit(X_train, Y_train)

# Y_pred_svm = sv.predict(X_test)

# from sklearn import svm

# sv = svm.SVC(kernel='linear')

# sv.fit(X_train, Y_train)

# Y_pred_svm = sv.predict(X_test)


# score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

# print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train,Y_train)
# Y_pred_knn=knn.predict(X_test)



# Y_pred_knn.shape





# score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

# print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")





# from sklearn.tree import DecisionTreeClassifier

# max_accuracy = 0


# for x in range(200):
#     dt = DecisionTreeClassifier(random_state=x)
#     dt.fit(X_train,Y_train)
#     Y_pred_dt = dt.predict(X_test)
#     current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
#     if(current_accuracy>max_accuracy):
#         max_accuracy = current_accuracy
#         best_x = x
        
# #print(max_accuracy)
# #print(best_x)


# dt = DecisionTreeClassifier(random_state=best_x)
# dt.fit(X_train,Y_train)
# Y_pred_dt = dt.predict(X_test)
# print(Y_pred_dt.shape)
# score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

# print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
# from sklearn.ensemble import RandomForestClassifier

# max_accuracy = 0


# for x in range(2000):
#     rf = RandomForestClassifier(random_state=x)
#     rf.fit(X_train,Y_train)
#     Y_pred_rf = rf.predict(X_test)
#     current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
#     if(current_accuracy>max_accuracy):
#         max_accuracy = current_accuracy
#         best_x = x
        
# #print(max_accuracy)
# #print(best_x)

# rf = RandomForestClassifier(random_state=best_x)
# rf.fit(X_train,Y_train)
# Y_pred_rf = rf.predict(X_test)

# Y_pred_rf.shape
# score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

# print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")
# import xgboost as xgb

# xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
# xgb_model.fit(X_train, Y_train)

# Y_pred_xgb = xgb_model.predict(X_test)
# Y_pred_xgb.shape


# score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

# print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")

# from keras.models import Sequential
# from keras.layers import Dense


# # https://stats.stackexchange.com/a/136542 helped a lot in avoiding overfitting

# model = Sequential()
# model.add(Dense(11,activation='relu',input_dim=13))
# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# model.fit(X_train,Y_train,epochs=300)

# Y_pred_nn = model.predict(X_test)




# Y_pred_nn.shape

# rounded = [round(x[0]) for x in Y_pred_nn]

# Y_pred_nn = rounded

# score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

# print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

# #Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11. 
# scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_xgb,score_nn]
# algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]    

# for i in range(len(algorithms)):
#     print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# sns.set(rc={'figure.figsize':(15,8)})
# plt.xlabel("Algorithms")
# plt.ylabel("Accuracy score")

# sns.barplot(algorithms,scores)











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
import joblib
from tqdm import tqdm  # For progress bar

# Ignore warnings
warnings.filterwarnings("ignore")

# Debugging statements
print("Script started...")

# Step 1: Load dataset
print("Loading dataset...")
dataset_path = "data/heart.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
dataset = pd.read_csv(dataset_path)
print(f"Dataset loaded successfully with shape {dataset.shape}")

# Step 2: Display basic dataset info
print("Displaying dataset info:")
print(dataset.head())
print("Correlation with target:\n", dataset.corr()["target"].abs().sort_values(ascending=False))

# Step 3: Visualize target distribution
print("Plotting target distribution...")
sns.countplot(dataset["target"])
plt.title("Distribution of Target")
plt.show()

# Step 4: Prepare data for training
print("Preparing data for training...")
predictors = dataset.drop("target", axis=1)
target = dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Step 5: Define and train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": svm.SVC(kernel='linear'),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42)
}

model_scores = {}

print("Training models...")
for name, model in tqdm(models.items(), desc="Training Progress", unit="model"):
    print(f"Training {name}...")
    model.fit(X_train, Y_train)
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")
    predictions = model.predict(X_test)
    accuracy = round(accuracy_score(predictions, Y_test) * 100, 2)
    model_scores[name] = accuracy
    print(f"{name} trained with accuracy: {accuracy}%")

# Step 6: Train Neural Network
print("Training Neural Network...")
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=13))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model with a progress bar
for epoch in tqdm(range(300), desc="Neural Network Training", unit="epoch"):
    model.fit(X_train, Y_train, epochs=1, verbose=0)  # Training one epoch at a time

model.save("neural_network_model.h5")
rounded_preds = [round(x[0]) for x in model.predict(X_test)]
nn_accuracy = round(accuracy_score(rounded_preds, Y_test) * 100, 2)
model_scores["Neural Network"] = nn_accuracy
print(f"Neural Network trained with accuracy: {nn_accuracy}%")

# Step 7: Display results
print("\nFinal Results:")
for name, score in model_scores.items():
    print(f"The accuracy score achieved using {name} is: {score}%")

# Visualization
print("Visualizing results...")
algorithms = list(model_scores.keys())
scores = list(model_scores.values())

sns.barplot(algorithms, scores)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.show()

print("Script completed successfully!")
