import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer
from embeddings import convert_to_embeddings

df = pd.read_csv("./data/SMSSpamCollection",
                  sep = "\t", 
                  names = ["types", "message"])

df["spam"] = df["types"] == "spam"
df.drop("types", axis = 1, inplace = True)

df_train = df.sample(frac=0.8, random_state = 42)
df_val = df.drop(index = df_train.index)


X_train = convert_to_embeddings(df_train["message"].tolist())
X_val = convert_to_embeddings(df_val["message"].tolist())


y_train = torch.tensor(df_train["spam"].values, dtype = torch.float32).reshape((-1,1))

y_val = torch.tensor(df_val["spam"].values, dtype = torch.float32).reshape((-1,1))

model = nn.Linear(768,1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(0,10000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if i % 1000:
        print(loss)

def evaluate_model(X,y):
    model.eval()
    with torch.no_grad():
        y_pred = nn.functional.sigmoid(model(X)) > 0.5
        print("Accuracy:", (y_pred == y).type(torch.float32).mean().item())
        print("Sensitivity:", (y_pred[y==1] == y[y== 1]).type(torch.float32).mean().item())
        print("Specificity:", (y_pred[y==0] == y[y== 0]).type(torch.float32).mean().item())
        print("Precision:", (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean().item())
        
print("Evaluating on the training data: ") 
evaluate_model(X_train, y_train)

print("Evaluating on the validation data: ")
evaluate_model(X_val, y_val)

custom_messages = [
    "Winner! Great deal, call us to get this produc for free!",
    "Tomorrow is my birthday, do you want to come?"
]


X_custom = convert_to_embeddings(custom_messages)
model.eval()
with torch.no_grad():
    pred = nn.functional.sigmoid(model(X_custom))
    print(pred)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'X_train': X_train,
    'y_train': y_train
}, "checkpoint.pt")

# Evaluating on the training data: 
# Accuracy: 0.9964109659194946
# Sensitivity: 0.9763912558555603
# Specificity: 0.9994825124740601
# Precision: 0.9965576529502869
# Evaluating on the validation data: 
# Accuracy: 0.9874326586723328
# Sensitivity: 0.9220778942108154
# Specificity: 0.9979166388511658
# Precision: 0.9861111044883728