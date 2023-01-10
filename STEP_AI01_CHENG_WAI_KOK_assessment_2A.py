#%%
#1. Import packages
from modules import text_cleaning
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, plot_model
import datetime, os, json, nltk, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
#2. Set parameters
learning_rate = 0.01
num_words = 2000 # unique number of words in all the sentences
oov_token = '<OOV>' # out of vocabulary
SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
embedding_layer = 64
nltk.download('stopwords')
stop_words = stopwords.words('english')
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

#%%
# 3. Data Loading
df = pd.read_csv(URL)

#%%
# 3.1 Data Inspection
df.describe()
df.info()
df.head()

#%%
#3.2 Inspect duplicated data here = 99
df.duplicated().sum()

#%%
#3.3 To check NaN = 0
df.isna().sum()

#%%
#3.4 To view the text
print(df['text'][0])

#%%
#4. Data Cleaning
for index, temp in enumerate(df['text']):
    df['text'][index] = text_cleaning(temp)
    

#%%
#5. Features Selection
text = df['text']
category = df['category']

# Filter the text using stopwords
text = [word for word in text if word.lower() not in stop_words]
#print(text)

#%%
#6. Data Preprocessing
#6.1 Tokenizer
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token) # instantiate

tokenizer.fit_on_texts(text)

#To transform the text using tokenizer --> mms.transform
text = tokenizer.texts_to_sequences(text)

#%%
#6.2 Padding
text = pad_sequences(text,maxlen=200, padding='post', truncating='post')

#%%
#6.3 One hot encoder
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(category[::, None])

#%%
#6.4 Train-test split
# expand dimension before feeding to train_test_split
#padded_text = np.expand_dims(padded_text, axis=-1)

X_train,x_test,y_train,y_test = train_test_split(text, category, shuffle=True, test_size=0.2, random_state=SEED)

#%%
#7. Model Development
model = Sequential()
model.add(Embedding(num_words,embedding_layer))
model.add(LSTM(embedding_layer, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(embedding_layer))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))

model.summary()

plot_model(model,show_shapes=True)

#%%
#8. Model Compilation
optimizer = keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics='acc')

#%%
#9. Define the callbacks function to use
es = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

log_path = os.path.join('log_dir',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = keras.callbacks.TensorBoard(log_path)

#%%
#10. Model Training
hist = model.fit(X_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es,tb])

#%%
#11. Model Analysis
hist.history.keys()
loss, accuracy= model.evaluate(x_test,y_test,verbose=0)

print('Accuracy:', accuracy)
print('Loss rate:', loss)

#%%
#11.1 Plot the graph
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()

#%%
#12. Make predictions on the test data
y_predicted = model.predict(x_test)
y_predicted = np.argmax(y_predicted,axis=1)
y_test = np.argmax(y_test,axis=1)
f1 = f1_score(y_test,y_predicted,average='macro')
print("F1 Score: ",f1)

#%%
#13. Print Classification Report
print(classification_report(y_test, y_predicted))
cm = (confusion_matrix(y_test, y_predicted))

#%%
#14. Plot Confusion Matrix
disp = ConfusionMatrixDisplay(cm)
disp.plot()

#%%
#15. Model Saving
# To create folder if not exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

saved_path = os.path.join('saved_models','model.h5')
model.save(saved_path)

#%%
#16. To save one hot encorder model
with open('ohe.pkl','wb') as f:
    pickle.dump(ohe, f)

# %%
#17. To save tokenizer
token_json = tokenizer.to_json()
with open('saved_models/tokenizer.json','w') as f:
    #json.dump(token_json,f)
    json.dump(tokenizer.to_json(),f)

# %%
