import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import re 


lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
tfifd_Vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))


st.title('EMOTION CLASSIFIER')
st.subheader('Joy 😊 Happy Sad 😢 Sadness Angry 😡 Anger Love ❤️ Love Fear 😨 Fear Surprise 😯 Surprise', divider='rainbow')
st.write(':red[Enter] :orange[text] :green[to] :blue[get] :violet[the] :rainbow[Emotion]')
def clean(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text
def predict(text):
    text = clean(text)
    text = tfifd_Vectorizer.transform([text])
    prediction = lg.predict(text)[0]
    return lb.inverse_transform([prediction])[0]
    


with st.form("my_form"):
   # Every form must have a submit button.
    inp = st.text_input('Enter some Text 👇' , placeholder="I Love You")
    submitted = st.form_submit_button("Submit")
    if submitted:
       prediction = predict(inp)
       if(prediction=='sadness'):
        emoji = '😢'
       elif(prediction=='joy'):
        emoji = '😊'
       elif(prediction=='fear'):
        emoji = '😨'
       elif(prediction=='anger'):   
        emoji = '😡'
       elif(prediction=='surprise'):   
        emoji = '😯'
       elif(prediction=='love'):    
        emoji = '❤️'
       else:
          emoji =''

       st.subheader('prediction : ' + prediction+ " " + emoji)

