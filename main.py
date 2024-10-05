import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction . text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('IMDB-Dataset.csv')

df = df.drop_duplicates()

stop = set(stopwords.words('english'))

# Expanding contractions
def expand_contractions(text):
    return contractions.fix(text)

# Function to clean data
def preprocess_text(text):

    wl = WordNetLemmatizer()

    soup = BeautifulSoup(text ,"html.parser")
    text = soup.get_text()
    
    text = expand_contractions(text)# Expanding chatwords and contracts clearingcontractions
    emoji_clean = re.compile("["
                            u"\U0001F600-\U0001F64F"# emoticons
                            u"\U0001F300-\U0001F5FF"# symbols & pictographs
                            u"\U0001F680-\U0001F6FF"# transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                          "]+", flags =re.UNICODE)
    text = emoji_clean.sub(r'',text)
    text = re.sub(r'\.(?=\ S)','.',text)
    text = re.sub(r'http \S+','', text)
    text ="".join([
        word.lower()for word in text if word not in string.punctuation
        ])# remove punctuation and make text lowercase
    text ="".join([
        wl.lemmatize(word)for word in text.split()if word not in stop and word .isalpha()])# lemmatize
    return text

df['review'] = df['review'].apply(preprocess_text)


def func(pct,allvalues):
    absolute = int(pct / 100.* np.sum(allvalues))
    return"{:.1f}%\n({:d})".format(pct , absolute)

freq_pos = len(df[df['sentiment'] =='positive'])
freq_neg = len(df[df['sentiment'] =='negative'])

data = [ freq_pos , freq_neg ]

labels = ['positive','negative']
# Create pie chart
pie , ax = plt.subplots(figsize =[11 ,7])
plt.pie(x=data , autopct = lambda pct: func(pct , data), explode =[0.0025]*2 ,
pctdistance =0.5 , colors =[ sns.color_palette()[0] ,'tab:red'] , textprops ={'fontsize': 16})
# plt.title('Frequencies of sentiment labels', fontsize =14 , fontweight ='bold')
labels = [r'Positive', r'Negative']
plt.legend(labels , loc ="best", prop ={'size': 14})
pie.savefig("PieChart.png")
plt.show()



words_len = df['review'].str.split().map(lambda x: len(x))
df_temp = df.copy()
df_temp ['words length'] = words_len

hist_positive = sns.displot(
                    data = df_temp [ df_temp ['sentiment'] =='positive'] ,
                    x="words length", hue="sentiment", kde=True , height =7 , aspect =1.1 , legend =False
                    ).set(title ='Words in positive reviews')
plt.show(hist_positive)

hist_negative = sns.displot(
                    data = df_temp [ df_temp ['sentiment'] =='negative'] ,
                    x="words length", hue="sentiment", kde=True , height =7 , aspect =1.1 , legend =False , palette =['red']
                    ).set(title ='Words in negative reviews')
plt.show(hist_negative)

plt.figure(figsize =(7 ,7.1))
kernel_distibution_number_words_plot = sns.kdeplot(
                    data = df_temp , x="words length", hue="sentiment", fill =True , palette =[sns.color_palette()[0] ,'red']
                    ).set(title ='Words in reviews')

plt.legend(title ='Sentiment', labels =['negative','positive'])
plt.show(kernel_distibution_number_words_plot)



label_encode = LabelEncoder()
df['sentiment'] = label_encode.fit_transform(df['sentiment'])

x_train , x_test , y_train , y_test = train_test_split(df , test_size =0.2 , random_state =42)

tfidf_vectorizer = TfidfVectorizer( max_features =10000)
tfidf_vectorizer.fit(x_train , y_train)

x_train_encoded = tfidf_vectorizer.transform(x_train)
x_test_encoded = tfidf_vectorizer.transform(x_test)


dt_classifier = DecisionTreeClassifier(
                                        criterion ='entropy ',
                                        random_state =42
                                        )
dt_classifier.fit(x_train_encoded , y_train)
y_pred = dt_classifier.predict(x_test_encoded)
accuracy_score(y_pred , y_test )


rf_classifier = RandomForestClassifier(random_state =42)
rf_classifier.fit(x_train_encoded , y_train)
y_pred = rf_classifier.predict( x_test_encoded )
accuracy_score (y_pred , y_test )