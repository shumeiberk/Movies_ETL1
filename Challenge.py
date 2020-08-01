import json
import pandas as pd
import numpy as np
import re
import time
import psycopg2
from sqlalchemy import create_engine
from config import db_password



# In[2]:


file_dir = "./Resources"


# In[3]:


with open (f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)


# In[4]:


len(wiki_movies_raw)


# In[5]:


# First 5 records
wiki_movies_raw[:5]


# In[6]:


# Last 5 records
wiki_movies_raw[-5:]


# In[7]:


# Some records in the middle
wiki_movies_raw[3600:3605]


# In[9]:


kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv')
ratings = pd.read_csv(f'{file_dir}/ratings.csv')


# In[10]:


kaggle_metadata.sample(n=5)


# In[12]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)


# In[13]:


wiki_movies_df.head()


# In[14]:


wiki_movies_df.columns.tolist()


# In[18]:


[movie for movie in wiki_movies_raw if ('Director' in movie or 'Directed by' in movie) and 'imdb_link' in movie]


# In[17]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie]
len(wiki_movies)


# In[19]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]


# In[20]:


x = 'global value'

def foo():
    x = 'local value'
    print(x)

foo()
print(x)


# In[21]:


my_list = [1,2,3]
def append_four(x):
    x.append(4)
append_four(my_list)
print(my_list)


# In[23]:


square = lambda x: x * x
square(5)


# In[24]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    return movie


# In[25]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]


# In[27]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    return movie


# In[28]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]


# In[29]:


wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


# In[30]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie


# In[31]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)


# In[32]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')


# In[33]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[34]:


[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[35]:


[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


# In[36]:


wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


# In[37]:


box_office = wiki_movies_df['Box office'].dropna() 


# In[38]:


len(box_office)


# In[39]:


def is_not_a_string(x):
    return type(x) != str


# In[40]:


box_office[box_office.map(is_not_a_string)]


# In[41]:


box_office[box_office.map(lambda x: type(x) != str)]


# In[42]:


some_list = ['One','Two','Three']
'Mississippi'.join(some_list)


# In[43]:


box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[44]:





# In[45]:


form_one = r'\$\d+\.?\d*\s*[mb]illion'


# In[46]:


box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[47]:


form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[48]:


matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)


# In[50]:


box_office[~matches_form_one & ~matches_form_two]


# In[51]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
form_two = r'\$\s*\d{1,3}(?:,\d{3})+'


# In[52]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+'


# In[53]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'


# In[54]:


box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[55]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'


# In[56]:


box_office.str.extract(f'({form_one}|{form_two})')


# In[57]:


def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[58]:


wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[59]:


wiki_movies_df.drop('Box office', axis=1, inplace=True)


# In[60]:


budget = wiki_movies_df['Budget'].dropna()


# In[61]:


budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[62]:


budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[63]:


matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget[~matches_form_one & ~matches_form_two]


# In[64]:


budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[65]:


wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[66]:


wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[67]:


release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[68]:


date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[69]:


release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)


# In[70]:


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[71]:


running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[72]:


running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()


# In[73]:


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


# In[74]:


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


# In[75]:


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]


# In[76]:


running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[77]:


running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[78]:


wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[79]:


wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[80]:


kaggle_metadata.dtypes


# In[81]:


kaggle_metadata['adult'].value_counts()


# In[82]:


kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]


# In[83]:


kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# In[84]:


kaggle_metadata['video'].value_counts()


# In[85]:


kaggle_metadata['video'] == 'True'


# In[86]:


kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[87]:


kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[88]:


kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[89]:


ratings.info(null_counts=True)


# In[90]:


pd.to_datetime(ratings['timestamp'], unit='s')


# In[91]:


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[92]:


ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[93]:


movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


# In[94]:


movies_df[['title_wiki','title_kaggle']]


# In[95]:


movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]


# In[96]:


movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[97]:


movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[98]:


movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[99]:


movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# In[100]:


movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# In[101]:


movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[102]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[103]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index


# In[104]:


movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[105]:


movies_df[movies_df['release_date_wiki'].isnull()]


# In[107]:


movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[108]:


movies_df['original_language'].value_counts(dropna=False)


# In[109]:


movies_df[['Production company(s)','production_companies']]


# In[110]:


movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[111]:


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[112]:


fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# In[113]:


for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)


# In[114]:


movies_df['video'].value_counts(dropna=False)


# In[115]:


movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


# In[116]:


movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


# In[117]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()


# In[118]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1) 


# In[119]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[120]:


rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[121]:


movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')


# In[122]:


movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[135]:





# In[136]:


db_string = f"postgres://postgres:{db_password}@localhost:XXXX/movie_data"


# In[137]:


engine = create_engine(db_string)


# In[138]:


movies_df.to_sql(name='movies', con=engine)


# In[141]:


rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')


# In[ ]:




