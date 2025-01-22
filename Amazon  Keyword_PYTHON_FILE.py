#!/usr/bin/env python
# coding: utf-8

# ## Steps
# 
# ### Scrape the bestseller, Top Rated, etc lists
# 1. Start with main books page, select each of the genre page to scrape. Open the 'Bestseller' and 'Top Rated' section.
# 2. Open each book page and scrape the title, author, and description in a table for that genre. Delete duplicates
# 3. Create SQL tables per genre and subgenre and insert all books into their tables.
# 4. Sort most popular keywords and count the number of times they appear per subgenre.

# In[ ]:


import scrapy
import requests

from scrapy.crawler import CrawlerProcess

import sys 
import csv


import time
import random

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras as extras
from psycopg2 import Error
from sqlalchemy import create_engine

import re
from urllib.parse import urlencode
from urllib.parse import urljoin


# In[ ]:


if "twisted.internet.reactor" in sys.modules:
    del sys.modules["twisted.internet.reactor"]

    
class KeywordSpider ( scrapy.Spider ):
    name = "amazon_keyword"
    asin = ''
    genre = ''
    subgenre = ''
    product_url = ''
    
    def start_requests( self ):
        

        HEADERS = ({'User-Agent':
            'a-keyword-project (https://amazon.com)'
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5',
            'Redirect_enabled': 'true'})
        
        url = 'https://www.amazon.com/books-used-books-textbooks/b/?ie=UTF8&node=283155'
        yield scrapy.Request(url = get_url(url), headers = HEADERS, callback = self.parse_books_page)

       
    def parse_books_page( self, response ):
        genres_start = response.xpath('//li[@class="a-spacing-micro apb-browse-refinements-indent-2"]')
        genre_links = genres_start.xpath('.//a/@href')
        links_to_follow = genre_links.extract()
        
        #Collect the genre titles
        genre_title = genres_start.xpath('.//span[@dir="auto"]//text()').extract()
        clean_genres_all = [ re.sub('\W+','_', element) for element in genre_title ]
        print(clean_genres_all)

                
        for i in range(len(links_to_follow)):
            links_to_follow[i] = "https://www.amazon.com" + links_to_follow[i]
        
        links_and_genres = list(zip(clean_genres_all, links_to_follow))
        
        global genre
        
        for genre, url in links_and_genres:
            print(genre)
            delay()
            yield response.follow(url = get_url(url), callback = self.parse_subgenres, meta={'genre': genre})

            
    def parse_subgenres( self, response ):
        global genre

        subgenres_start = response.xpath('//li[@class="a-spacing-micro apb-browse-refinements-indent-2"]')
        subgenre_links = subgenres_start.xpath('.//a/@href')
        links_to_follow = subgenre_links.extract()
        
        #Collect the genre titles
        subgenre_title = subgenres_start.xpath('.//span[@dir="auto"]//text()').extract()
        clean_subgenre = [ re.sub('\W+','_', element) for element in subgenre_title ]
        print(clean_subgenre)
        
                
        for i in range(len(links_to_follow)):
            links_to_follow[i] = "https://www.amazon.com" + links_to_follow[i]
        
        links_and_genres = list(zip(clean_subgenre, links_to_follow))
        global subgenre
        
        for subgenre, url in links_and_genres:
            print(subgenre)
            delay()
            yield response.follow(url = get_url(url), callback = self.parse_genres, meta={'genre': genre, 'subgenre' : subgenre})
            
        
    def parse_genres (self, response):
 
        genre = response.meta['genre']
        subgenre = response.meta['subgenre']
        
        top_rated_block = response.xpath('//div[@class="a-section octopus-pc-card-title"]')
        top_rated_links = top_rated_block.xpath('.//a/@href')
        links_to_follow = top_rated_links.extract()
        
        print("in parse_genres: ", genre)

        for i in range(len(links_to_follow)):
            links_to_follow[i] = "https://www.amazon.com" + links_to_follow[i]
       

        for url in links_to_follow:
            delay()
            yield response.follow(url = get_url(url), callback = self.parse_keyword_response, meta={'genre': genre, 'subgenre' : subgenre})
            
    
    def parse_keyword_response(self, response):
        
        global asin
        global product_url
        
        
        genre = response.meta['genre']
        subgenre = response.meta['subgenre']
        
        df_csv_file = pd.read_csv('amazon_keywords.csv')
        asin_column = df_csv_file['asin'] 
        my_set = set(asin_column)
        
        products = response.xpath('//*[@data-asin]')

        for product in products:
            asin = product.xpath('@data-asin').extract_first()
            print(asin)
            
            if asin in my_set: 
                pass
                print("PASSED")
            else:
                product_url = f"https://www.amazon.com/dp/{asin}"
                print(product_url)
                yield response.follow(url= get_url(product_url), callback=self.parse_all, meta={'asin': asin, 'genre': genre, 'subgenre' : subgenre, 'product_url': product_url})

        next_page = response.xpath('//li[@class="a-last"]/a/@href').extract_first()
        if next_page:
            print("Next Page")
            url = urljoin("https://www.amazon.com",next_page)
            yield scrapy.Request(url = get_url(product_url), callback = self.parse_keyword_response, meta={'genre': genre, 'subgenre' : subgenre})

            
    
    def parse_all (self, response):

        asin = response.meta['asin']
        genre = response.meta['genre']
        subgenre = response.meta['subgenre']
        product_url = response.meta['product_url']
        
        
        print("in parse_all: ", subgenre)
        # Create a SelectorList of the course titles text
        crs_title = response.xpath('//span[@id="productTitle"]//text()').get() 
        print(crs_title)

        # Create a SelectorList of course descriptions text
        crs_descr = response.xpath('//*[@id="bookDescription_feature_div"]//text()').extract()
                
        #Clean up repeat blank elements in list
        crs_descr = [i for a,i in enumerate(crs_descr) if i!=' ' ]
        crs_descr = [i for a,i in enumerate(crs_descr) if i!='  ' ]
        crs_descr = [i for a,i in enumerate(crs_descr) if i!='                                 ' ]
        crs_descr = [i for a,i in enumerate(crs_descr) if i!='\n                                     ' ]
        crs_descr = [i for a,i in enumerate(crs_descr) if i!='Read more' ]
        crs_descr = [i for a,i in enumerate(crs_descr) if i!='Read less' ]
        crs_descr = [i for a,i in enumerate(crs_descr) if i!='xa0' ]
        
        
        all_asin.append(asin)
        all_genres.append(genre)
        all_subgenres.append(subgenre)
        all_product_url.append(product_url)
        all_titles.append(crs_title)
        all_descr.append(crs_descr)
        


API = 'API nonsense: letters numbers and such'
def get_url(url):
    payload = {'api_key': API, 'url': url}
    proxy_url = 'http://api.scraperapi.com/?' + urlencode(payload)
    return proxy_url



def delay():
    time.sleep(random.randint(3, 10))

clean_genres_all = list()
all_asin = []
all_genres = []
all_subgenres = []
all_product_url = []
all_titles = []
all_descr = []


process = CrawlerProcess()
process.crawl(KeywordSpider)
process.start()

df = pd.DataFrame({
    'asin': all_asin,
    'genre': all_genres,
    'subgenre': all_subgenres,
    'product_url': all_product_url,
    'title': all_titles, 
    'description': all_descr
})

print("Finished")


# In[ ]:


import csv

df.to_csv('amazon_keywords.csv', mode='a', index=True, header=False)


# In[ ]:



df_csv_file = pd.read_csv('amazon_keywords.csv')


# In[ ]:


print(df)


# In[ ]:


# Import and download packages
import requests
from bs4 import BeautifulSoup
import nltk
from collections import Counter
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Create a tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# Tokenize the text
df_token = df['description'].astype(str)
token_two = [tokenizer.tokenize(i)for i in df_token]


# Create a list called words containing all tokens transformed to lowercase
words = [[word.lower() for word in token] for token in token_two]

# Get the English stop words from nltk
all_stopwords = nltk.corpus.stopwords.words('english')

# Add Amazon books specific stopwords
keyword_stop_list = ['bestseller', 'bestselling', '1', 'bestseller', 'author', 'amazon', 'national', 'international',
                     'new york times', 'audible', 'kindle', 'must', 'never', 'novel', 'book', 'also', 'brandon', 'every',
                     'would', 'get', 'even', 'new', 'york', 'times', 'us', 'kickstarter', 'well', 'xa0', 'f', 
                     't', 'ck', 'n', 'preorder', 'x', 'b', 'c', 'd', 'e', 'h', 'j', 'k', 'l', 'm', 'o', 'p', 'q', 
                     'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'st']
all_stopwords.extend(keyword_stop_list)

# Remove all stopwords
output_array=[]
for sentence in words:
    temp_list=[]
    for word in sentence:
        if word not in all_stopwords:
            temp_list.append(word)
    output_array.append(temp_list)
        
#print(output_array)

# Make a list of all the words in the genre descriptions
concat_list = [j for i in output_array for j in i]

# Make a concatinated list of all the keywords in each description
concat_list_group = [" ".join(i) for i in output_array]


# In[ ]:


# Example: 'postgresql://username:password@localhost:5432/your_database'
engine = create_engine('postgresql://username:password@localhost:5432/keyword_database')

start_time = time.time() # get start time before insert

df_csv_file.to_sql(
    name="all_genres_with_subgenres", # table name
    con=engine,  # engine
    if_exists="append", #  If the table already exists, append
    index=True # no index
)

end_time = time.time() # get end time after insert
total_time = end_time - start_time # calculate the time
print(f"Insert time: {total_time} seconds") # print time


# In[ ]:


conn = psycopg2.connect("dbname=keyword_database user=username password=password")

# Open a cursor to perform database operations
cur = conn.cursor()

def delete_duplicates():
    cur.execute("""DELETE FROM all_genres_with_subgenres
                    WHERE (asin) IN (
                        SELECT asin
                        FROM all_genres_with_subgenres
                        GROUP BY  asin
                        HAVING COUNT(*) > 1
                    );
                """)
    conn.commit()

def insert_keywords( ):
    
    row_counter = 0
    # Looping through each item in the list
    for descr_list in concat_list_group:
        list_counter = 0
        print(descr_list)
        count_indiv_descr = Counter(descr_list.split())
        top_five = count_indiv_descr.most_common(5)
        print(top_five)
        
        cur.execute("UPDATE all_genres_with_subgenres SET all_keywords = '{0}' WHERE index = %s".format(descr_list), [row_counter])
        conn.commit()
        for indiv in top_five:
            print(indiv)
            top_numerated = ['first', 'second', 'third', 'fourth', 'fifth']
            print(top_numerated[list_counter])
            cur.execute("UPDATE all_genres_with_subgenres SET " + top_numerated[list_counter] + "_top_keyword = {0} WHERE index = %s".format(indiv), [row_counter])
            list_counter += 1
            conn.commit()
        row_counter += 1
    conn.commit()


cur.execute("""ALTER TABLE all_genres_with_subgenres
                ADD COLUMN IF NOT EXISTS top_keywords_for_genre VARCHAR(255),
                ADD COLUMN IF NOT EXISTS all_keywords VARCHAR(10000),
                ADD COLUMN IF NOT EXISTS first_top_keyword VARCHAR(255),
                ADD COLUMN IF NOT EXISTS second_top_keyword VARCHAR(255),
                ADD COLUMN IF NOT EXISTS third_top_keyword VARCHAR(255),
                ADD COLUMN IF NOT EXISTS fourth_top_keyword VARCHAR(255),
                ADD COLUMN IF NOT EXISTS fifth_top_keyword VARCHAR(255)
                """)
conn.commit()



insert_keywords()
delete_duplicates()


cur.close()
conn.close()


# In[ ]:


clean_genres_all = ['Arts_Photography', 'Biographies_Memoirs', 'Business_Money', 'Calendars', 'Children_s_Books', 'Christian_Books_Bibles', 'Comics_Graphic_Novels', 'Computers_Technology', 'Cookbooks_Food_Wine', 'Crafts_Hobbies_Home', 'Education_Teaching', 'Engineering_Transportation', 'Health_Fitness_Dieting', 'History', 'Humor_Entertainment', 'Law', 'LGBTQ_Books', 'Literature_Fiction', 'Medical_Books', 'Mystery_Thriller_Suspense', 'Parenting_Relationships', 'Politics_Social_Sciences', 'Reference', 'Religion_Spirituality', 'Romance', 'Science_Fiction_Fantasy', 'Science_Math', 'Self_Help', 'Sports_Outdoors', 'Teen_Young_Adult', 'Test_Preparation', 'Travel']
print(clean_genres_all)


# In[ ]:


# Connect to your postgres DB
conn = psycopg2.connect("dbname=keyword_database user=username password=password")

# Open a cursor to perform database operations
cur = conn.cursor()

def create_genre_tables(genre):
    cur.execute("""CREATE TABLE IF NOT EXISTS """ + genre + """(
               id SERIAL PRIMARY KEY,
               genre VARCHAR(255),
               subgenre VARCHAR(255),
               title VARCHAR(255),
               description VARCHAR(10000),
               all_keywords VARCHAR(10000),
               first_top_keyword VARCHAR(255),
               second_top_keyword VARCHAR(255),
               third_top_keyword VARCHAR(255),
               fourth_top_keyword VARCHAR(255),
               fifth_top_keyword VARCHAR(255),
               asin VARCHAR(255),
               product_url VARCHAR(255)
               )
               """)
    
    conn.commit()
    
    
def create_subgenre_tables(genre, subgenre):
    cur.execute("""CREATE TABLE IF NOT EXISTS """ + subgenre + """(
               id SERIAL PRIMARY KEY,
               genre VARCHAR(255),
               subgenre VARCHAR(255),
               title VARCHAR(255),
               description VARCHAR(10000),
               all_keywords VARCHAR(10000),
               first_top_keyword VARCHAR(255),
               second_top_keyword VARCHAR(255),
               third_top_keyword VARCHAR(255),
               fourth_top_keyword VARCHAR(255),
               fifth_top_keyword VARCHAR(255),
               asin VARCHAR(255),
               product_url VARCHAR(255),
               FOREIGN KEY (subgenre)
                  REFERENCES """ + genre + """(subgenre)
                  ON DELETE CASCADE)
               """)
    
    conn.commit()
    
    
for x in clean_genres_all:
    print(x)
    create_genre_tables(x)
    for y in <something>:
        print(y)
        create_subgenre_tables(x, y)
    
cur.close()
conn.close()


# In[ ]:


conn = psycopg2.connect("dbname=keyword_database user=username password=password")

# Open a cursor to perform database operations
cur = conn.cursor()

    
def insert_genres (genre):
    cur.execute("""INSERT INTO """ + genre + """ ( genre, subgenre, asin, product_url, title, description, 
                        all_keywords, first_top_keyword, second_top_keyword, 
                        third_top_keyword, fourth_top_keyword, fifth_top_keyword)
                SELECT DISTINCT genre, subgenre, asin, product_url, title, description, all_keywords, 
                        first_top_keyword, second_top_keyword, 
                        third_top_keyword, fourth_top_keyword, fifth_top_keyword
                FROM all_genres_with_subgenres
                WHERE genre = '""" +  genre + """'
                """)
    print (genre)
    conn.commit()

    
def insert_subgenres (genre, subgenre):
    cur.execute("""INSERT INTO """ + subgenre + """ ( genre, subgenre, asin, product_url, title, description, 
                        all_keywords, first_top_keyword, second_top_keyword, 
                        third_top_keyword, fourth_top_keyword, fifth_top_keyword)
                SELECT DISTINCT genre, subgenre, asin, product_url, title, description, all_keywords, 
                        first_top_keyword, second_top_keyword, 
                        third_top_keyword, fourth_top_keyword, fifth_top_keyword
                FROM """ + genre + """
                WHERE subgenre = '""" +  subgenre + """'
                """)
    print (genre)
    conn.commit()
def find_top_keys (genre):
    
    query = "SELECT all_keywords FROM {0} ;".format(genre)

    cur.execute(query)
    
    all_keys = cur.fetchall()
    word_two = [x for word in all_keys for x in word]
    token_two = [tokenizer.tokenize(i) for i in word_two]

    output_array=[]
    for sentence in word_two:
        temp_list=[]
        for word in sentence:
            temp_list.append(word)
        output_array.append(temp_list)


    concat_key_list = [j for i in token_two for j in i]

    
    # Initialize a Counter object from our processed list of words
    count_all = Counter(concat_key_list)
    print(count_all)
    
    # Store ten most common words and their counts as top_ten
    top_ten = count_all.most_common(10)

    # Print the top ten words and their counts
    print("Added the top ten in", genre)
    print(top_ten)
    cur.execute("ALTER TABLE "+ genre +" ADD COLUMN IF NOT EXISTS top_keywords_for_genre VARCHAR(255)")

    all_words = []
    for common_words, *rest in top_ten:
        all_words.append(common_words)
        
    concat_keys = ', '.join(all_words)

    cur.execute("UPDATE " + genre + " SET top_keywords_for_genre = '{0}' ".format(concat_keys) )

    conn.commit()


for x in clean_genres_all:
    insert_genres(x)
    find_top_keys(x)
   
    for y in <something>:
        insert_subgenres(x,y)
        find_top_keys(y)

    
cur.close()
conn.close()


# In[ ]:




