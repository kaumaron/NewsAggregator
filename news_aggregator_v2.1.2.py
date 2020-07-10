# -*- coding: utf-8 -*-
# encoding='utf-8'
from pytz import timezone
import time
import feedparser
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from subprocess import call
import string
import BeautifulSoup
import re
import math
from collections import Counter
from operator import itemgetter
import operator
import numpy
import scipy
from hcluster import linkage, dendrogram
from datetime import datetime
import urlparse
import urllib2 as request
import httplib
from http.cookiejar import CookieJar
import smtplib
from email.mime.text import MIMEText

from sources_dict import sources_dict as sources
from newsfeeds import feeds
import secrets

#Time vars
start_time = time.time()
feed_read_time = datetime.now(timezone('US/Eastern'))\
                .strftime("%Y-%m-%d--%H:%M:%S")
webpage_feed_day = datetime.now(timezone('US/Eastern'))\
                    .strftime("%A, %d %b %Y")
webpage_feed_time = datetime.now(timezone('US/Eastern')).strftime("%H:%M EST")

#Email Params
email_from = secrets.gmail_user
email_to = secrets.sender # must be a list
email_bcc = []
with open('/home/ubuntu/email_list.txt', 'r') as emails:
    for line in emails:
        email_bcc.append(line.strip())

#File Names
file_path = 'News_Unspun/article_aggregator/'
log_path = 'cronlogs/'
file_name = file_path+feed_read_time+'.html'
log_name = log_path+str(start_time)+'.log'
template_path = 'Tech_News/html5up-fractal/template.html'
html_file_path = 'Tech_News/html5up-fractal/webpages/'
html_file = html_file_path+'archive/'+feed_read_time+'.html'
index_file = html_file_path+'index.html'

# RegEx
www = re.compile('(www\.)?')
end = re.compile('(\.com)?(\.co\.uk)?(\.org)?')

#NLTK Tools
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')
lemmer = WordNetLemmatizer()

def realurl(url): # python 2.7 version
    req = request.Request(url)
    cj = CookieJar()
    opener = request.build_opener(request.HTTPCookieProcessor(cj))
    response = opener.open(req)
    raw_response = response.read().decode('utf8', errors='ignore')
    response.close()
    real_url = response.geturl()
    real_url_split = urlparse.urlparse(real_url)._replace(query=None)
    return urlparse.urlunparse(real_url_split)

def clean_html(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()

def save_output():
    with open(file_name, "w") as text_file:
        text_file.write(todays_news.encode('utf-8')) # Encode error here too

    with open(log_name, 'w') as log_file:
        for i in log:
            log_file.write(' '.join(i).encode('utf-8'))
    print 'File written to {}'.format(file_name)
    print 'Log written to {}'.format(log_name)

def send_email():
    now = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    with open(file_name, 'rb') as fp:
        msg = MIMEText(fp.read())
    subject = 'News Report from The News Unspun -  {}'.format(feed_read_time)

    msg['Subject'] = subject
    msg['From'] = email_from
    msg['To'] = ', '.join(email_to)
    email_recipients = email_to# + email_bcc

    gmail_user = secrets.gmail_user
    gmail_pwd = secrets.gmail_pwd
    smtpserver = smtplib.SMTP_SSL("smtp.gmail.com",465)
    smtpserver.ehlo()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    smtpserver.sendmail(email_from, email_recipients, msg.as_string())
    smtpserver.quit()
    print 'Email sent!'

# Reinventing the wheel, eventually use skikit-learn to handle this.
def freq(word, document):
    return document.count(word)

def wordCount(document):
    return len(document)

def numDocsContaining(word,documentList):
  count = 0
  for document in documentList:
    if freq(word,document) > 0:
      count += 1
  return count

def tf(word, document):
    return (freq(word,document) / float(wordCount(document)))

def idf(word, documentList):
    return math.log(len(documentList) / numDocsContaining(word,documentList))

def tfidf(word, document, documentList):
    return (tf(word,document) * idf(word,documentList))

def top_keywords(n,doc,corpus):
    d = {}
    for word in set(doc):
        d[word] = tfidf(word,doc,corpus)
    sorted_d = sorted(d.iteritems(), key=operator.itemgetter(1))
    sorted_d.reverse()
    return [w[0] for w in sorted_d[:n]]

def extract_clusters(Z,threshold,n):
   clusters={}
   ct=n
   for row in Z:
      if row[2] < threshold:
          n1=int(row[0])
          n2=int(row[1])

          if n1 >= n:
             l1=clusters[n1]
             del(clusters[n1])
          else:
             l1= [n1]

          if n2 >= n:
             l2=clusters[n2]
             del(clusters[n2])
          else:
             l2= [n2]
          l1.extend(l2)
          clusters[ct] = l1
          ct += 1
      else:
          return clusters


# Define empty containers
log=[]
corpus = []
titles = []
links = []
things = []
errors = []
# Initialize counters
ct = -1
uni = 0

for feed in feeds:
    d = feedparser.parse(feed)
    for e in d['entries']:
        source = re.sub(end, '',
                        urlparse.urlparse(d['feed']['links'][0]['href'])[1])
        source = re.sub(www, '', source).split('.')[-1]
        if source in sources:
            source = sources[source]
        try:
            words = nltk.wordpunct_tokenize(clean_html(e['description']))
            words.extend(nltk.wordpunct_tokenize(e['title']))
            words = [lemmer.lemmatize(word) for word in words if word\
                    not in stopwords]
            lowerwords=[x.lower() for x in words if len(x) > 1]
            ct += 1
            #print(ct, "TITLE",e['title'],urlparse(e['link'])[2])
            idfer = re.split(r'/', urlparse.urlparse(e['link'])[2])[-2:]
            try:
                if idfer[1]=='index.html' or idfer[1] == '':
                    idfer = idfer[0]
                else:
                    idfer =idfer[1]
            except IndexError:
                idfer = idfer[0]
            #print source, idfer # Helpful for troubleshooting
            if idfer.lower() not in links and (e['title'].lower(),source)\
                not in things:
                uni += 1
                links.append(e['link'].lower())
                things.append((e['title'].lower(),source))
                corpus.append(lowerwords)
                titles.append((e['title'],e['link'],source))
        except KeyError:
            errors.append(e)
    articles_read = ct + 1
    duplicates = ct - uni + 1

print '{} articles read, {} duplicates and {} errors'.format(
                                                    articles_read, 
                                                    duplicates, 
                                                    len(errors))
print 'Articles Read. \n Creating Keywords...\n Time to Task: {}'.format(
                                                    time.time() - start_time)
nerrors = len(errors)
#Determine Keywords for Articles
key_word_list=set()
nkeywords=4
[[key_word_list.add(x) for x in top_keywords(nkeywords,doc,corpus)]\
        for doc in corpus]

ct=-1
for doc in corpus:
    ct+=1
    log.append([str(ct),"KEYWORDS"," ",' '.join(
                top_keywords(nkeywords,doc,corpus)), '\n'])
print 'Keywords determined.\n Running Analysis...\n Time to Task: {}'\
        .format(time.time() - start_time)

#Vectorize and Find Distances
feature_vectors=[]
n=len(corpus)

for document in corpus:
    vec=[]
    [vec.append(tfidf(word, document, corpus) if word in document else 0)\
        for word in key_word_list]
    feature_vectors.append(vec)
print 'Vectorized. Time to Task: {}'.format(time.time() - start_time)

mat = numpy.empty((n, n))
for i in xrange(0,n):
    for j in xrange(0,n):
        mat[i][j] = scipy.spatial.distance.cosine(
                            feature_vectors[i],
                            feature_vectors[j])
print 'Cosine distance determined. Time to Task: {}'\
            .format(time.time() - start_time)

#Threshold and Linkage
t = 0.8
Z = linkage(mat, 'single')
print 'Linkage analyzed. Time to Task: {}'.format(time.time() - start_time)

with open(template_path) as file:
    template = file.read()

clusters = extract_clusters(Z,t,n)

print 'Clustered.\n Time to Task: {}\n\nOutputting data.'\
                            .format(time.time() - start_time)
todays_news = 'To Unsubscribe reply to this email.\n{} articles read, {} \
    duplicates and {} errors\n'.format(articles_read, duplicates, len(errors))
html_news = re.sub('{date-day}', webpage_feed_day, 
                                template.split(r'<!-- One')[0])
html_news = re.sub('{date-time}', webpage_feed_time, html_news)
html_news = re.sub('{number-of-articles}', 
                str(articles_read - duplicates - nerrors),
                html_news)
html_news += '''
			<section id="two" class="wrapper">
				<div class="inner alt">
'''

checksLogger = []

rotating_divs = [template.split(r'<!--First-->')[1],
                template.split(r'<!--Second-->')[1],
                template.split(r'<!--Third-->')[1]]
div_ct = 0

for key in clusters:
    todays_news += "\n"
#    html_news += "\n"

    key_clusters = []
    for id in clusters[key]:
        key_clusters.extend(
        [word for word in nltk.wordpunct_tokenize(titles[id][0]) if \
            (word not in stopwords) & (len(word) > 1)])
    key_clusters = ' <strong>|</strong> '.join([k for k,v in Counter(
                        [k for k,v in nltk.pos_tag(key_clusters) \
                        if v[0] in ['N','V','R','J']]
                        ).most_common(4)])

    html_news += re.sub('{keywords}', key_clusters, rotating_divs[div_ct % 3])
    html_news += '\n<p>\n'
    div_ct += 1

    for id in clusters[key]:
        try:
            titles[id] = (titles[id][0], realurl(titles[id][1]),titles[id][2])
        except request.HTTPError, e:
            checksLogger.append('For url '+ titles[id][1] + ' HTTPError = ' + str(e.code))
            titles[id] = (titles[id][0], titles[id][1],titles[id][2])
        except request.URLError, e:
            checksLogger.append('For url '+ titles[id][1] + ' URLError = ' + str(e.reason))
            titles[id] = (titles[id][0], titles[id][1],titles[id][2])
        except httplib.HTTPException, e:
            checksLogger.append('For url '+ titles[id][1] + ' HTTPException')
            titles[id] = (titles[id][0], titles[id][1],titles[id][2])
        except Exception:
            import traceback
            checksLogger.append('For url '+ titles[id][1] + ' generic exception: ' + traceback.format_exc())
            titles[id] = (titles[id][0], titles[id][1],titles[id][2])
        # Joining like this prevents an encoding error
        _article = ':::'.join([str(id),titles[id][2],titles[id][0],titles[id][1]]).split(':::')

        _title = _article[2]
        _url = _article[-1]
        _source = _article[1]

        todays_news += '- '.join([str(id),titles[id][2],titles[id][0],titles[id][1]])
        todays_news += '\n'

        article_builder = '<a href={article_url} target="_blank">[{source}] - {article_title}</a><br/>'
        article_builder = re.sub('{article_url}', _url, article_builder)
        article_builder = re.sub('{article_title}', _title, article_builder)
        article_builder = re.sub('{source}', _source, article_builder)
        html_news += article_builder

    html_news += '\n</p>\n'
    html_news += '''
        						</div>
        					</section>'''
    if div_ct % 5 == 0:
        html_news += '''\n<section class="" style="text-align:center">

                                <ul class="icons">
                                        <li>
                                                <a href="https://www.facebook.com/sharer.php?u=http://newsunpun.com" 
						id="shareBtn" class="icon fa-facebook" target="_blank">
                                                        <span class="label">Facebook</span></a>
                                        </li>
                                        <li>
                                                <a href="https://twitter.com/intent/tweet?
						text=Check%20out%20today%27s%20news%20from%20quality%20sources%3A%20https%3A%2F%2Fnewsunpun.ml%20via%20%40thenewsunspun" 
						class="icon fa-twitter" target="_blank">
                                                        <span class="label">Twitter</span></a>
                                        </li>
                               </ul>

</section>'''

html_news += '''
				</div>
			</section>
'''
html_news += template.split('<!-- Three -->')[-1]
save_output()
with open(html_file, 'w') as file:
    file.write(html_news.encode('utf-8'))
with open(index_file, 'w') as file:
    file.write(html_news.encode('utf-8'))
print 'Preparing Email.'

#send_email()

call(["aws", "s3", "cp", "Tech_News/html5up-fractal/webpages/index.html", "s3://newsunspun.com/", "--acl", "public-read", "--debug"])

call(["aws", "s3", "cp", "Tech_News/html5up-fractal/webpages/index.html", "s3://thenewsunspun/html5up-fractal/", "--acl", "public-read", "--debug"])

end_time = time.time()
print 'Completed in {} seconds'.format(end_time - start_time)
with open(file_path + 'runtimes.log', 'a') as file:
    file.write( '{}'.format( (end_time - start_time) / 60 ))

try:
    print 'URL related errors:\n' + '\n'.join(checksLogger)
except Exception:
    print 'URL error:: \n' + '\n'.join(checksLogger.encode('utf-8'))
