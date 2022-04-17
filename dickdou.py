import re
from urllib.parse import urlparse
import tld
import pandas
import whois
import requests
from bs4 import BeautifulSoup
import validators

def ifEmail(site): # see if the site send email to his personal emailbox
    try:
        x=requests.get(url=site)
        index1= x.content.find('mail()')
        index2=x.content.find('mailto')
        if index1 == -1 & index2 == -1:
            return True
        else:
            return False
    except:
        return False
      
def is_valid_domain(site_name):# this is for BasedRank
    if validators.domain(site_name):
        return True
    else:
        return False
      
def BasedRank(url):# see traffic by Alexa Rank
    if not is_valid_domain(url):
        return False
    alexaurl = 'https://alexa.com/siteinfo/'
    a=urlparse(url)
    site=a.netloc

    rank = alexaurl + site

    # Request formatted url for rank(s)
    page = requests.get(rank)
    soup = BeautifulSoup(page.content, 'html.parser')

    #country_ranks = soup.find_all('div', id='CountryRank')
    global_rank = soup.select('.rank-global .data')
    try:
        match = re.search(r'[\d,]+', global_rank[0].text.strip())
        if match.group()<100000:
            return True
        else:
            return False
    except:
        return False
      
def DNSrecord(url):# see if the DNS is recorded by WHOIS
    global flags
    flags = 0
    flags = flags | whois.NICClient.WHOIS_QUICK 
    try:
        w = whois.whois(url,flags=flags)
    except:# The DNS is not contained in the database
        return False
    try:
        name = w['domain_name']
    except:
        name = w['name']
    if name == None:
        return False
    else:
        return True
      
def ifSubdomain(url):#see if the site has subdomain, False represents Phishing
    a=urlparse(url)
    netlocation=a.netloc
    a=netlocation.replace('.','1')#remove dot from main domain
    ind=1
    while ind>=0:
        ind=a.find('0x')
        a=a[0:ind]+a[ind+4:]#see if IPv6
    if a.isdecimal():#if url is represented by IP
        count=0
        for cha in url:
            if cha == '.':
                count+=1
        if count==3:
            return True
        else:
            return False
    else:#if url is represented by string
        #get ccTLD
        try:
            obj = tld.get_tld(url, as_object=True)
        except:
            return False
        tldlist=str(obj)
        #remove www.
        if url[4]=='s':
            if url[8:11]=='www.':
                url=url[0:7]+url[12:]
        else:
            if url[7:10]=='www.':
                url=url[0:6]+url[11:-1]
        count = 0
        for cha in url:# count subdomain
            if cha == '.':
                count+=1
        for cha in tldlist:# remove ccTLDs
            if cha == '.':
                count-=1
        if count==1:
            return True
        else:
            return False
          
def ifSymbolAT(url):# normally a legitmate site url doesn't contain @ symbol
    index = url.find('@')
    if index == -1:
        return True
    return False
