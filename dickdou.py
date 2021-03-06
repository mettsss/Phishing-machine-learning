import re
from urllib.parse import urlparse
import tld
import pandas
import whois
import requests
from bs4 import BeautifulSoup
import validators

def ifEmail(site):
    try:
        x=requests.get(site,timeout = 5)
    except:
        return False
    te=str(x.content)
    index= te.find('mailto:')
    if index == -1:
        index1= te.find('mail()')
        if index == -1:
            return True
        else:
            return False
    else:
        return False
    '''start=0
    while True:
        index= te.find('mail'.encode(),start)
        if index==-1:
            return True
        else:
            index1=te.find('to:'.encode(),index+4,index+6)
            index2=te.find('()'.encode(),index+4,index+5)
            if  not index1 == -1:
                return False
            elif not index2 == -1:
                return False
            else:
                start = index + 1
                continue
    return False'''#优化方法
      
def is_valid_domain(site_name):
    if validators.domain(site_name):
        return True
    else:
        return False
      
def BasedRank(url):
    alexaurl = 'https://alexa.com/siteinfo/'
    a=urlparse(url)
    site=a.netloc
    if not is_valid_domain(site):
        return False
    rank = alexaurl + site

    # Request formatted url for rank(s)
    page = requests.get(rank)
    soup = BeautifulSoup(page.content, 'html.parser')

    #country_ranks = soup.find_all('div', id='CountryRank')
    global_rank = soup.select('.rank-global .data')
    if global_rank:
        match = re.search(r'[\d,]+', global_rank[0].text.strip())
        grank='0'
        for cha in match.group():
            if cha.isdecimal():
                grank = grank + cha
        if int(grank)<100000:
            return True
        else:
            return False
    else:
        return False
      
def DNSrecord(url):
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
      
def ifSubdomain(url):#False represents Phishing
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
