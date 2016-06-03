
import subprocess
from geoip import geolite2
import requests
 
def check_cert_validity(url):
    try:
        req = requests.get(url, verify=True)
        return True
    except requests.exceptions.SSLError:
        return False     

""" domain shouldn't start with http"""	
		
def ipInfo(domain):
    res = subprocess.check_output("dig %s"%(domain),shell = True)
    res = res.split("\n")  
    res = filter(None,res)
    ipDict = {}
    try:
        i = res.index(";; ANSWER SECTION:") 
    except ValueError:
        ipDict['ip_list'] = 'NA'
        ipDict['loc_list'] = 'NA'
        return ipDict
    else:
        pass 
    ip_list = [] 
    loc_list = []
    i = i +1 
    while(res[i]!= ';; AUTHORITY SECTION:'):
	ip = filter(None,res[i].split('\t')) [4] 
	ip_list.append(ip) 
	i = i+1 
   
    for ip in ip_list:
       match = geolite2.lookup(ip)
       if(match): 
	   if(match.country in loc_list):
	       pass
	   else:
	       loc_list.append(match.country)
       else:
	  loc_list.append('NA')
     
    ipDict['ip_list'] = ip_list
    ipDict['loc_list'] = loc_list
    return ipDict
    
def URLinfo(url,domain):
    #print "In URLinfo" 
    ssl = False
    if(url.split(':')[0] == 'https'):
        ssl = True
    url_len = len( list(url.split('//')[1]) ) 
    #print url.split('//')[1] 
    domain_len = len(list(domain))
    #print "going to check certificate"
    #valid_cert = check_cert_validity(url)
    #print " checked certificte"
    UrlDict = {} 
    ipDict= {}
    ipDict = ipInfo(domain)
    UrlDict['url_len'] = url_len
    UrlDict['domain_len'] = domain_len
    UrlDict['ip_list'] =  ipDict['ip_list'] 
    UrlDict['loc_list'] = ipDict['loc_list']
    UrlDict['ssl'] = ssl
    UrlDict['valid_cert'] = None
    return UrlDict
