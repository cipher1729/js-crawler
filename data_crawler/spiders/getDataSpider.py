import scrapy
from urlparse import urlparse
from bs4 import BeautifulSoup
from data_crawler.items import DataCrawlerItem 
import urllib2
import datetime
import hashlib
import re
import structural_analysis
import getAllJS
import os
import subprocess
import urllib2
import URLinfo
import writeToFile
import thread
from socket import timeout
import timeit
import logging
from time import sleep
import json
import sys

#reload(sys)
#sys.setdefaultencoding('utf-8')


class getDataSpider(scrapy.Spider):
    name = "getData"
    #handle = open('check-250.txt')
    #start_urls =handle.read()
    start_urls = ["http://www.wired.com", "http://www.amazon.com", "http://paym.buylownowchicago.com/oqJybxcMDE/CnhmoZjFis/UhHxKt-XmLsSxxFBR/", "http://bbc.mytoxictestscore.com/dc/R/63/", "http://www.ronaldo7.com"]
    #print start_urls 
    url_map = {}
    website_counter = 0
    p = 0
    hashedScripts=[];
    allHashedScripts=[];
    landingUrls=[]
    startTime=0
   
     
    def start_requests(self):
	#self.startTime = timeit.default_timer()
	c=0	
	for url in self.start_urls:
	#logging.debug("Pre request for  " +  url)
		c+=1
		#if c%10==0:
		#sleep(0.5)
		print "do request for " + url

		#if (c%10==0 ):
		#	sleep(10)
		yield scrapy.Request(url, self.parse, meta={
				'splash': {
					'endpoint': 'render.html',
					'args': {'wait': 0.2, "images": 0,"timeout":60 ,"resource_timeout": 10}
					}
					})	
   		

    def parse(self, response):
	print
	print	
	#elapsed = timeit.default_timer()- self.startTime
	#print "elapsed time is ............................."  + str(elapsed)
	#print response.url	
	#print "...............................................................................Website: "+ response.url+ " with count "+ str(self.website_counter)
	#print "Returning from splash parse!"
	actUrl = self.start_urls[self.website_counter];
	#filename = actUrl.split("/")[-2] + '.html'
	url_total = actUrl
	url = url_total.split("/")[2]
	parsed_uri = urlparse(actUrl)
    	domain_name = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
	#Convert JSON to DICT
	soup = BeautifulSoup(response.body, 'html.parser')
	
	try:
		title = soup.title.string
	except:
		title = "n/a"	
	items = DataCrawlerItem();

	urls_this_page =  soup.find_all('a')

	try:
		style_sheet = hashlib.sha512(str( soup.style.string )).hexdigest()
	except UnicodeEncodeError, e:
		style_str = soup.style.string
		style_str = style_str.encode("utf-8")
		style_sheet = hashlib.sha512( style_str ).hexdigest()
	except AttributeError, e:
		style_sheet = "0" 
		#print "NO STYLESHEET PRESENT " + str(soup.style)
	

	images_this_page = soup.find_all('img')
	image_sources = []
	
	for item in images_this_page:
		if ( not item.has_attr( 'src' ) ):
			continue
		image_sources.append(item['src'])
	#SEND ALL ALL SOURCES TO HARSHA's SCRIPT	

	#urlInfoDict = URLinfo.URLinfo(self.start_urls[self.website_counter],domain_name)
	#for key in urlInfoDict.keys():
	#	print "key: " + key + " " + str(urlInfoDict[key])

	#iframes_this_page =  soup.find_all('iframe')
	#iframe_sources = []
	"""
	for item in iframes_this_page:
		if( not item.has_attr('src') ):
			continue
		iframe_sources.append(item['src'])
	"""
	scripts_this_page = soup.find_all('script')

	#print "..........................................................................POST REQUESTS.........................................."
	#Get external and internal scripts using chandan's script
	#getAllJS.initPool()
	allScripts = getAllJS.getAllJs(soup, domain_name,None)
	
	#print len(scriptObj.internal_scripts)
	#print len(scriptObj.external_scripts)

	#print "hashed scripts is " + str(len(self.hashedScripts));
	
	#internal_scripts_src = scriptObj.internal_scripts
	internal_scripts_src=[]
	internal_scripts = []
	#external_scripts_src = scriptObj.external_scripts
	external_scripts = []
	external_scripts_src = []

	for script in allScripts:
		if(script.isInternal==True):		
			internal_scripts_src.append(script.text);
		else:
			external_scripts_src.append(script.text);
	print "number of internal scripts " + str(len(internal_scripts_src));
	print "number of external scripts " + str(len(external_scripts_src));		
	
	#write the js to file
	if( not os.path.exists('/home/group9p1/javascripts/' + self.start_urls[self.website_counter])):
		os.makedirs('/home/group9p1/javascripts/' + self.start_urls[self.website_counter])
	
	thread.start_new_thread(writeToFile.writeToFile,(self.start_urls,self.website_counter, internal_scripts_src, external_scripts_src))	
	#writeToFile.writeToFile(self.start_urls,self.website_counter, internal_scripts_src, external_scripts_src)
	
	#MORE SRC TAGS
	#print "Find more media sources"
	media_sources = []
	audio_sources = []
	video_sources = []


	videos_this_page = soup.find_all('video')
	for item in videos_this_page:
		if( not item.has_attr("src") ):
			continue
		video_sources.append(item['src'])

	audios_this_page = soup.find_all('audio')
	for item in audios_this_page:
		if( not item.has_attr("src") ):
			continue
		audio_sources.append(item['src'])

	embeds_this_page = soup.find_all('embed')
	for item in embeds_this_page:
		if( not item.has_attr("src") ):
			continue
		media_sources.append(item['src'])

	objects_this_page = soup.find_all('object')
	for item in objects_this_page:
		if( not item.has_attr("src") ):
			continue
		media_sources.append(item['src'])

	mediasources_this_page = soup.find_all('source')
	for item in mediasources_this_page:
		if( not item.has_attr("src") ):
			continue
		media_sources.append(item['src'])
	#print "find text"
	texts = soup.findAll(text=True)
	visible_texts_list = filter(visible, texts)
	visible_text = "\n".join(visible_texts_list)
	all_text_list = visible_text.split()
	for item in all_text_list:
		item = item.encode('utf-8')
	#print "structural analysis"
	#add structural analysis
        structural_data = structural_analysis.struct_analyze(soup)	
	
	print "Adding to database"
	#Copy everything to database
	items['title'] = title;
	items['url']= url; 
	items['domain_name'] = domain_name;
	items['timestamp'] = str(datetime.datetime.now())
	items['all_list_data'] = {'internal_scripts' : [], 'external_scripts' : [], 'audio_sources' : [], 'video_sources' : [], 'media_sources' : [], 'image_sources' : []}

	(items['all_list_data'])['internal_scripts'] = internal_scripts_src	
	(items['all_list_data'])['external_scripts'] = external_scripts_src	
	(items['all_list_data'])['media_sources'] = media_sources
	(items['all_list_data'])['video_sources'] = video_sources
	(items['all_list_data'])['audio_sources'] = audio_sources
	(items['all_list_data'])['image_sources'] = image_sources
	
	
	items["scriptObjs"] =[]
	#print "length of all scrippts in this page" + str(len(allScripts))
	#for script in allScripts:
		#print "script has parent"  + script.parent
		#print "script has hash"  + str(script.textHash)
	
	dictList=[]
        for item in allScripts:
		try:	
			item.text=""
			s = json.dumps(item.__dict__, encoding= 'utf-8', ensure_ascii= False)
			dictList.append(s)
		except:
			print "JSON dump ERROR"
			continue
	items["scriptObjs"] = dictList
	
	#for item in scriptObj.allScripts:
		#print "Items parent is " + item["parent"]
		#if item["sourceUrl"] != "None":
		#	print "items url is " + item["sourceUrl"]
		#print item["isIframeHeightZero"]

	#(items['all_list_data'])['iframe_sources'] = iframe_sources					

	items['page_text'] = all_text_list					
	items['num_scripts_present'] = structural_data.num_scripts_present
	items['scripts_types'] = structural_data.scripts_types
	items['num_iframes_present'] = structural_data.num_iframes_present
	items['num_embed_objs'] = structural_data.num_embed_objs
	items['num_forms_present'] = structural_data.num_forms_present
	items['titlePresent'] = structural_data.titlePresent
	items['stylePresent'] = structural_data.stylePresent
	items['iframesPresent'] = structural_data.iframesPresent
	items['textInputPresent'] = structural_data.textInputPresent
	items['ranking'] = self.website_counter
	#from harshas script
	items['url_len']= len(self.start_urls[self.website_counter])
	items['domain_len']= len(domain_name)
	
	#if(len(urlInfoDict['ip_list'])!=0):	
	#	items['ip_list']= urlInfoDict['ip_list']
	#if(len(urlInfoDict['loc_list'])!=0):	
	#	items['loc_list']= urlInfoDict['loc_list']
		
	#items['ssl']= urlInfoDict['ssl']
	#items['valid_cert']= urlInfoDict['valid_cert']

	print "sent to DB!"
	self.website_counter += 1 # Keep increasing ranking

	yield items;
	

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    #elif re.match('<!--.*-->', str(element)):
    #    return False
    return True
