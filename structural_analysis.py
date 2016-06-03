from bs4 import BeautifulSoup

class struct_obj:
	def __init__(self, num_scripts_present, scripts_types, num_iframes_present, num_embed_objs, num_forms_present, titlePresent, stylePresent, iframesPresent, textInputPresent):
		self.num_scripts_present = num_scripts_present
		self.scripts_types = scripts_types
		self.num_iframes_present = num_iframes_present
		self.num_embed_objs = num_embed_objs
		self.num_forms_present = num_forms_present
		self.titlePresent = titlePresent
		self.stylePresent = stylePresent
		self.iframesPresent = iframesPresent
		self.textInputPresent = textInputPresent

def struct_analyze(soup):
	#Number of scripts
	scripts_present = soup.find_all('script')
 	num_scripts_present = len( scripts_present )
	scripts_types = []
	for item in scripts_present:
		if(item.has_attr('type')):
			scripts_types.append(item['type'])
	
	#Number of iFrames
	num_iframes_present = len(soup.find_all('iframe'))
	
	#Number of Embedded Objects
	num_embed_objs = len(soup.find_all('audio')) + len(soup.find_all('video')) + len(soup.find_all('source')) + len(soup.find_all('embed')) + len(soup.find_all('object'))

	#Number of Forms
	num_forms_present = len(soup.find_all('form'))
	
	#Title Present
	titlePresent = (len(soup.find_all('title')) != 0)

	#Style Sheet Present
	stylePresent = (len(soup.find_all('style')) != 0)

	#iFrames Present
	iframesPresent = (len(soup.find_all('iframe')) != 0)
	
	#TextInput Present
	textInputPresent = False
	input_data = soup.find_all('input')
	for item in input_data:
		if(item.has_attr('type')):
			if(item['type'] == 'text'):
				textInputPresent = True
	
	return struct_obj(num_scripts_present, scripts_types, num_iframes_present, num_embed_objs, num_forms_present, titlePresent, stylePresent, iframesPresent, textInputPresent)
