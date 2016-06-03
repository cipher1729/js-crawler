# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Field

class DataCrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = Field()
    url   = Field()
    domain_name = Field()
    timestamp = Field()
    all_list_data = Field()
    ranking = Field()
    page_text = Field()
    num_scripts_present = Field()
    scripts_types = Field()
    num_iframes_present = Field()
    num_embed_objs = Field()
    num_forms_present = Field()
    titlePresent = Field()
    stylePresent = Field()
    iframesPresent = Field()
    textInputPresent = Field()
    url_len= Field()
    domain_len= Field()
    scriptObjs= Field()
    pass
    
