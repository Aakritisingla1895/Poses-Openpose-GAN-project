"""
How to install flickr-api

https://github.com/ultralytics/flickr_scraper/issues/5

Warning: create a new environment before conducting these steps

!git clone https://github.com/ultralytics/flickr_scraper
%cd flickr_scraper
%pip install -qr requirements.txt

How to get a flickr API key
http://www.cmssupport.utoronto.ca/help/Creating_a_Flickr_API_key.htm
"""

"""
extras argument:
url_sq : s  small square 75x75
url_q  : q  large square 150x150
url_t  : t  thumbnail, 100 on longest side
url_s  : m  small, 240 on longest side
url_n  : n  small, 320 on longest side
url_m  : -  medium, 500 on longest side
url_z  : z  medium 640, 640 on longest side
url_c  : c  medium 800, 800 on longest side†
url_l  : b  large, 1024 on longest side*
url_o  : o  original image, either a jpg, gif or png, depending on source format
"""

import flickrapi
import urllib.request

api_key = '------'
secret = '------'

image_save_directory = 'images/dancer/img_'
image_save_count = 5000 # 5000
extras = "url_q"

print(flickrapi.__version__)

flickr = flickrapi.FlickrAPI(api_key,secret)
count = 0

for photo in flickr.walk(tag_mode='all',
                         text ='dancer, dance, contemporary, solo',
                         media='photos',
                         sort='relevance',
                         extras=extras):
    try:
        photo_url = photo.get(extras)
        filename = image_save_directory+('{:0>10}'.format(count))+'.jpg'
        urllib.request.urlretrieve(photo_url, filename)
        count += 1
        
        print("get photo index {} name {} ".format(count, filename) )
    except:
        pass
    
    if count > image_save_count:
        break
