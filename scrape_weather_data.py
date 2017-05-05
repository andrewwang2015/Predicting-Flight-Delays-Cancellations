"""
Example script that scrapes data from the IEM ASOS download service
"""
import json
import datetime
from urllib.request import urlopen
import pandas as pd
# timestamps in UTC to request data for
startts = datetime.datetime(2015, 1, 1)
endts = datetime.datetime(2016, 1, 2)

SERVICE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
SERVICE += "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

SERVICE += startts.strftime('year1=%Y&month1=%m&day1=%d&')
SERVICE += endts.strftime('year2=%Y&month2=%m&day2=%d&')

states = """AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME
 MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
 WA WI WV WY PR_ GU_ AS_ VI_"""

airports = pd.read_csv('./airports.csv')
airport_codes = airports['IATA_CODE'].tolist()
# IEM quirk to have Iowa AWOS sites in its own labeled network
networks = []#['AWOS']
for state in states.split():
    networks.append("%s_ASOS" % (state,))

for network in networks:
    # Get metadata
    uri = ("https://mesonet.agron.iastate.edu/"
           "geojson/network/%s.geojson") % (network,)
    init_data = urlopen(uri).read()
    data = init_data.decode('utf-8')
    jdict = json.loads(data)
    for site in jdict['features']:
        faaid = site['properties']['sid']
        if faaid in airport_codes or faaid in ['PHTO', 'PHKO', 'PHLI', 'PHNL', 'PHOG', 'TJPS', 'TJBQ', 'TJSJ', 'PGUM', 'NSTU', 'TIST', 'TISX', \
            'PADK', 'PANC', 'PABR', 'PABE', 'PACV', 'PASC', 'PADL', 'PAFA', 'PAGS', 'PAJN', 'PAKT', 'PAKN', 'PADQ', \
            'PAOT', 'PAOM', 'PAPG', 'PASI', 'PAYA', 'PAWG', 'CRQ', 'SGJ', 'SAW', 'UNV', 'NYL', 'GPI']:
            '''if faaid == 'NYL':'''
            sitename = site['properties']['sname']
            uri = '%s&station=%s' % (SERVICE, faaid)
            print ('Network: %s Downloading: %s [%s]' % (network, sitename, faaid))
            init_data = urlopen(uri).read()
            data = init_data.decode('utf-8')
            outfn = '%s_%s_%s.csv' % (faaid, startts.strftime("%Y%m%d%H%M"),
                                      endts.strftime("%Y%m%d%H%M"))
            out = open(outfn, 'w')
            out.write(data)
            out.close()
