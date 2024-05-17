# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from random import random
from django.core.files.storage import FileSystemStorage
import random
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from datetime import datetime
import requests
import cv2
import numpy as np
from PIL import Image

import base64

from Testmodel import test_image



@csrf_exempt
def Login_check(request):
	uname=request.POST.get("uname")
	pswrd=request.POST.get("pswrd")
	print("username and password-->",uname,pswrd)
	try:
		if uname=="user@user.com" and pswrd=="user@123":

			data={"msg":"yes"}
		else:
			data={"msg":"invalid"}
		return JsonResponse(data,safe=False)
	except Exception as e:
		print(e)
		data={"msg":"no"}
		return JsonResponse(data,safe=False)


usedict={
'Alpinia Galanga (Rasna)':'Rasna plant is used in many Ayurvedic medicines in India, Tibet, Africa to help with inflammation, bronchitis, asthma, cough, indigestion, piles, joint pains,obesity, diabetes. The paste of the leaf is also applied externally to reduce swelling. There are many varieties of rasna available throughout India',
'Amaranthus Viridis (Arive-Dantu)':'Amaranthus viridis is used as traditional medicine in the treatment of fever, pain, asthma, diabetes, dysentery, urinary disorders, liver disorders, eye disorders and venereal diseases. The plant also possesses anti-microbial properties',
'Artocarpus Heterophyllus (Jackfruit)':'Jackfruit (Artocarpus heterophyllus Lam) is a rich source of several high-value compounds with potential beneficial physiological activities. It is well known for its antibacterial, antifungal, antidiabetic, anti-inflammatory, and antioxidant activities',
'Azadirachta Indica (Neem)':'Neem is a rich source of limonoids that are endowed with potent medicinal properties predominantly antioxidant, anti-inflammatory, and anticancer activities',
'Basella Alba (Basale)': 'Basella alba is reported to improve testosterone levels in males, thus boosting libido. Decoction of the leaves is recommended as a safe laxative in pregnant women and children. Externally, the mucilaginous leaf is crushed and applied in urticaria, burns and scalds.',
'Brassica Juncea (Indian Mustard)':'It is a folk remedy for arthritis, foot ache, lumbago and rheumatism. Brassica juncea is grown mainly for its seed used in the fabrication of brown mustard or for the extraction of vegetable oil. Brown mustard oil is used against skin eruptions and ulcers',
'Carissa Carandas (Karanda)':'Its fruit is used in the ancient Indian herbal system of medicine, Ayurvedic, to treat acidity, indigestion, fresh and infected wounds, skin diseases, urinary disorders and diabetic ulcer, as well as biliousness, stomach pain, constipation, anemia, skin conditions, anorexia and insanity',
'Citrus Limon (Lemon)':'Aside from being rich in vitamin C, which assists in warding off infections, the juice is traditionally used to treat scurvy, sore throats, fevers, rheumatism, high blood pressure, and chest pain',
'Ficus Auriculata (Roxburgh fig)':'The fruits are edible and are used to make jams, juices and curries in India. In Vietnam, unripe fruits are also used in salads. Leaves are used as fodder for ruminants',
'Ficus Religiosa (Peepal Tree)':'Ficus religiosa, commonly known as pepal belonging to the family Moraceae, is used traditionally as antiulcer, antibacterial, antidiabetic, in the treatment of gonorrhea and skin diseases',
'Hibiscus Rosa-sinensis':'Hibiscus rosa-sinensis is a flowering plant native to tropical Asia. Hibiscus is commonly consumed in teas made from its flowers, leaves, and roots. In addition to casual consumption, Hibiscus is also used as an herbal medicine to treat hypertension, cholesterol production, and cancer progression',
'Jasminum (Jasmine)':'Jasmine is inhaled to improve mood, reduce stress, and reduce food cravings. In foods, jasmine is used to flavor beverages, frozen dairy desserts, candy, baked goods, gelatins, and puddings. In manufacturing, jasmine is used to add fragrance to creams, lotions, and perfumes',
'Mangifera Indica (Mango)':'Various parts of plant are used as a dentrifrice, antiseptic, astringent, diaphoretic, stomachic, vermifuge, tonic, laxative and diuretic and to treat diarrhea, dysentery, anaemia, asthma, bronchitis, cough, hypertension, insomnia, rheumatism, toothache, leucorrhoea, haemorrhage and piles',
'Mentha (Mint)':'Mentha species are widely used in savory dishes, food, beverages, and confectionary products. Phytochemicals derived from mint also showed anticancer activity against different types of human cancers such as cervix, lung, breast and many others',
'Moringa Oleifera (Drumstick)':'Moringa supplies a good source of vitamin C, an antioxidant nutrient that supports immune function and collagen production',
'Muntingia Calabura (Jamaica Cherry-Gasagase)':'Antiseptic properties and therapeutic uses of the flowers include the treatment of abdominal cramps and spasms',
'Murraya Koenigii (Curry)':'They are used as antihelminthics, analgesics, digestives, and appetizers in Indian cookery . The green leaves of M. koenigii are used in treating piles, inflammation, itching, fresh cuts, dysentery, bruises, and edema. The roots are purgative to some extent',
'Nerium Oleander (Oleander)':'Anvirze is an aqueous extract of the plant Nerium oleander which has been utilized to treat patients with advanced malignancies . Other medicinal uses of Nerium oleander include treating ulcers, haemorrhoids, leprosy, to treat ringworm, herpes, and abscesses',
'Nyctanthes Arbor-tristis (Parijata)':'The leaves of the Nyctanthes arbor-tristis plant find their use in Ayurveda and Homoeopathy for the treatment of sciatica, fevers, and arthritis. They are also used as a laxative for treating constipation. The plant has properties that help treat snake bites',
'Ocimum Tenuiflorum (Tulsi)':'Tulsi has also been shown to counter metabolic stress through normalization of blood glucose, blood pressure and lipid levels, and psychological stress through positive effects on memory and cognitive function and through its anxiolytic and anti-depressant properties',
'Piper Betle (Betel)':'Traditionally, the plant is used to cure many ailments such as cold, bronchial asthma, cough, stomachalgia and rheumatism',
'Plectranthus Amboinicus (Mexican Mint)':'It is widely used in folk medicine to treat conditions like cold, asthma, constipation, headache, cough, fever and skin diseases. The leaves of the plant are often eaten raw or used as flavoring agents, or incorporated as ingredients in the preparation of traditional food',
'Pongamia Pinnata (Indian Beech)':'The Indian beech, Pongam seed oil tree or Hongay seed oil, is a medium-sized, glabrous, semi-evergreen tree. The fruits, bark, seeds, seed oil, leaves, roots and flowers of Pongamia pinnata have all been recommended for analgesic, arthritis and inflammatory activity',
'Psidium Guajava (Guava)':'Although guava has a number of medicinal properties, it is the most common and popular traditional remedy for gastrointestinal infections such as diarrhea, dysentery, stomach aches, and indigestion and it is used across the world for these ailments',
'Punica Granatum (Pomegranate)':'The pomegranate polyphenol; punicalagin, is known to have potent anticancer activity in breast, lung, and cervical cells. All parts of the fruit were reported to have therapeutic activity including anticancer, anti-inflammatory, anti-atherogenic, anti-diabetes, hepato protective, and antioxidant activity, etc',
'Santalum Album (Sandalwood)':'Sandalwood has antipyretic, antiseptic, antiscabetic, and diuretic properties. It is also effective in treatment of bronchitis, cystitis, dysuria, and diseases of the urinary tract',
'Syzygium Cumini (Jamun)':'The bark is acrid, sweet, digestive, astringent to the bowels, anthelmintic and used for the treatment of sore throat, bronchitis, asthma, thirst, biliousness, dysentery and ulcers. It is also a good blood purifier',
'Syzygium Jambos (Rose Apple)':'Rose apple has a long history of being used in traditional and folk medicine in various cultures. In the Chinese system of traditional medicine, the fruit and root bark are believed to be of use as a blood coolant. The fruit has been used as a diuretic and as a tonic for better health of the brain and liver',
'Tabernaemontana Divaricata (Crape Jasmine)':'Tabernaemontana divaricata has several uses in medicine. In Ayurvedic medicine, the juice from the flower buds is mixed with oil and applied to the skin to treat inflammation. It is also used in dental care, for scabies, as cough medicine and for eye ailments',
'Trigonella Foenum-graecum (Fenugreek)':'Fenugreek is a herb that is widely used in cooking and as a traditional medicine for diabetes in Asia. It has been shown to acutely lower postprandial glucose levels, but the long-term effect on glycemia remains uncertain'


}


@csrf_exempt
def Upload_image(request):

	encodedstr=request.POST.get("encodedstr")
   
	print(encodedstr)

	try:
	
	   
		base64_img_bytes = encodedstr.encode('utf-8')
		with open('test.jpg', 'wb') as file_to_save:
			decoded_image_data = base64.decodebytes(base64_img_bytes)
			file_to_save.write(decoded_image_data)

		rec_text=test_image('test.jpg')
		print("result--->",rec_text)   
		defval=usedict[rec_text] 
		
		data={}
		data['msg']=rec_text
		data['def']=defval
		
		return JsonResponse(data,safe=False)
	except Exception as e:
		print(e)
		data={"msg":"no"}
		return JsonResponse(data,safe=False)

