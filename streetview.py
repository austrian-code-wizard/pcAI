from PIL import Image
from io import BytesIO
import googlemaps
import polyline
import requests
import json
import csv

API_KEY = "AIzaSyBws6z5GqhwZ6MiDyMRL32napv0PSK-7FI"
counties = ["CorpusChristi2018.JSON", ]

gmaps = googlemaps.Client(key=API_KEY)

with open("CSV_Data/PCI.csv", mode='w', newline='') as csvfile:
    csvwriter = csv.writer(
        csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    counter = 1
    for county in counties:

        with open(f'PCI_Data/{county}') as f:
            data = json.load(f)

            for road in data["features"]:
                try:
                    directions_result = gmaps.directions(road["attributes"]["BEGIN_LOCATION"],
                                                         road["attributes"]["END_LOCATION"],
                                                         mode="driving",
                                                         avoid="indoor")
                    way_points = polyline.decode(
                        directions_result[0]['overview_polyline']['points'])
                    for i in range(0, len(way_points), 5):
                        lat = way_points[i][0]
                        lng = way_points[i][1]
                        response = requests.get(
                            f'https://maps.googleapis.com/maps/api/streetview?size=512x512&location={lat},{lng}&fov=80&heading=110&pitch=-45&key={API_KEY}')

                        metadata = requests.get(
                            f'https://maps.googleapis.com/maps/api/streetview/metadata?size=512x512&location={lat},{lng}&fov=80&heading=110&pitch=-45&key={API_KEY}')
                        metadata = metadata.json()

                        if response.status_code == 200 and metadata["status"] == "OK":
                            image = Image.open(BytesIO(response.content))
                            with open(f'Images/image_{counter}.jpg', "wb") as f:
                                image.save(f)
                            csvwriter.writerow(
                                [f'image_{counter}', road['attributes']['PCI_SCORE']])
                            counter += 1
                except:
                    pass
