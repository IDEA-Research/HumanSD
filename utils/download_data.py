from pandas import read_parquet
import os
import requests

BASE_DIR="humansd_data/datasets/LaionAesthetics"

for idx in range(0,287):
    data = read_parquet(f"{BASE_DIR}/images/{str(idx).zfill(5)}.parquet")

    for i in range(len(data)):
        try:
            url=data["url"][i]
            key=data["key"][i]
            image_path=os.path.join(BASE_DIR,"images",str(idx).zfill(5),key)+'.jpg'
            
            if not os.path.exists(os.path.dirname(image_path)):
                os.makedirs(os.path.dirname(image_path))
                
            if os.path.exists(image_path):
                print(f"Image {key}.jpg already exists!")
                continue
                
            r = requests.get(url)
            
            if r.status_code!=200:
                print(f"Error! Unable to download image {key}.jpg")
                continue
            
            with open(image_path, 'wb') as f:
                f.write(r.content)    

            print(f"Sucessfully download image {key}.jpg")
        except:
            print(f"Error! Unable to download image {key}.jpg")
            continue