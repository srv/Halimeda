from mecoda_minka import get_obs
import requests
import os

species = [
    "Asparagopsis taxiformis"
]

save_path = r"/home/slimbook/Escritorio/Minka"

for specie in species: 
    base = '_'.join(specie.split(' '))
    if not os.path.exists(os.path.join(save_path, base)):
        os.makedirs(os.path.join(save_path, base))
       
    obs = get_obs(query=specie, starts_on='2024-02-20') # num_max to limit each search
    for element in obs: 
        print(element)
        # print(element.photos[0])
        if element.quality_grade == 'research': # Minimum quality requested
            try: 
                url = element.photos[0].large_url
            except:
                continue  
            
            extension = url.split('.')[-1]  
            idx = element.id # Observation number
            filename = f"{base}_minka_{idx}"

            img_data = requests.get(url).content
            with open(os.path.join(save_path, base, filename + '.' + extension), 'wb') as handler:
                handler.write(img_data)