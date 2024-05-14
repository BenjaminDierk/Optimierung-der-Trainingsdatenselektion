import os
import pandas as pd
import random
import json
import queue
import csv

def erstelle_Dict(path_to_oens, List_of_all_Pg):
    
    data_dict = {}  # Das Dictionary, in dem alle Informationen gespeichert werden
    existing_oen_folders = [entry.name for entry in os.scandir(path_to_oens) if entry.is_dir()]

    # Schreibe alle PGs ins Dictionary und initialisiere bool und Zählvariablen
    for pg in List_of_all_Pg:
        
        data_dict[pg] = {'used': False, 'count': 0, 'OENs': {}}
        search_oen_list = [oen for oen in df.loc[df['productGroup'] == pg, 'OEN'].unique()]
        existing_oen_numbers = [oen for oen in search_oen_list if oen in existing_oen_folders]
    
        for oen_folder in existing_oen_numbers:    
            oen_path = os.path.join(path_to_oens, oen_folder)

            # Überprüfe, ob es sich um ein Verzeichnis handelt
            if os.path.isdir(oen_path):
                data_dict[pg]['OENs'][oen_folder] = {'used': False, 'Sets': {}}

                # Finde alle Sets und füge sie zum Dictionary hinzu
                for set_folder in os.scandir(oen_path):
                    # Gehe durch alle SET-Ordner
                    set_path = os.path.join(oen_path, set_folder)
                    
                    
                    if os.path.isdir(set_path):
                        data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name] = {'used': False, 'Bilder': {}}
                        
                        for bild in os.listdir(set_path):
                            if 'Part' in bild and not 'mask' in bild and bild.endswith('.jpg'):
                                data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name]['Bilder'][bild] = {'used': False}

                        # Wenn das Set keine Bilder enthält, setze es auf False
                        if not data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name]['Bilder']:
                            data_dict[pg]['OENs'][oen_folder]['Sets'][set_folder.name]['used'] = True

                    # Wenn der OEN-Ordner keine Sets enthält, setze ihn auf True
                    if not data_dict[pg]['OENs'][oen_folder]:
                        data_dict[pg]['OENs'][oen_folder]['used'] = True

    return data_dict



def choose_set_and_images(data_dict):
    # Sortiere die Produktgruppen nach der count-Variable aufsteigend
    sorted_pgs = sorted([pg for pg in pgs if not data_dict.get(pg, {}).get('used', True)], key=lambda pg: data_dict[pg]['count'])
    
    if not sorted_pgs:
        return  # Alle Produktgruppen sind bereits als 'used' markiert oder es gibt keine Produktgruppen
    
    pg = sorted_pgs[0]  
    pg_data = data_dict[pg]
    
    if not pg_data['OENs'] or all(oen_data['used'] for oen_data in pg_data['OENs'].values()):
        data_dict[pg]['used'] = True
        return
    
    oens_mit_unused_pg = [oen_name for oen_name, oen_data in data_dict[pg]['OENs'].items() if not oen_data['used']]
    
    if not oens_mit_unused_pg:
        data_dict[pg]['used'] = True
        return  # Alle OENs in der ausgewählten Produktgruppe sind bereits als 'used' markiert
    
    oen_name = random.choice(oens_mit_unused_pg)
    sets_in_oen = [set_name for set_name, set_data in data_dict[pg]['OENs'][oen_name]['Sets'].items() if not set_data['used']]
    
    if not sets_in_oen:
        data_dict[pg]['OENs'][oen_name]['used'] = True
        return  # Alle Sets in der ausgewählten OEN sind bereits als 'used' markiert
    
    set_name = random.choice(sets_in_oen)

    for bild_name, bild_data in data_dict[pg]['OENs'][oen_name]['Sets'][set_name]['Bilder'].items():
        if ('Top' in bild_name or 'Left' in bild_name or 'Right' in bild_name) and bild_name.endswith('jpg'):
            bild_pfad = os.path.join(path_to_oens, oen_name, set_name, bild_name)
            bild_queue.put(bild_pfad)
            data_dict[pg]['OENs'][oen_name]['Sets'][set_name]['Bilder'][bild_name]['used'] = True


    data_dict[pg]['OENs'][oen_name]['Sets'][set_name]['used'] = True
    data_dict[pg]['count'] += 1
    
    return


def remove_duplicates_from_queue(queue,cleaned_queue):
    seen = set()  # Set zur Speicherung eindeutiger Element

    # Durchlaufe die ursprüngliche Warteschlange
    while not queue.empty():
        element = queue.get()

        # Wenn das Element noch nicht gesehen wurde, füge es zur neuen Warteschlange hinzu und markiere es als gesehen
        if element not in seen:
            cleaned_queue.put(element)
            seen.add(element)

    return cleaned_queue


def display_partial_dict(data_dict, max_depth=2):
    partial_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict) and max_depth > 0:
            partial_dict[key] = display_partial_dict(value, max_depth=max_depth-1)
        else:
            partial_dict[key] = value
    return partial_dict





if __name__ == '__main__':
   
    path_to_oens = r'Z:\DataSets\EIBA\01_Verlesedaten\03_DataSet\Part Number'
    df = pd.read_csv(r'Z:\DataSets\EIBA\01_Verlesedaten\02_MetaDaten\data_objects\prodGroupData.csv')
    pgs = df['productGroup'].unique()
    bild_queue = queue.Queue()
    cleaned_queue = queue.Queue()
    image_queue_length = 5

    
    # Prüfe, ob das Dictionary verfügbar ist
    if os.path.exists('data_dict.json'):
        # Lade das vorhandene Dictionary
        with open('data_dict.json', 'r') as datei:
            data_dict = json.load(datei)

    else:
        # Erstelle das Dictionary, wenn es nicht vorhanden ist
        print('dict_data nicht gefunden. Erstelle neues Dictionary')
        data_dict = {}
        data_dict = erstelle_Dict(path_to_oens, pgs)



# Beispielaufruf
partial_data_dict = display_partial_dict(data_dict, max_depth=1)
formatted_partial_dict = json.dumps(partial_data_dict, indent=4)
print(formatted_partial_dict)





'''
# Beispielaufruf der Funktion zum Auswählen eines Bildes zur Bewertung
while bild_queue.qsize() < image_queue_length: 
    selected_image_path = choose_set_and_images(data_dict)
    if all(data_dict[pg]['used'] for pg in data_dict):
        break

cleaned_queue = remove_duplicates_from_queue(bild_queue, cleaned_queue)

print(cleaned_queue.qsize()) 

'''
