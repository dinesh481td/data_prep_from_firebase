
# Import necessary libraries
from google.cloud import storage
import requests
import logging
import json
import os
from datetime import timedelta, datetime
from google.cloud import firestore
from concurrent.futures import ThreadPoolExecutor, as_completed
import certifi
from google.api_core.retry import Retry
import ast
from sklearn.model_selection import train_test_split
import shutil
import yaml
from github import Github,GithubException
from pathlib import Path
import platform
import subprocess
from github import Github, UnknownObjectException
import dvc.api
import dvc.repo
from dvc.repo import Repo
from dvc.exceptions import DvcException



def check_dvc_cache_directory():
    dvc_cache_path = os.path.join(os.getcwd(), '.dvc', 'cache')
    if os.path.exists(dvc_cache_path):
        print(f"The DVC cache directory '{dvc_cache_path}' exists.")
    else:
        print(f"The DVC cache directory '{dvc_cache_path}' does not exist or is not accessible.")
        source_path = "/home/setup/dinesh001/.dvc/cache"
        destination_path = dvc_cache_path
        try:
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            print(f"Cache successfully transferred from '{source_path}' to '{destination_path}'.")
        except Exception as e:
            print(f"Error transferring cache directory: {e}")



def dvc_pull():
    try:
        # repo = Repo('.')
        repo.pull(targets=['Dataset.dvc'])
        print("DVC pull successful")
    except DvcException as e:
        print(f"DVC pull failed: {e}")

def dvc_add_and_push():
    try:
        # repo = Repo('.')
        subprocess.run(['dvc', 'unprotect', 'Dataset'], check=True)
        repo.add('Dataset')
        repo.push(targets=['Dataset.dvc'])
        print("Data added and pushed to DVC")
    except DvcException as e:
        print(f"DVC add/push failed: {e}")


# Define the function to split the URL and return the first part
def check_url_split(url):
    if isinstance(url, (list, str)):
        url_parts = url[0].split('/') if isinstance(url, list) else url.split('/')
        return "/".join(url_parts)
    else:
        print(f"Invalid input. Expected a string or a list containing a single string. Got: {type(url)}")
        return None

# Define the function to get the labels
def get_labels(i,label_dir = 'labels'):
    if 'bbox' in i and i['bbox'] and i['bbox'] != 'None':
        url = check_url_split(i['file_name'])
        ext = "." + url.split('.')[-1]
        print(ext)
        file_name = check_url_split(i['file_name']).split('/')[-1].replace(ext, '.txt')
        txt_output = Path(os.getcwd()) / label_dir / file_name
        txt_output.parent.mkdir(parents=True, exist_ok=True)

        if 'normalized_bbox' not in i:
            bbox = i['bbox']
            img_width = int(i['width'][0])
            img_height = int(i['height'][0])
            category_ids = [int(cat_id) for cat_id in i['category_id']]

            with open(txt_output, 'w') as f:
                for bx in range(len(bbox)):
                    try:
                        coordinates = ast.literal_eval(bbox[bx])
                        x_center = (coordinates[0] + (coordinates[2] / 2.0)) / img_width
                        y_center = (coordinates[1] + (coordinates[3] / 2.0)) / img_height
                        width = min(coordinates[2] / img_width, 1)
                        height = min(coordinates[3] / img_height, 1)
                        x_center = min(max(x_center, 0), 1)
                        y_center = min(max(y_center, 0), 1)
                        f.write(f"{category_ids[bx]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    except (ValueError, IndexError, KeyError) as e:
                        print(f"Error processing bounding box: {e}")
        else:
            normalized_bbox = i['normalized_bbox']
            with open(txt_output, 'w') as f:
                for bx in normalized_bbox:
                    try:
                        f.write(f"{bx}\n")
                    except (ValueError, IndexError, KeyError) as e:
                        print(f"Error processing bounding box: {e}")

    else:
        url = check_url_split(i['file_name'])
        ext = "." + url.split('.')[-1]
        print(ext)  
        file_name = url.split('/')[-1].replace(ext, '.txt')
        txt_output = Path(os.getcwd()) / label_dir / file_name
        txt_output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(txt_output, 'w') as f:
            f.write('')


# Define the function to get the attributes
def get_attributes(i,attr_dir = 'atributes'):
    try:
        i['average_green'] = float(i.get('average_green')[0])
        i['area_of_image'] = float(i.get('area_of_image')[0])
        if i.get('bbox') is not None:
            i['bbox'] = [ast.literal_eval(i) for i in i.get('bbox')]
        i['average_red'] = float(i.get('average_red')[0])
        i['contrast'] = float(i.get('contrast')[0])
        i['width'] = int(i.get('width')[0])
        if i.get('id') is not None:
            i['id'] = [int(i) for i in i.get('id') if i is not None]

        if i.get('object_count') is not None:
            i['object_count'] = int(i.get('object_count')[0])
        i['brightness'] = float(i.get('brightness')[0])
        i['sharpness'] = float(i.get('sharpness')[0])

        if i.get('area') is not None:
            i['area'] = [int(i) for i in i.get('area')]
        i['aspect_ratio'] = float(i.get('aspect_ratio')[0])

        if i.get('category_id') is not None:
            i['category_id'] = [int(i) for i in i.get('category_id')]
        i['height'] = int(i.get('height')[0])

        if i.get('image_id') is not None:
            i['image_id'] = [int(i) for i in i.get('image_id')]
        i['average_blue'] = float(i.get('average_blue')[0])

        if i.get('object_density') is not None:
            i['object_density'] = float(i.get('object_density')[0])
        i['file_name'] = check_url_split(i.get('file_name'))
        if i.get('bbox_area') is not None:
            i['bbox_area_mean'] = float(i.get('bbox_area_mean')[0])
        file_name = check_url_split(i.get('file_name')).split('/')[-1]
        print(file_name)
        os.makedirs(os.path.join(os.getcwd(),"Dataset",attr_dir), exist_ok=True)
        atribute_path = os.path.join(os.getcwd(),"Dataset",attr_dir,file_name +'.json')
        with open(atribute_path,'w') as f:
            json.dump(i,f)
    except Exception as e:
        print(e)
        print(i)

# Define the function to get the signed URL
def get_signed_url(url):

    bucket_name = "".join(url.split('/')[3])
    blob_name = "/".join(url.split('/')[-2:])
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    signed_url = blob.generate_signed_url(expiration=timedelta(hours=1))
    return signed_url

def download_image_from_url(url: str):
    img_name = url.split('/')[-1]
    signed_url = get_signed_url(url)
    response = requests.get(signed_url,verify = certifi.where())
    img_path = os.path.join(os.getcwd(),'images',img_name)
    with open(img_path, 'wb') as f:
        f.write(response.content)




# Define the function to split the data
def split_data_yolov8(image_folder_local, labels_folder_local, train_ratio, validation_ratio):
    # Get set of file names in directories
    os.makedirs(os.path.join(os.getcwd(), 'Dataset','train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),  'Dataset','val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),  'Dataset','train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),  'Dataset','val', 'labels'), exist_ok=True)

    # Remove the extension from the image files
    image_folder_local = [f for f in image_folder_local if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
    # Perform the split
    train_filenames, validation_filenames = train_test_split(
        image_folder_local, train_size=train_ratio, test_size=validation_ratio, random_state=42
    )
    # Move the files
    for filename in train_filenames:
        ext = "." + check_url_split(filename).split('.')[-1]
        label_file_name = filename.replace(ext,'.txt')

        if filename.replace(ext,'.txt') in labels_folder_local:
            shutil.move(os.path.join(os.path.join(os.getcwd(), 'images'), filename ), os.path.join(os.path.join(os.getcwd(),'Dataset', 'train','images'), filename))
            shutil.move(os.path.join(os.path.join(os.getcwd(), 'labels'), label_file_name), os.path.join(os.path.join(os.getcwd(), 'Dataset', 'train', 'labels'), label_file_name))

    for filename in validation_filenames:
        ext = "." + check_url_split(filename).split('.')[-1]
        label_file_name = filename.replace(ext,'.txt')
        if filename.replace(ext,'.txt') in labels_folder_local:
            shutil.move(os.path.join(os.path.join(os.getcwd(), 'images'), filename ), os.path.join(os.path.join(os.getcwd(),'Dataset','val', 'images'), filename))
            shutil.move(os.path.join(os.path.join(os.getcwd(), 'labels'), label_file_name), os.path.join(os.path.join(os.getcwd(),'Dataset','val', 'labels'), label_file_name))

# Define the function to extract the number of classes
def extract_no_of_classes(list_docs):
    classes = []
    
    for i in list_docs:
        i = ast.literal_eval(json.dumps(i))
        if i.get('category_id') is not None:
            cat = [int(i) for i in i.get('category_id') if i is not None]
            if cat is not None:
                classes.extend(cat)
    return list(set(classes))

    
    
def yolo_to_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format (x_center, y_center, bbox_width, bbox_height) to (x, y, width, height).
    
    :param yolo_bbox: YOLO format bounding box (x_center, y_center, bbox_width, bbox_height)
    :param img_width: Width of the image
    :param img_height: Height of the image
    :return: Bounding box in (x, y, width, height) format
    """
    x_center, y_center, bbox_width, bbox_height = yolo_bbox
    
    x = (x_center - bbox_width / 2) * img_width
    y = (y_center - bbox_height / 2) * img_height
    width = bbox_width * img_width
    height = bbox_height * img_height
    
    return [int(x), int(y), int(width), int(height)]



# Define the function to set the credentials
def set_credentials(filepath: str):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = filepath

print(os.getcwd())
# os.chdir(os.path.join(os.getcwd(),"data_prep_from_firebase"))
# os.chdir(os.path.dirname(os.getcwd()))
print(os.getcwd())


set_credentials(os.path.join(os.getcwd(), 'keys',"gcp_key.json"))


g = Github('Token')
repo = g.get_repo('repo_name')
contents = repo.get_contents('DVC_datatset_file_name.dvc')
decoded = contents.decoded_content
with open(os.path.join(os.getcwd(),'DVC_datatset_file_name.dvc'), 'wb') as f:
    f.write(decoded)




# Define the base directory
base_dir = Path(".dvc/tmp")

# Define the specific lock files to delete
lock_file_path = base_dir / "lock"
rwlock_file_path = base_dir / "rwlock"

# Function to delete a file if it exists
def delete_file(file_path):
    if file_path.exists():
        file_path.unlink()
        print(f"Deleted {file_path}")
    else:
        print(f"{file_path} does not exist")

# Delete specific lock files
delete_file(lock_file_path)
delete_file(rwlock_file_path)

# Delete all .lock files in the directory
for lock_file in base_dir.glob("*.lock"):
    delete_file(lock_file)

check_dvc_cache_directory()

# Initialize DVC repo
repo = Repo()

dvc_pull()

set_credentials(os.path.join(os.getcwd(), 'keys',"firebase creds.json"))
image_files_local = os.listdir(os.path.join(os.getcwd(), 'Dataset','train', 'images')) + os.listdir(os.path.join(os.getcwd(), 'Dataset','val', 'images'))
# image_files_local = os.listdir(os.path.join(os.getcwd(), 'images')) 

print("image_files_local : ",len(image_files_local))


image_folder = [i for i in image_files_local if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg')]
list_docs = []
db = firestore.Client()
data = db.collection('sample_project')
data = data.stream(retry=Retry(deadline=60))
for i in data:
    list_docs.append(i.to_dict())


print("list_docs from firebase : ",len(list_docs))


filtered_img_files = [
    i for i in list_docs 
    if i.get('file_name') is not None and check_url_split(i.get('file_name')) and check_url_split(i.get('file_name')).split('/')[-1]  not in image_folder
]
print("filtered_img_files from firebase which is not present in Dataset.dvc : ",len(filtered_img_files)) 


# Create necessary directories
for dir_name in ['labels', 'images']:
    os.makedirs(os.path.join(os.getcwd(), dir_name), exist_ok=True)
counters = 0
for i in filtered_img_files:
    i = ast.literal_eval(json.dumps(i))
    if check_url_split(i.get('file_name')).split('/')[-1].endswith(('.jpg', '.png', '.jpeg')): 
        get_labels(i)
        get_attributes(i)
        url_name = i['file_name']
        url = check_url_split(url_name)
        local_path = os.path.join(os.getcwd(), *url.split('/')[-2:])

        if url_name not in image_folder:
            print(url)
            download_image_from_url(url)
        counters += 1
        print("--------->",counters)



# List filenames in the images and labels directories
image_folder_local = os.listdir(os.path.join(os.getcwd(), 'images'))
labels_folder_local = os.listdir(os.path.join(os.getcwd(), 'labels'))
print(f"Number of images: {len(image_folder_local)}")
if len(image_folder_local) != 0:
    # Specify a ratio for splitting
    train_ratio = 0.75
    validation_ratio = 0.25
    split_data_yolov8(image_folder_local, labels_folder_local, train_ratio, validation_ratio)
    
    
    categories =  [
      {
       "id": 0,
       "name": "UPSTruck"
      },
      {
       "id": 1,
       "name": "FerrariTruck"
      },
      {
       "id": 2,
       "name": "FedexTruck"
      },
      {
       "id": 3,
       "name": "DHLTruck"
      },
      {
       "id": 4,
       "name": "USPSTruck"
      },
      {
       "id": 5,
       "name": "Person"
      },
      {
       "id": 6,
       "name": "RolexLogo"
      },
      {
       "id": 7,
       "name": "Truck"
      },
      {
       "id": 8,
       "name": "DoorOpen"
      },
      {
       "id": 9,
       "name": "AmazonTruck"
      },
      {
       "id": 10,
       "name": "GarbageCan"
      },
      {
       "id": 11,
       "name": "Dog"
      },
      {
       "id": 12,
       "name": "Package"
      },
      {
       "id": 13,
       "name": "Cleaning"
      },
      {
       "id": 14,
       "name": "Laptop"
      },
      {
       "id": 15,
       "name": "Car"
      },
      {
       "id": 16,
       "name": "Motorbike"
      },
      {
       "id": 17,
       "name": "PersonwithPPE"
      },
      {
       "id": 18,
       "name": "PersonwithoutPPE"
      },
      {
       "id": 19,
       "name": "PersonwithPPE_Mask"
      },
      {
       "id": 20,
       "name": "Helmet"
      },
      {
       "id": 21,
       "name": "Mobile_Phone"
      },
      {
       "id": 22,
       "name": "Glouse"
      },
      {
       "id": 23,
       "name": "Eye_Glass"
      },
      {"id": 24, "name": "gun"},
     ]
    # classes = extract_no_of_classes(list_docs)
    # Define the configuration
    # data_config = {"train" : os.path.join(os.getcwd(),'train'), "validation" : os.path.join(os.getcwd(), 'validation' ), "nc" : len(classes), "names" : {i['id'] : i['name'] for i in categories if i['id'] in classes}}
    data_config = {"path" : os.path.join(os.getcwd(),"Dataset") ,"train" : os.path.join(os.getcwd(),"Dataset",'train'), "val" : os.path.join(os.getcwd(), "Dataset",'val' ), "nc" : len(categories), "names" : {i['id'] : i['name'] for i in categories }}
    
    with open(os.path.join(os.getcwd(), 'data.yaml'), 'w') as f:
        yaml.dump(data_config, f)
    
    
    # os.system('dvc add Dataset')
    # os.system('dvc push Dataset.dvc')
    set_credentials(os.path.join(os.getcwd(), 'keys',"gcp_key.json"))
    dvc_add_and_push()
    # print("Data added and pushed to DVC moving to next steps")
    
    image_folder = os.listdir(os.path.join(os.getcwd(), 'Dataset','train', 'images')) + os.listdir(os.path.join(os.getcwd(), 'Dataset','val', 'images'))
    
    images = []
    annotations = []
    ann_count = 1
    img_count = 1 
    for i in list_docs:
        i = ast.literal_eval(json.dumps(i))
        if i.get('file_name') is not None and check_url_split(i.get('file_name')).split('/')[-1] in image_folder:
            images.append({
                'id': img_count,
                'width': i.get('width')[0],
                'height': i.get('height')[0],
                'file_name': check_url_split(i.get('file_name'))
            })
            
            if i.get('bbox') is not None:
                bbox = i.get('bbox')
                category_ids = [int(cat_id) for cat_id in i.get('category_id')]
                norm = i.get('normalized_bbox')
                for index, bbox_string in enumerate(bbox):
                    bbox_coordinates = ast.literal_eval(bbox_string)
                    if i.get('normalized_bbox') and i.get('normalized_bbox') is not None:
                        yolox_bbox = ast.literal_eval(norm[index].replace(" ",","))[1:]
                        annotations.append({
                            'id': ann_count,
                            'image_id': img_count,
                            'bbox': yolo_to_bbox(yolox_bbox, int(i.get('width')[0]), int(i.get('height')[0])),
                            'category_id': category_ids[index] 
                            })
                        ann_count += 1
                    else:
                        annotations.append({
                            'id': ann_count,
                            'image_id': img_count,
                            'bbox': bbox_coordinates,
                            'category_id': category_ids[index] 
                            })
                        ann_count += 1
            img_count += 1
            # annotations.append({'normalized_bbox': i.get('normalized_bbox')})
    data = {'images': images, 'annotations': annotations, 'categories': categories}
    with open(os.path.join(os.getcwd(), 'firebase_coco_data.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print("firebase_coco_data has been created")
            
    
    
    g = Github('Token')
    repo = g.get_repo('Repo_name')
    files_to_update = {
        "data1.yaml": "data.yaml",
        "Dataset.dvc": "Dataset.dvc"
    }
    
    
    for file_path, local_file in files_to_update.items():
        try:
            # Attempt to get the contents of the file
            contents = repo.get_contents(file_path, ref="main")
            # If successful, update the file
            with open(os.path.join(os.getcwd(), local_file), 'r') as file:
                file_content = file.read()
                repo.update_file(contents.path, f"Update {file_path}", file_content, contents.sha, branch="main")
        except UnknownObjectException:
            # If the file does not exist, create it
            with open(os.path.join(os.getcwd(), local_file), 'r') as file:
                file_content = file.read()
                repo.create_file(file_path, f"Add {file_path}", file_content, branch='main')
    
    
    
    def dvc_cache_transfer():
        dvc_cache_path = os.path.join(os.getcwd(), '.dvc', 'cache')
        if os.path.exists(dvc_cache_path):
            print(f"The DVC cache directory '{dvc_cache_path}' exists.")
            destination_path = "/home/setup/dinesh001/.dvc/cache"
            try:
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                shutil.copytree(dvc_cache_path, destination_path, dirs_exist_ok=True)
                print(f"Cache successfully transferred from '{dvc_cache_path}' to '{destination_path}'.")
            except Exception as e:
                print(f"Error transferring cache directory: {e}")
        else:
            print(f"The DVC cache directory '{dvc_cache_path}' does not exist or is not accessible.")
    
    
    dvc_cache_transfer()

else:
    print(f"There is no Data to add in dvc : ----> data in the images dir = {image_folder_local} ")
    print("dvc --> Up to date") 
    categories =  [
      {
       "id": 0,
       "name": "UPSTruck"
      },
      {
       "id": 1,
       "name": "FerrariTruck"
      },
      {
       "id": 2,
       "name": "FedexTruck"
      },
      {
       "id": 3,
       "name": "DHLTruck"
      },
      {
       "id": 4,
       "name": "USPSTruck"
      },
      {
       "id": 5,
       "name": "Person"
      },
      {
       "id": 6,
       "name": "RolexLogo"
      },
      {
       "id": 7,
       "name": "Truck"
      },
      {
       "id": 8,
       "name": "DoorOpen"
      },
      {
       "id": 9,
       "name": "AmazonTruck"
      },
      {
       "id": 10,
       "name": "GarbageCan"
      },
      {
       "id": 11,
       "name": "Dog"
      },
      {
       "id": 12,
       "name": "Package"
      },
      {
       "id": 13,
       "name": "Cleaning"
      },
      {
       "id": 14,
       "name": "Laptop"
      },
      {
       "id": 15,
       "name": "Car"
      },
      {
       "id": 16,
       "name": "Motorbike"
      },
      {
       "id": 17,
       "name": "PersonwithPPE"
      },
      {
       "id": 18,
       "name": "PersonwithoutPPE"
      },
      {
       "id": 19,
       "name": "PersonwithPPE_Mask"
      },
      {
       "id": 20,
       "name": "Helmet"
      },
      {
       "id": 21,
       "name": "Mobile_Phone"
      },
      {
       "id": 22,
       "name": "Glouse"
      },
      {
       "id": 23,
       "name": "Eye_Glass"
      },
      {"id": 24, "name": "gun"},
     ]
    image_folder = os.listdir(os.path.join(os.getcwd(), 'Dataset','train', 'images')) + os.listdir(os.path.join(os.getcwd(), 'Dataset','val', 'images'))
    
    images = []
    annotations = []
    ann_count = 1
    img_count = 1 
    for i in list_docs:
        i = ast.literal_eval(json.dumps(i))
        if i.get('file_name') is not None and check_url_split(i.get('file_name')).split('/')[-1] in image_folder:
            images.append({
                'id': img_count,
                'width': i.get('width')[0],
                'height': i.get('height')[0],
                'file_name': check_url_split(i.get('file_name'))
            })
            
            if i.get('bbox') is not None:
                bbox = i.get('bbox')
                category_ids = [int(cat_id) for cat_id in i.get('category_id')]
                norm = i.get('normalized_bbox')
                for index, bbox_string in enumerate(bbox):
                    bbox_coordinates = ast.literal_eval(bbox_string)
                    if i.get('normalized_bbox') and i.get('normalized_bbox') is not None:
                        yolox_bbox = ast.literal_eval(norm[index].replace(" ",","))[1:]
                        annotations.append({
                            'id': ann_count,
                            'image_id': img_count,
                            'bbox': yolo_to_bbox(yolox_bbox, int(i.get('width')[0]), int(i.get('height')[0])),
                            'category_id': category_ids[index] 
                            })
                        ann_count += 1
                    else:
                        annotations.append({
                            'id': ann_count,
                            'image_id': img_count,
                            'bbox': bbox_coordinates,
                            'category_id': category_ids[index] 
                            })
                        ann_count += 1
            img_count += 1
            # annotations.append({'normalized_bbox': i.get('normalized_bbox')})
    data = {'images': images, 'annotations': annotations, 'categories': categories}
    with open(os.path.join(os.getcwd(), 'firebase_coco_data.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print("firebase_coco_data has been created")
    
