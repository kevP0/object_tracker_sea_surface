"""
Generates COCO data and annotation structure from drone data.
"""
import json
import os
import cv2

from generate_coco_from_mot import check_coco_from_mot

DATA_ROOT = 'data/drone'
TRAIN = 'instances_train_objects_in_water.json'
TEST = 'instances_test_objects_in_water.json'
VAL = 'instances_val_objects_in_water.json'

def generate_coco_from_drone(split_name='train_val', split='train_val'):
    """
    Generate COCO data from drone.
    """
    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = []
    annotations['annotations'] = []
    annotations['sequences'] = []
    annotations['frame_range'] = []

    # open original annotations file
    orig_annotations_path = os.path.join(DATA_ROOT, f'annotations/{split_name}')
    # open json file
    with open(orig_annotations_path, 'r+') as orig_annotations_file:
        orig_annotations = json.load(orig_annotations_file)

    # for each video_id find lowest image_id and highest image_id
    video_lowest_image_id = {}
    video_highest_image_id = {}
    for image in orig_annotations['images']:
        video_id = image['video_id']
        image_id = image['id']
        if video_id in video_lowest_image_id:
            if image_id < video_lowest_image_id[video_id]:
                video_lowest_image_id[video_id] = image_id
        else:
            video_lowest_image_id[video_id] = image_id

        if video_id in video_highest_image_id:
            if image_id > video_highest_image_id[video_id]:
                video_highest_image_id[video_id] = image_id
        else:
            video_highest_image_id[video_id] = image_id

    id_counter = 0
    first_frame_id = 0
    # IMAGES
    for image in orig_annotations['images']:

        seq_len = video_highest_image_id[image['video_id']] - video_lowest_image_id[image['video_id']] + 1

        original_id = image['id']
        if(original_id == int(video_lowest_image_id[image['video_id']])):
            first_frame_id = id_counter
    
        file_name = image['file_name'].split('.')[0]
        file_name += '.jpg'
        annotations['images'].append({
            "file_name": file_name,
            "height": image['height'],
            "width": image['width'],
            "id": id_counter,
            "frame_id": id_counter - first_frame_id,
            "seq_length": seq_len,
            "first_frame_image_id": first_frame_id
        })
        id_counter += 1

    # categories are same as original
    annotations['categories'] = orig_annotations['categories']

    # convert annotations to COCO format
    # iterate over all annotations
    # skip if it is test split
    image_id = 0
    last_id = 0

    name = ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
            '06', '07', '08', '09', '10', '11', '12', '13', '14',
            '15', '16', '17', '18', '19', '20']
    already_added = []
    if not split == 'test':
        for annotation_json in orig_annotations['annotations']:

            ignore = False
            visiblity = 1.0
            original_image_id = annotation_json['id']
            if original_image_id != last_id:
                image_id += 1
                last_id = original_image_id

            # vmesni seq generator
            index_of_seq = int(annotation_json['video_id'])
            seq = f'MOT17-{name[index_of_seq]}-ALL'
            if str(annotation_json['video_id']) not in already_added:
                already_added.append(str(annotation_json['video_id']))
            # seq = "video_" + str(annotation_json['video_id'])
            # if seq not in annotations['sequences']:
            #     annotations['sequences'].append(seq)
            
            annotation = {
                "id": annotation_json['id'],
                "bbox": annotation_json['bbox'],
                "image_id": image_id,
                "segmentation": [],
                "ignore": int(ignore),
                "visibility": visiblity,
                "area": annotation_json['area'],
                "iscrowd": 0,
                "seq": str(seq),
                "category_id": annotation_json['category_id'],
                "track_id": annotation_json['track_id']
            }

            annotations['annotations'].append(annotation)

        # max objs per image
        num_objs_per_image = {}
        for anno in annotations['annotations']:
            image_id = anno["image_id"]
            if image_id in num_objs_per_image:
                num_objs_per_image[image_id] += 1
            else:
                num_objs_per_image[image_id] = 1

        print(f'max objs per image: {max([n for n  in num_objs_per_image.values()])}')
        print(len(annotations['images']))

    # edit frame range
    annotations['frame_range'] = {
            "start": 0.0,
            "end": 1.0
        }

    # save annotations to json file
    annotations_path = os.path.join(DATA_ROOT, f'annotations/{split}.json')
    print("Annotation path: ", annotations_path)
    with open(annotations_path, 'w+') as annotations_file:
        json.dump(annotations, annotations_file, indent=4)

if __name__ == '__main__':
    generate_coco_from_drone(split_name=TRAIN, split='train')
    generate_coco_from_drone(split_name=VAL, split='val')
    # generate_coco_from_drone(split_name=TEST, split='test')

    coco_dir = os.path.join('data/drone', 'train')
    annotation_file = os.path.join('data/drone/annotations', 'train.json')
    check_coco_from_mot(coco_dir, annotation_file, img_id=1000)

""" GOAL IS TO GENERATE STRUCTURE LIKE THIS:
    {
        "type": "instances,
        "images" : [
            {
                "file_name": "ime slike",
                "height": int,
                "width": int,
                "id": int,
                "frame_id": int,
                "seq_length": int,
                "first_frame_image_id": int
            },
            ...
        ],
        "categories": [
            {
                "supercategory": "person",
                "name": "person",
                "id": int
            }
        ],
        "annotations": [
            {
                "id": int  # ti sam narascajo za vsako detekcijo
                "bbox": [
                    int,
                    int,
                    int,
                    int
                ],
                "image_id": int,
                "segmentation": [],
                "ignore": int,
                "visibility": float,
                "area" : int,
                "iscrowd": int, <- 0
                "seq": string, <- video_id
                "category_id": int,
                "track_id: int
            }
        ],
        "sequences": [
            string
        ],
        "frame_range": {
            "start": 0.0,
            "end": 1.0
        }
    }
    """