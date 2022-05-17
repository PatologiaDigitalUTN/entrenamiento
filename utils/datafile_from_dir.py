import os
import json

def create_datafile(rootdir, dest_file, dest_label_to_content):
    output = open(dest_file, "w")
    label_idx = -1
    label = ""
    last_label = ""
    label_to_content = {}

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.normpath(os.path.join(subdir, file))
            path = path.replace(os.path.normpath(rootdir), '')

            label = path.split("\\")[1]

            if (label != last_label):
                last_label = label
                label_idx = label_idx + 1
                label_to_content[label_idx] = label

            output.write(path[1:]+' '+str(label_idx)+'\n')
    output.close()
    create_label_to_content_file(label_to_content, dest_label_to_content)


def create_label_to_content_file(label_to_content, dest_label_to_content):
    json_object = json.dumps(label_to_content, indent = 4) 
    label_to_content_file = open(dest_label_to_content, "w")
    label_to_content_file.write(json_object)
    label_to_content_file.close()

create_datafile('C:/Implantes/Implants/', 'C:/Implantes/test.txt', 'C:/Implantes/label_to_content.json')