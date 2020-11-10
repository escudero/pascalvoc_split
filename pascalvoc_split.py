import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import os
from glob import glob
import sys, getopt


def parse_xml(pascalvoc_path):
    file_list = []
    parse_list = []
    names_list = []
    for xml_file in glob(os.path.join(pascalvoc_path, 'Annotations', '*.xml')):
        tree = ET.parse(open(xml_file))
        root = tree.getroot()
        image_name = root.find('filename').text
        image_name = '.'.join(image_name.split('.')[:-1])
        klass_names = [obj.find('name').text for obj in root.iter('object')]
        names_list.extend(klass_names)
        if len(klass_names) == 0:
            klass_names = ['X']
        klass_names = '#'.join(klass_names)
        file_list.append(image_name)
        parse_list.append(klass_names)
    names_list = list(set(names_list))
    return file_list, parse_list, names_list

def run_pascalvoc_split(pascalvoc_path, test_size, random_state=2):
    file_list, parse_list, names_list = parse_xml(pascalvoc_path)
    X_train, X_test, y_train, y_test = train_test_split(file_list, parse_list, test_size=test_size, random_state=random_state)
    y_train = [e.split('#') for e in y_train]
    y_test = [e.split('#') for e in y_test]
    for klass_name in names_list:
        with open(os.path.join(pascalvoc_path, 'ImageSets', 'Main', f'{klass_name}_train.txt'), 'w') as filehandle:
            for filename, klasses in zip(X_train, y_train):
                flag = ' 1' if klass_name in klasses else '-1'
                filehandle.writelines(f"{filename} {flag}\n")
        with open(os.path.join(pascalvoc_path, 'ImageSets', 'Main', f'{klass_name}_val.txt'), 'w') as filehandle:
            for filename, klasses in zip(X_test, y_test):
                flag = ' 1' if klass_name in klasses else '-1'
                filehandle.writelines(f"{filename} {flag}\n")

if __name__ == '__main__':
    command_help = 'pascalvoc_split.py --pascalvoc_path <pascalvocpath> --test_size <float>'
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, 'h', ['=', 'pascalvoc_path=', 'test_size='])
    except getopt.GetoptError:
        print(command_help)
        sys.exit(2)
    pascalvoc_path, test_size = [None, None]
    for opt, arg in opts:
        if opt == '-h':
            print(command_help)
            sys.exit()
        elif opt in ("--pascalvoc_path"):
            pascalvoc_path = arg
        elif opt in ("--test_size"):
            test_size = float(arg)
    if None in [pascalvoc_path, test_size]:
        print(command_help)
        sys.exit()

    run_pascalvoc_split(pascalvoc_path, test_size)

    
