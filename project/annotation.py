import glob

def main(glob_path_input_img, annotation_file_path):

    gt_paths = sorted(glob.glob(glob_path_input_img))

    with open(annotation_file_path, 'w') as f:
        for p in gt_paths:
            filename = p.split('/')[-1]
            line = f'{filename} (512,512,3)\n'
            f.write(line)
        
    #with open(annotation_file_path,'r') as f:
    #    file_content =f.read()
    #    print(file_content)


if __name__ == '__main__':

    glob_input_img = './data/train/GT/*.png'
    annotation_file = 'data/train_ann.txt'

    main(glob_input_img, annotation_file)