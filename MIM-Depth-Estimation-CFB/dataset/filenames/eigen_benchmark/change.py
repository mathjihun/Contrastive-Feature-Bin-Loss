import os

file_list = ['train_list_original.txt', 'test_list_original.txt']
save = []

with open(file_list[0], 'r') as f:
    for p in f.readlines():
        x, y = p.split()

        yy = y.split('/')
        
        if 'train' in yy:
            yy.remove('train')
        elif 'val' in yy:
            yy.remove('val')

        y = '/'.join(yy)

        save.append(x + ' ' + y + '\n')

new_file_list = ['train_list.txt', 'test_list.txt']

with open(new_file_list[0], 'w') as f:
    f.writelines(save)
