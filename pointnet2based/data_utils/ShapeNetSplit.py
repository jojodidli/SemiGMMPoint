# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
warnings.filterwarnings('ignore')
class SplitShapeNet():
    def __init__(self) -> None:
        self.root = ""
        self.meta = {}
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        # print(self.classes_original)

        class_choice = None
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)
    def main(self,split = "train"):

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([d for d in json.load(f)])
        print(train_ids)
        train_ids = np.array(list(train_ids))
        choice50u = np.random.choice(len(train_ids), len(train_ids)//2, replace=False)
        choice50l = []
        for idx in range(len(train_ids)):
            if(idx not in choice50u):
                choice50l.append(idx)
        choice50l = np.array(choice50l)
        print(choice50u)

        choice1lIdx = np.random.choice(len(choice50l), len(choice50l)//50, replace=False)
        choice1l = choice50l[choice1lIdx]
        
        choice5lIdx = np.random.choice(len(choice50l), len(choice50l)//10, replace=False)
        choice5l = choice50l[choice5lIdx]
        choice10lIdx = np.random.choice(len(choice50l), len(choice50l)//5, replace=False)
        choice10l = choice50l[choice10lIdx]
        choice25lIdx = np.random.choice(len(choice50l), len(choice50l)//2, replace=False)
        choice25l = choice50l[choice25lIdx]
        
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_50u.json'), 'w') as f:
            # train_ids = set([d for d in json.load(f)])
            tmp = train_ids[choice50u]
            json.dump(list(tmp),f)
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_1l.json'), 'w') as f:
            # train_ids = set([d for d in json.load(f)])
            tmp = train_ids[choice1l]
            json.dump(list(tmp),f)
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_5l.json'), 'w') as f:
            # train_ids = set([d for d in json.load(f)])
            tmp = train_ids[choice5l]
            json.dump(list(tmp),f)
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_10l.json'), 'w') as f:
            # train_ids = set([d for d in json.load(f)])
            tmp = train_ids[choice10l]
            json.dump(list(tmp),f)
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_25l.json'), 'w') as f:
            # train_ids = set([d for d in json.load(f)])
            tmp = train_ids[choice25l]
            json.dump(list(tmp),f)

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list_50l.json'), 'w') as f:
            # train_ids = set([d for d in json.load(f)])
            tmp = train_ids[choice50l]
            json.dump(list(tmp),f)
        # print(choice1l.shape,choice5l.shape,choice10l.shape,choice25l.shape,choice50l.shape)
        

        print(len(train_ids))
        exit(0)
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

if __name__ =="__main__":
    splitClass = SplitShapeNet()
    splitClass.main()