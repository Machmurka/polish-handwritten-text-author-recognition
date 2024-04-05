"""
Contains functionality for creating PyTorch DataLoaders for 
Handwritten text author recognition classification data.
"""

import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from collections import Counter
from tqdm.auto import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader, random_split


class RawData:
    '''
        Przekształcanie poszczególnych słów autora z tekstówa na 
    '''
    def __init__(self) -> None:
        pass

    def count_word_occurrences(self,AuthorText:str)->dict:
        '''
        Zliczenie poszczególnych słów w tekście dla pojedyńczego authora 

        Args: 
            AuthorText: text autora (str) po użyciu funkcji .open() .read()

        Returns:
            dict ze słowem(str) i liczbą wystapienia w tekście(int)
            
            {'ala': 1, 'w': 41,...}
        '''
        
        words=[]
        text = AuthorText
        lines = text.split('\n')
        number_of_lines = lines.__len__() - 1

        for i in range(number_of_lines):
            row_values = lines[i].split()
            if len(row_values) != 0 and row_values[0] != '%':
                words.append(row_values[1])


        word_counts=Counter(words)
        
        return word_counts
    
    def filter_words(self, AuthorText: str) -> dict:
        '''
        Zliczenie poszczególnych słów w tekście które występują więcej niż 1 raz dla pojedyńczego authora 


        Args: 
            AuthorText: text autora (str) po użyciu funkcji .open() .read()

        Returns:
            dict ze słowem(str) i liczbą wystapienia w tekście(int) >1 
            
            {'na': 46, 'w': 41,...}
        '''

        word_counts = self.count_word_occurrences(AuthorText)
        return {word: count for word, count in word_counts.items() if count > 1}
    
    def save_word_to_bmp(self, subimage, filename)->None:
        '''
        Zapis słowa do pliku png w folderze autor/slowa 

        Arg:
            subimage: wycinek słowa ze zdjęcia wykorzystując image[row1:row2,column1:column2]
            filename: Nazwa pod jaką słowo ma być zapisane

        '''

        if os.path.exists(filename+'.bmp'):
            print(f"File {filename} already exists.")
        else:
            try:

                mpimg.imsave(filename+'.bmp',subimage)
            except SystemError:
                a=10
                # print("An error occurred while saving the image. Skipping...")

                



        

    def save_words_to_file(self,Num_of_authors:int):
        ''''
            Zapisywanie w poczególnych folderach autora słów w formie (BMP/PNG do przedyskutowania )!!!
            PATRZ: https://www.adobe.com/pl/creativecloud/file-types/image/comparison/bmp-vs-png.html  
            
            Nazwa pliku : słowo + wystąpienie słowa w tekście ( Użycie count_word_occurrences() w celu ówczesnego zliczenia wystąpienia słowa )

            Arg:
                Num_of_authors: Liczba autorów licząc od 0 dla których słową będą zapisane 

            
        '''
        
        # System odczytu pliku zostaje taki sam jak w gotowym pliku word_display.py
        Num_of_authors

        for author_no in range(Num_of_authors):

            file_desc_name = "Data/author" + str(author_no + 1) + "/word_places.txt"
            file_desc_ptr = open(file_desc_name, 'r')
            text = file_desc_ptr.read()
            lines = text.split('\n')
            number_of_lines = lines.__len__() - 1
            row_values = lines[0].split()
            # number_of_values = row_values.__len__()
            
            # Będziemy potrzebować słów które występują więcej niż 1 raz 
            words=self.filter_words(text)

            image_file_name_prev = ""
            for i in tqdm(range(number_of_lines)):
                row_values = lines[i].split()
                # print(len(row_values))
                if len(row_values) != 0 and row_values[0] != '%' and len(row_values)==6:
                    # number_of_values = len(row_values)
                    image_file_name = "Data/author" + str(author_no + 1) + "\\" + row_values[0][1:-1]

                    if image_file_name != image_file_name_prev:   
                        image = mpimg.imread(str(image_file_name))
                        image_file_name_prev = image_file_name



                    word = row_values[1]
                    word = word.replace('/', '_')
                    word = word.replace('"',' ')
                    word = word.replace('?',' ')

                    if word[0]=='<' or word=='||' or word[0]=='-' or word[-1]=='\\': continue
                    if word=='com': word+='_'

                    # Zmiana nazwy na słowo+wystąpienie 
                    if word in words.keys():
                        words[word]-=1
                        word= word+str(words[word]) if words[word] > 0 else word
                       
                        # print(word)
                        # print(f"\nbefore\nword: {word}, count: {words[word]}")
                        # print(f"after \n word: {word}, count: {words[word]}")

                    filename="Data/Words/author" + str(author_no + 1)+"/" +word
                

                    if os.path.exists(filename+".bmp") == False :
                        
                        row1, column1, row2, column2 = abs(int(row_values[2])), abs(int(row_values[3])), \
                            int(row_values[4]), int(row_values[5])
                        subimage = image[row1:row2,column1:column2] 
                        self.save_word_to_bmp(subimage,filename)
                    
                    


                        

                    # plt.title("Author "+str(author_no+1)+", image = "+row_values[0][1:-1]+", word = "+word)
                    # plt.xlabel("X")
                    # plt.ylabel("Y")
                    # plt.imshow(subimage)
                    # plt.show()


                    

        file_desc_ptr.close()
        
        
class AuthorImagesDataset:
    def __init__(self, root_dir, batch_size:int,DataProcent:float,transform=None):
        self.BATCH_SIZE=batch_size
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)

        subset_length = int(len(self.dataset) * DataProcent)
        rest_length = len(self.dataset) - subset_length
        self.subset_data, _ = random_split(self.dataset, [subset_length, rest_length])


        test_length = int(len(self.subset_data) * 0.2)
        train_length = len(self.subset_data) - test_length

        # Split the dataset
        self.train_data, self.test_data = random_split(self.subset_data, [train_length, test_length])
        
        self.into_data_loaders()

    def into_data_loaders(self):


        self.train_dataloader= DataLoader(dataset=self.train_data, # use custom created train Dataset
                                     batch_size=self.BATCH_SIZE, # how many samples per batch?
                                    #  num_workers=NUM_WORKERS, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

        self.test_dataloader = DataLoader(dataset=self.test_data, # use custom created test Dataset
                                    batch_size=self.BATCH_SIZE, 
                                    # num_workers=NUM_WORKERS, 
                                    shuffle=False) # don't usually need to shuffle testing data
        
        
        # word,label=next(iter(self.test_dataloader))
        # print(f"shape of dataloader {word.shape} \n and label {label}")

        
    
    def __len__(self):
        return len(self.dataset) 
            


# if __name__=='__main__':
#     t=RawData()
#     t.save_words_to_file(8)
#     s=AuthorImagesDataset(r'Data/Words',transform=transforms.Compose([
#             transforms.Resize(size=(64,64)),
#             transforms.TrivialAugmentWide(num_magnitude_bins=31),
#             transforms.ToTensor()
#         ]))
    

def create_dataloaders(
        DatasetDir:str,
        transform: transforms.Compose, 
        batch_size: int,
        DataProcent:float
):
    # t=RawData()
    # t.save_words_to_file(8)
    s=AuthorImagesDataset(DatasetDir,batch_size,DataProcent,transform)
    
    return s.train_dataloader,s.test_dataloader,s.dataset.classes