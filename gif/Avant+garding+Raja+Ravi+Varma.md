
<img src="Notebook_images/Ravivarma1.png">
Raja Ravi Varma [https://en.wikipedia.org/wiki/Raja_Ravi_Varma] is one of the most celebrated and divisive figures in late 19th century / early 20th century Indian art.

On one front, he is praised as the first *artist of the masses* known for his pioneering efforts in marrying Indian artistic vibes and mythology with European techniques. The grand doyen of Indian literarture, Rabindranath Tagore, said these words about Ravi Varma:
<img src="Notebook_images/tagore.png">
His paintings and lithographs are a commonplace through the length and breadth of India.
Some of the most popular examples of his art are as shown in the collage below. (Sourced from https://commons.wikimedia.org/wiki/Raja_Ravi_Varma)
<img src="Notebook_images/Ravivarma_collage.png">
On the other hand, the palpably naive flavor of realism espoused in his works are widely  seen amongst critics as being derivative, bland and too inspired by Western academic constructs for one's liking. To a savant of the stature of Aurobindo, he was 'the grand debaser of Indian taste and artistic culture'. (Source:https://www.aurobindo.ru/workings/sa/37_01/0102_e.htm)

As a response to his growing clout, The Bengal School of Art (https://en.wikipedia.org/wiki/Bengal_School_of_Art) emerged as the avant garde nationalistic art-movement that sought to develop a more homegrown hue of art that was philosophically and artistically distinct from the European academic art.
It was via this movement that Abanindranath Tagore,<img src="Notebook_images/A_tagore.jpg"> (who also happened to be the nephew of Rabindranath Tagore), emerged as a leading figure. Combining the home-grown Moghul and Rajput styles with orientalist sensibilities, he ushered in a sea change in the style of depictions of the human and divine subjects derived out Indian bodies of literature.
Some of the most enduring examples of his art-form are depicted in this collage below;
(Sourced from https://www.wikiart.org/en/abanindranath-tagore):
<img src="Notebook_images/A_Tagore_collage.png"> 


In this work, I aspire to harness the fast, arbitrary artistic neural style transfer technique of [1], to sample images from Raja Ravi Varma's body of work and *transfer* it to the avant garde Bengal school of art style captured in Abanindranath Tagore's paintings.
I used the open-sourced implementation of [1] that is part of the 'Magenta' project available at [2].

## References:

[1] https://arxiv.org/pdf/1705.06830.pdf


[2] https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization

# Step-1 : Resizing and squarization of images:


```python
from PIL import Image
from resizeimage import resizeimage
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
# Crucial for large images:
# Source: https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
```


```python
# Option: -1:
# def resize_image(rv_folder,file_name,n_pixels=256):
#     with open(os.path.join(rv_folder,file_name), 'r+b') as f:
#         with Image.open(f) as image:
#             cover = resizeimage.resize_contain(image, [n_pixels, n_pixels])
#             cover.save(os.path.join(rv_folder,str(n_pixels)+'_'+file_name), image.format)
from PIL import Image

def image_resize(file_in,n_pixels,dir_in):
    file_path=os.path.join(os.getcwd(),dir_in,file_in)
    im = Image.open(file_path)
    im_resized=im.resize((n_pixels,n_pixels),Image.LANCZOS)
    file_out=file_in.strip('.jpg')+ '_'+str(n_pixels)+'.jpg'
    dir_out=dir_in+'_'+str(n_pixels)
    file_out_path=os.path.join(os.getcwd(),dir_out,file_out)
    im_resized.save(file_out_path, "JPEG")
```

# Source Images:


```python
rv_folder=os.path.join(os.getcwd(),'RRV')
RV_image_list=os.listdir(rv_folder)
RV_image_list
```




    ['Raja_Ravi_Varma,_Expectation.jpg',
     'Raja_Ravi_Varma,_Kadambari.jpg',
     'Raja_Ravi_Varma,_Ladies_in_the_moonlight.jpg',
     'Raja_Ravi_Varma,_Malabar_Lady.jpg',
     'Raja_Ravi_Varma,_Nair_Lady_Adorning_Her_Hair.jpg',
     'Raja_Ravi_Varma,_Saraswati_1896.jpg',
     'Raja_Ravi_Varma,_The_suckling_child.jpg',
     'Raja_Ravi_Varma,_Woman_with_veena.jpg']



# Style images:


```python
at_folder=os.path.join(os.getcwd(),'AT')
AT_image_list=os.listdir(at_folder)
AT_image_list
```




    ['Abanindranath_Tagore_-Ganesh_Janani.jpg',
     'Abanindranath_Tagore_-_ Bharat Mata_1905.jpg',
     'Abanindranath_Tagore_-_Journeys_End_-_Google_Art_Project.jpg',
     'Abanindranath_Tagore_-_My_Mother_-_Google_Art_Project.jpg']




```python
for img_file in AT_image_list:
    image_resize(img_file,n_pixels=512,dir_in='AT')
for img_file in RV_image_list:
    image_resize(img_file,n_pixels=512,dir_in='RRV')
```

```
1: bash
2: Source activate magentapython
3:
python arbitrary_image_stylization_with_weights.py \
  --checkpoint=./arbitrary_style_transfer/model.ckpt \
  --output_dir=output_imgs \
  --style_images_paths=images/style_images/*.jpg \
  --content_images_paths=images/content_images/*.jpg \
  --image_size=512 \
  --content_square_crop=True \
  --style_image_size=512 \
  --style_square_crop=True \
  --logtostderr
```

The output images are in the 'output_imgs' folder:
```
output_image_list=os.listdir(os.path.join(os.getcwd(),'output_imgs'))
gif_dir=os.path.join(os.getcwd(),'gif')

###  Script for generating the gif:
import imageio
with imageio.get_writer(gif_dir, mode='V') as writer:
    for filename in output_image_list:
        file_loc=os.path.join(os.getcwd(),'output_imgs',filename)
        image = imageio.imread(file_loc)
        writer.append_data(image)
```

# Highlights:

## 1: Transfer of style in Motherhood.

2 of the iconic images that resulted from the artists co-exploring the exalted theme of motherhood were:
- *The suckling child* by Raja Ravi Varma
<img src="RRV/Raja_Ravi_Varma,_The_suckling_child.jpg"> 
- *Ganesh Janani* by  Abanindranath Tagore
<img src="AT/Abanindranath_Tagore_-Ganesh_Janani.jpg">
Using [2], the image I got that I felt aptly amalgamated the artistic sensibilities of the 2 colossal figures being explored in this experiment was:
<img src="output_imgs/Raja_Ravi_Varma,_The_suckling_child_512_stylized_Abanindranath_Tagore_-Ganesh_Janani_512_0.jpg"> 
One startling aspect of this style transfer was the emergence of the variation in the eyes of the child:
<img src="output_imgs/child_eyes.png"> 

## 2: Transfer of style in Feminine representation of divinity/nationhood:
Again as with highlight-1 above, we sample 2 of the iconic images that resulted from the artists co-exploring feminine representation of divinity/nationhood.

- *Saraswati*  (circa 1896) by Raja Ravi Varma
<img src="RRV/Raja_Ravi_Varma,_Saraswati_1896.jpg"> 
- *Bharat Mata* by  Abanindranath Tagore
<img src="output_imgs/Abanindranath_Tagore_-_ Bharat Mata_1905_512.jpg">
Again, using [2], the image I got was:
<img src="output_imgs/Raja_Ravi_Varma,_Saraswati_1896_512_stylized_Abanindranath_Tagore_-_ Bharat Mata_1905_512_0.jpg"> 


# Art animation showcasing avant-garding of all of Raja Ravi Varma's images using A.Tagore's style:


```python
from IPython.display import HTML

HTML("""
<video width="320" height="240" controls>
  <source src="gif/combined.mp4" type="video/mp4">
</video>
""")
```





<video width="320" height="240" controls>
  <source src="gif/combined.mp4" type="video/mp4">
</video>



