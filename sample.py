import os
import pandas as pd

path = "mnist_png/testing/"
dir_list = ['0', '1']
img_list = []
tag_list = []
result = pd.DataFrame()


for i in range(len(dir_list)):
	d = dir_list[i]
	file_list = os.listdir(path + d)
	img_list += file_list
	tag_list += d * len(file_list)

	
test_list = pd.DataFrame(
	{
		'img_file' : img_list,
		'tag' : tag_list
	})

test_list.to_excel('file_list.xlsx', encoding = 'UTF8')
