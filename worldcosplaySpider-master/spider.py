# coding=utf-8
import json
import requests
import os

ROOT_GALLERY_NAME = 'cos_top_200'

# @创建gallery文件夹
# @input:GALLERY_NAME gallery保存的文件夹
# @output:
def mkdir(GALLERY_NAME):
    GALLERY_NAME = ROOT_GALLERY_NAME + '\\' + GALLERY_NAME
    GALLERY_NAME = GALLERY_NAME.strip()
    GALLERY_NAME = GALLERY_NAME.rstrip("\\")

    if not os.path.exists(GALLERY_NAME):  # 如果不存在则创建目录
        print(GALLERY_NAME + ' Success')   # 创建目录操作函数
        os.makedirs(GALLERY_NAME)
        return True
    else:  # 如果目录存在则不创建，并提示目录已存在
        print(GALLERY_NAME + ' existence')
        return False

def download_character(character_id, name,index=0, page=1, ):
    url = 'https://worldcosplay.net/api/photo/list?sort=good_cnt&limit=24&page=' + str(page) + '&character_id=' + character_id 
    data = requests.get(url).content.decode('utf-8')   
    data = json.loads(data)
    if data['has_error'] != 0:
        print ('有问题' + character_id)
        return

    photo_data_list = data['list']
    for photo_data in photo_data_list:
        url = photo_data['photo']['sq300_url']
        file_local_url = ROOT_GALLERY_NAME + '\\' + name + '\\' + str(index) + '.jpg'
        print(file_local_url)
        try:
            pic = requests.get(url, stream=True, timeout=12)
            index += 1
            if os.path.exists(file_local_url):
                print('pic has been downloaded!')
                continue
            else:
                print('pic is downloaded, start to writing to local ')    
                # 推荐使用witho open,避免忘记进行fp.close()操作，buffering设置是为了IO加速
                with open(file_local_url, 'wb',buffering = 4*1024) as fp:
                    fp.write(pic.content) #写入file_local_url内容到图片
                    fp.flush()  
        except Exception:
            print(u'这张图片下载出问题了： %s' % url)
    page += 1
    print(index)
    if (index < 500):
        download_character(character_id, name, page=page, index=index)
    else: 
        file_write_obj = open("character_list.txt", 'w+') 
        line = character_id
        file_write_obj.writelines(line)  
        file_write_obj.write('\n')  
    return
   

def main(page=1, index=0):
    url = 'https://worldcosplay.net/api/ranking/character?sort=post_cnt&limit=200'
    data = requests.get(url).content.decode('utf-8')   
    data = json.loads(data)
    if data['has_error'] != 0:
        print (u'接口出错了')
        exit(1)
    file_write_obj = open("character_list.txt", 'r+') 
    photo_data_list = data['list']
    lines = file_write_obj.readlines() #读取所有行
    last_line = lines[-1] #取最后一行 
    last_id = last_line.split(',')[0]
    flag = True
    for photo_data in photo_data_list:
        character_id = str(photo_data['character']['id'])
        if (character_id == last_id):
            flag = False
            file_write_obj.write('\n')  
        if (flag):
            continue
        name = photo_data['character']['name']
        line = character_id + ',' + name
        file_write_obj.writelines(line)  
        file_write_obj.write('\n')  
        #新建用户文件夹
        mkdir(name)
        download_character(character_id, name)

if __name__ == '__main__':
    
    main()