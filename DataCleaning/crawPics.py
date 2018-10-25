# encoding: utf-8

import requests
import urllib.request
from bs4 import BeautifulSoup
import os
import threading
import threadpool

class BatchPicsCrawler:

    def __init__(self):
        self.PAGE_URL_LIST = []
        self.gLock = threading.Lock()

        self.batch_size=3
        self.pool=threadpool.ThreadPool(self.batch_size)
        print("init pics crawler")


    # 获取每一页的图片链接
    def get_page_pics(self,page_url):

        filename=page_url.split("/").pop()
        response = requests.get(page_url)
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        img_list = soup.select('img')

        [print(img) for img in img_list]
        count=0
        for img in img_list:
            url = img['src']
            path = os.path.join('images', filename+"-"+str(count)+".jpg")
            res=urllib.request.urlretrieve(url, filename=path)
            count+=1

    def batchDownloader(self):

        for i in range(0,len(self.PAGE_URL_LIST)-self.batch_size,self.batch_size):
            works=self.PAGE_URL_LIST[i:i+self.batch_size]
            requets=threadpool.makeRequests(self.get_page_pics,works)
            [self.pool.putRequest(req) for req in requets]
            self.pool.wait()
            print("finished {} batches".format((i+self.batch_size)//self.batch_size))


def main():
    bpc=BatchPicsCrawler()

    bpc.get_page_pics("http://view.inews.qq.com/a/20171213A09JBV00");exit(2)#test one page request

    from DataCleaning.DataLoader import NewsDataset
    nwdata=NewsDataset("data\\trainex.xls")
    extrurl = lambda data: data.col_values(2, start_rowx=1)
    urls=extrurl(nwdata.data[0])
    bpc.PAGE_URL_LIST=urls[:]
    bpc.batchDownloader()
    pass


if __name__ == "__main__":
    main()








