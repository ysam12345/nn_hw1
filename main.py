#!/usr/bin/env python
#coding:utf-8
import numpy as np
from tkinter import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.filedialog import askopenfilename
import pandas as pd
from neural import Neural

file_path = ""

def drawPic():
    try:sampleCount=int(inputEntry.get())
    except:
        sampleCount=50
        print('請輸入整數')
        inputEntry.delete(0,END)
        inputEntry.insert(0,'50')
       
    #清空影象，以使得前後兩次繪製的影象不會重疊
    drawPic.f.clf()
    drawPic.a=drawPic.f.add_subplot(111)
       
    #在[0,100]範圍內隨機生成sampleCount個數據點
    x=np.random.randint(0,100,size=sampleCount)
    y=np.random.randint(0,100,size=sampleCount)
    color=['b','r','y','g']
       
    #繪製這些隨機點的散點圖，顏色隨機選取
    drawPic.a.scatter(x,y,s=3,color=color[np.random.randint(len(color))])
    drawPic.a.set_title('Demo: Draw N Random Dot')
    drawPic.canvas.draw()

def select_file():
    filename = askopenfilename()
    print(filename)
    file_path = filename
    df = pd.read_table(filename, sep=" ", header=None)
    print(df)
    dfs = np.split(df, [len(df.columns)-1], axis=1)
    X_df = dfs[0]
    y_df = dfs[1]
    X = X_df.values.tolist()
    y = y_df.values.reshape(-1,).tolist()
    learning_rate = 0.8
    n = Neural(X, y, learning_rate)
    for i in range(10):
        n.train()

if __name__ == '__main__':    
	
	matplotlib.use('TkAgg')
	root = Tk()  
    #在Tk的GUI上放置一個畫布，並用.grid()來調整佈局
	drawPic.f = Figure(figsize=(5,4), dpi=100)

	drawPic.canvas = FigureCanvasTkAgg(drawPic.f, master=root) 
	drawPic.canvas.draw() 
	drawPic.canvas.get_tk_widget().grid(row=0, columnspan=3)    
    
    #放置標籤、文字框和按鈕等部件，並設定文字框的預設值和按鈕的事件函式
	Label(root,text='請輸入樣本數量：').grid(row=1,column=0)
	inputEntry=Entry(root)
	inputEntry.grid(row=1,column=1)
	inputEntry.insert(0,'50')
	Button(root,text='畫圖',command=drawPic).grid(row=1,column=2,columnspan=3)
	Button(root,text='open file',command=select_file).grid(row=1,column=2,columnspan=3)
       
    #啟動事件迴圈
	root.mainloop()