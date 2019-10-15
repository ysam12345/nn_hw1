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

def drawPic(train_result_list):
    #清空影象，以使得前後兩次繪製的影象不會重疊
    drawPic.f.clf()
    drawPic.a=drawPic.f.add_subplot(111)
    
    print(train_result_list)

    acc_list = [i['acc'] for i in train_result_list]
    print(acc_list)
    #繪製這些隨機點的散點圖，顏色隨機選取
    drawPic.a.plot(acc_list)
    drawPic.a.set_title('Demo: acc_list')
    drawPic.canvas.draw()

def drawPic1():
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

    # split traning data and testing data
    train_df=df.sample(frac=0.666666)
    test_df=df.drop(train_df.index)

    train_dfs = np.split(train_df, [len(train_df.columns)-1], axis=1)
    train_X_df = train_dfs[0]
    train_y_df = train_dfs[1]
    train_X = train_X_df.values.tolist()
    train_y = train_y_df.values.reshape(-1,).tolist()

    test_dfs = np.split(test_df, [len(test_df.columns)-1], axis=1)
    test_X_df = test_dfs[0]
    test_y_df = test_dfs[1]
    test_X = test_X_df.values.tolist()
    test_y = test_y_df.values.reshape(-1,).tolist()

    learning_rate = 0.8
    n = Neural(train_X, train_y, learning_rate)
    train_result_list = []
    print("### training start ###")
    for i in range(10):
        train_result = n.train()
        train_result_list.append(train_result)
    print("### training end ###")
    drawPic(train_result_list)

    print("### predict start ###")
    n.predict(test_X, test_y)
    print("### predict end ###")



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
	Button(root,text='畫圖',command=drawPic).grid(row=1,column=1,columnspan=1)
	Button(root,text='open file',command=select_file).grid(row=1,column=2,columnspan=1)
       
    #啟動事件迴圈
	root.mainloop()