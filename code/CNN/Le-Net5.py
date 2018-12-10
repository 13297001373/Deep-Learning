with open('./config.txt','wb') as f:
    for i in range(10):
        f.writelines(i)
