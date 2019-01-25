## Mxnet 安装
``` shell
pip uninstall mxnet
pip install --pre mxnet-cu80 
pip install --pre mxnet-cu90 
```


## jupyter安装

jupyter notebook --generate-config

``` python
#在服务器上输入python，进入anaconda的编辑器中，输入下面的代码
from notebook.auth import passwd
passwd()
#就会提示输入两次密码
# 输出的是一个秘钥
#sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
```

### Jupyter配置 
vi ~/.jupyter/jupyter_notebook_config.py
+ c.NotebookApp.password = u'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed' 
+ c.NotebookApp.ip = '*'  
+ c.NotebookApp.open_browser = False 
+ c.NotebookApp.port = 8888 
+ c.NotebookApp.notebook_dir = u''
+ c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager' 

### 端口映射通过本地访问
ssh myserver -L 8888:localhost:8888
http://localhost:8888

## mxnet环境安装
``` shell
git clone https://github.com/mli/gluon-tutorials-zh
conda env create -f environment.yml
#conda env update -f environment.yml
source activate gluon
```


## notedown md编辑插件安装
```shell 
git clone https://github.com/mli/gluon-tutorials-zh
pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

## notedown jupyter md文档中文异常处理

+ UnicodeEncodeError: 'ascii' codec can't encode characters in position
+ 通过文件内容判断文件存在出问题

>  vi lib/python3.6/site-packages/notedown/main.py
``` python
 def convert(content, informat, outformat, strip_outputs=False):
     try:  #add
         flag=os.path.exists(content)  #add
     except Exception as E:  #add
         flag=False  #add
     if flag:  #modify
         with io.open(content, 'r', encoding='utf-8') as f:
             contents = f.read()
     else:
         contents = content
```

```{.python .input}

```
