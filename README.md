# classification_task
### 分类任务工程说明

##### 目录

+ │  pipeline.conf

  │  tools.py
  
  │  train.py

  ├─data
  
  │  ├─dev
  
  │  │      train_pipeline.conf 
  
  │  │      
  
  │  └─raws
  
  │          merge3_v1.json
  
  │          
  ├─dataset
  
  │  │  transformer_series.py
  
  │          
  │      
  ├─model
  
  │      bert_linear.py
  
  │      
  └─utils
  \
          dataset_merge_and_shuffle.py
          mysql_util.py
          samples_generate.py
          _utils.py


+  [0] pipeline.conf 说明

  + ```
    [user_config]
    project_name=classification_task
    labels_set=type1:其他,乳头状瘤,囊腺瘤,小细胞癌,炎性假瘤,畸胎瘤,神经内分泌肿瘤,类癌,肉瘤样癌,胸腺瘤,腺瘤,腺癌,腺鳞癌,错构瘤,鳞癌 degree1:其他,中低分化,中分化,低分化,未分化,高中分化,高分化
    train_raw_path=/nlp_data/qinye/Pathology/data/raws/merge3_v1.json
    train_config_path=/nlp_data/qinye/Pathology/data/dev/train_pipeline.conf
    host=192.168.3.228
    user=root
    password=root
    database=deepwise_single_disease
    port=3307
    charset=utf8
    tablename=Pathology
    sql_words=id,Exam_Result
    id=id
    value=stand_sign
    ```

  + ```
    project_name:工程名字，必须和train.py所在文件夹的名字一样
    labels_set:应为可能需要对同一段文本多次分类，所以需要给出多词分类的标签集合，每个集合中必须包含“其他”。格式是  类别集合1:标签1,标签2,, 类别集合2:标签1,标签2,    ps:类别集合之间用空格隔开，类别集合内部用英文逗号隔开，类别集合为英文。
    train_raw_path:原始文件的路径，需要将训练的文件放入该路径上
    train_config_path:可以不用修改，代码中会自动替换
    host:mysql的ip地址
    user:mysql登录用的用户名
    password:mysql登录用的密码
    database:mysql数据库名
    port:mysql端口号
    charset:mysql编码格式
    tablename:mysql数据库表名
    sql_word:mysql数据库查询使用的字段，只能包含两个，其中第一个是唯一标识的字段，第二个为输入模型的文本数据
    id:mysql唯一标识的字段
    value:需要修改的目标输出字段
    ```

+ [1]json数据格式说明

  + 每行为一个字典（dict）对象，字典对象包含两个键“text”和“label”,其中“text”为字符串（str）,"label"为列表对象（list）,列表对象中每个元素是一个字符串的类别标签名字。

+ [2]train.py文件说明

  + 包含 PreProcess,Train,Test,PostProcess，Pipeline一共五个函数
  + PreProcess:对原始数据进行标签和数据集的划分，在dataset下生成对应标签类型的目录。
  + Train,Test训练和测试，测试在pred目录下生成对数据库表的测试结果的np文件和转换为对应标签名的json文件
  + PostProcess：对模型跑出的结果后处理，加入一些规则限制。

#### 流程

+ source activate qEnv,切到train.py的目录下
+ 根据需要修改pipeline.conf和train.py下的PostProcess函数
+ 运行python train.py
