# Doc for mtphotos

## 不建议用于正式环境,这个程序仅用于测试！！！
## 不建议用于正式环境,这个程序仅用于测试！！！
## 不建议用于正式环境,这个程序仅用于测试！！！

## 使用

- 使用除了搜索外的其他功能前建议点击login按钮
- 额外依赖：`pip install loguru`
- 首先复制并编辑配置文件`config.yaml`
- `python webui2.py`，访问`http://<ip>:7860/`即可


## conf说明

```yaml
db:
  dburl: postgresql://dev:dev@<dbip>:<dbport>/dev
devdb:
  dburl: postgresql://dev:dev@<dbip>:<dbport>/dev
model:
  batch_size: 4 # 每一批次的图片数量
  device: cpu # cpu or cuda:<num>
  model_name: OFA-Sys/chinese-clip-vit-large-patch14-336px # 模型名称
mtphotos:
  access_token: str # 占位留用校验格式 不用管
  admin_access_token: str # 占位留用校验格式 不用管
  admin_auth_code: str # 占位留用校验格式 不用管
  admin_password: admin_password # 需要管理员账号列出全部文件夹，不需要所有图库权限
  admin_username: admin_username # 需要管理员账号列出全部文件夹，不需要所有图库权限
  auth_code: str # 占位留用校验格式 不用管
  emb_folders: '[{''id'': ''487'', ''path'': ''/photos-1/dev''}]' # 选择需要的root文件夹，webui中点击get_folders按钮获取，复制到这里, 注意格式，需要重启服务
  password: dev # 任意账号密码，不需要所有图库权限
  username: dev # 任意账号密码，不需要所有图库权限
  weburl: http://<mtphotosip>:<port>/

```