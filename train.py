import paddle

# 指定需要迁移的代码目录或文件路径
paddle.utils.cpp_extension.upgrade_from_fluid("my_project/")