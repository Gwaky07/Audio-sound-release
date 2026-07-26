"""可安装的仓库脚本兼容包。

新代码应优先放在 ``audio_sound`` 包内；保留此包是为了让历史兼容模式在
wheel、editable install 和 ``git archive`` 安装后仍能导入现有脚本模块。
"""
